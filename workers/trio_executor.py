"""
Trio-based structured concurrency executor for video generation jobs.

This module provides structured concurrency using Trio task groups and cancellation scopes
for efficient resource management and proper job cancellation handling.
"""

import logging
import os
import signal
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Awaitable

import trio
import trio.lowlevel

from api.models import JobStatus
from api.queue import get_job_queue

logger = logging.getLogger(__name__)


class TrioCancellationToken:
    """Trio-based cancellation token for cooperative cancellation"""
    
    def __init__(self, job_id: str, cancel_scope: trio.CancelScope):
        self.job_id = job_id
        self.cancel_scope = cancel_scope
        self._cancelled = False
        self._cleanup_callbacks: List[Callable[[], None]] = []
    
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested"""
        return self._cancelled or self.cancel_scope.cancelled_caught
    
    def cancel(self):
        """Request cancellation"""
        self._cancelled = True
        self.cancel_scope.cancel()
        logger.info(f"Cancellation requested for job {self.job_id}")
    
    def add_cleanup_callback(self, callback: Callable[[], None]):
        """Add a cleanup callback to be called on cancellation"""
        self._cleanup_callbacks.append(callback)
    
    def cleanup(self):
        """Execute cleanup callbacks"""
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Cleanup callback failed for job {self.job_id}: {e}")
        
        self._cleanup_callbacks.clear()


class ProcessManager:
    """Manages subprocess lifecycle with Trio-based cancellation"""
    
    def __init__(self, job_id: str, cancellation_token: TrioCancellationToken):
        self.job_id = job_id
        self.cancellation_token = cancellation_token
        self.processes: List[subprocess.Popen] = []
        self._nursery: Optional[trio.Nursery] = None
    
    async def start_process(
        self, 
        cmd: List[str], 
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 300.0
    ) -> subprocess.Popen:
        """
        Start a subprocess with cancellation support
        
        Args:
            cmd: Command and arguments
            cwd: Working directory
            env: Environment variables
            timeout: Process timeout in seconds
            
        Returns:
            Subprocess instance
            
        Raises:
            trio.Cancelled: If cancelled
            subprocess.SubprocessError: If process fails
        """
        logger.info(f"Starting process for job {self.job_id}: {' '.join(cmd)}")
        
        # Check for cancellation before starting
        if self.cancellation_token.is_cancelled():
            raise trio.Cancelled()
        
        try:
            # Start process
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes.append(process)
            
            # Add cleanup callback for this process
            def cleanup_process():
                self._terminate_process(process)
            
            self.cancellation_token.add_cleanup_callback(cleanup_process)
            
            # Monitor process with timeout
            await self._monitor_process(process, timeout)
            
            return process
            
        except trio.Cancelled:
            # Clean up on cancellation
            if 'process' in locals():
                self._terminate_process(process)
            raise
        except Exception as e:
            logger.error(f"Failed to start process for job {self.job_id}: {e}")
            raise
    
    async def _monitor_process(self, process: subprocess.Popen, timeout: float):
        """Monitor process execution with cancellation support"""
        start_time = trio.current_time()
        
        while process.poll() is None:
            # Check for cancellation
            if self.cancellation_token.is_cancelled():
                self._terminate_process(process)
                raise trio.Cancelled()
            
            # Check for timeout
            if trio.current_time() - start_time > timeout:
                self._terminate_process(process)
                raise subprocess.TimeoutExpired(process.args, timeout)
            
            # Sleep briefly to avoid busy waiting
            await trio.sleep(0.1)
        
        # Check exit code
        if process.returncode != 0:
            stdout, stderr = process.communicate()
            error_msg = f"Process failed with exit code {process.returncode}"
            if stderr:
                error_msg += f": {stderr}"
            raise subprocess.CalledProcessError(process.returncode, process.args, stdout, stderr)
    
    def _terminate_process(self, process: subprocess.Popen):
        """Terminate a process gracefully with SIGTERM -> SIGKILL escalation"""
        if process.poll() is not None:
            return  # Process already terminated
        
        try:
            logger.info(f"Terminating process {process.pid} for job {self.job_id}")
            
            # Send SIGTERM first
            process.terminate()
            
            # Wait up to 5 seconds for graceful shutdown
            try:
                process.wait(timeout=5.0)
                logger.info(f"Process {process.pid} terminated gracefully")
                return
            except subprocess.TimeoutExpired:
                pass
            
            # Escalate to SIGKILL if needed
            logger.warning(f"Process {process.pid} did not terminate gracefully, sending SIGKILL")
            process.kill()
            
            # Wait for final termination
            try:
                process.wait(timeout=2.0)
                logger.info(f"Process {process.pid} killed successfully")
            except subprocess.TimeoutExpired:
                logger.error(f"Process {process.pid} could not be killed")
                
        except Exception as e:
            logger.error(f"Error terminating process {process.pid}: {e}")
    
    def cleanup_all_processes(self):
        """Clean up all managed processes"""
        for process in self.processes:
            self._terminate_process(process)
        
        self.processes.clear()


class TrioJobExecutor:
    """Trio-based job executor with structured concurrency"""
    
    def __init__(self):
        self.active_jobs: Dict[str, TrioCancellationToken] = {}
    
    async def execute_job(
        self, 
        job_id: str, 
        job_function: Callable[[str, TrioCancellationToken, ProcessManager], Awaitable[str]],
        timeout: float = 3600.0
    ) -> str:
        """
        Execute a job with structured concurrency and cancellation support
        
        Args:
            job_id: Job identifier
            job_function: Async function to execute the job
            timeout: Job timeout in seconds
            
        Returns:
            Job result (typically GCS URI)
            
        Raises:
            trio.Cancelled: If job is cancelled
            Exception: If job execution fails
        """
        logger.info(f"Starting Trio job execution for {job_id}")
        
        # Create cancellation scope for this job
        with trio.CancelScope() as cancel_scope:
            # Create cancellation token
            cancellation_token = TrioCancellationToken(job_id, cancel_scope)
            self.active_jobs[job_id] = cancellation_token
            
            try:
                # Set up timeout
                cancel_scope.deadline = trio.current_time() + timeout
                
                # Create process manager
                process_manager = ProcessManager(job_id, cancellation_token)
                
                # Add cleanup callback for process manager
                cancellation_token.add_cleanup_callback(process_manager.cleanup_all_processes)
                
                # Execute job in nursery for structured concurrency
                async with trio.open_nursery() as nursery:
                    # Store nursery reference for potential use
                    process_manager._nursery = nursery
                    
                    # Execute the job function
                    result = await job_function(job_id, cancellation_token, process_manager)
                    
                    logger.info(f"Job {job_id} completed successfully")
                    return result
                    
            except trio.Cancelled:
                logger.info(f"Job {job_id} was cancelled")
                
                # Update job status
                try:
                    job_queue = get_job_queue()
                    job_queue.update_job_status(job_id, JobStatus.CANCELED)
                    job_queue.add_job_log(job_id, "Job cancelled during execution")
                except Exception as e:
                    logger.error(f"Failed to update cancelled job status: {e}")
                
                raise
                
            except Exception as e:
                logger.error(f"Job {job_id} failed: {e}")
                
                # Update job status
                try:
                    job_queue = get_job_queue()
                    job_queue.update_job_status(job_id, JobStatus.FAILED, error=str(e))
                    job_queue.add_job_log(job_id, f"Job failed: {e}")
                except Exception as update_error:
                    logger.error(f"Failed to update failed job status: {update_error}")
                
                raise
                
            finally:
                # Cleanup
                cancellation_token.cleanup()
                self.active_jobs.pop(job_id, None)
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel an active job
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if cancellation was initiated, False if job not found
        """
        cancellation_token = self.active_jobs.get(job_id)
        if cancellation_token:
            cancellation_token.cancel()
            return True
        
        logger.warning(f"Cannot cancel job {job_id}: not found in active jobs")
        return False
    
    def get_active_jobs(self) -> List[str]:
        """Get list of active job IDs"""
        return list(self.active_jobs.keys())


class AsyncioTrioBridge:
    """Bridge for running asyncio code within Trio context"""
    
    @staticmethod
    async def run_asyncio_function(func: Callable, *args, **kwargs):
        """
        Run an asyncio function within Trio context
        
        This provides fallback mechanism for dependencies that require asyncio.
        
        Args:
            func: Asyncio function to run
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        import asyncio
        
        # Create asyncio event loop in thread
        def run_in_asyncio():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                if asyncio.iscoroutinefunction(func):
                    return loop.run_until_complete(func(*args, **kwargs))
                else:
                    return func(*args, **kwargs)
            finally:
                loop.close()
        
        # Run asyncio code in thread
        return await trio.to_thread.run_sync(run_in_asyncio)


@asynccontextmanager
async def trio_job_context(job_id: str):
    """
    Context manager for Trio job execution with proper cleanup
    
    Args:
        job_id: Job identifier
        
    Yields:
        Tuple of (cancellation_token, process_manager)
    """
    with trio.CancelScope() as cancel_scope:
        cancellation_token = TrioCancellationToken(job_id, cancel_scope)
        process_manager = ProcessManager(job_id, cancellation_token)
        
        # Add cleanup callback
        cancellation_token.add_cleanup_callback(process_manager.cleanup_all_processes)
        
        try:
            yield cancellation_token, process_manager
        finally:
            # Ensure cleanup happens
            cancellation_token.cleanup()


async def check_cancellation_async(cancellation_token: TrioCancellationToken, job_id: str) -> bool:
    """
    Async version of cancellation check with job status update
    
    Args:
        cancellation_token: Cancellation token
        job_id: Job identifier
        
    Returns:
        True if job was cancelled, False otherwise
    """
    if cancellation_token.is_cancelled():
        logger.info(f"Job {job_id} cancellation detected")
        
        try:
            job_queue = get_job_queue()
            job_queue.update_job_status(job_id, JobStatus.CANCELED)
            job_queue.add_job_log(job_id, "Job cancelled during processing")
        except Exception as e:
            logger.error(f"Failed to update cancelled job status for {job_id}: {e}")
        
        return True
    
    return False


async def cleanup_temp_files_async(temp_dir: str):
    """
    Async cleanup of temporary files
    
    Args:
        temp_dir: Temporary directory to clean up
    """
    def cleanup():
        try:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")
    
    # Run cleanup in thread to avoid blocking
    await trio.to_thread.run_sync(cleanup)


# Global executor instance
_trio_executor: Optional[TrioJobExecutor] = None


def get_trio_executor() -> TrioJobExecutor:
    """Get the global Trio executor instance"""
    global _trio_executor
    if _trio_executor is None:
        _trio_executor = TrioJobExecutor()
    return _trio_executor


def initialize_trio_executor() -> TrioJobExecutor:
    """Initialize the global Trio executor"""
    global _trio_executor
    _trio_executor = TrioJobExecutor()
    logger.info("Trio executor initialized")
    return _trio_executor