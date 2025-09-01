"""
Tests for Trio-based structured concurrency executor.

This module tests the Trio executor implementation including
task groups, cancellation scopes, and subprocess management.
"""

import asyncio
import os
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock

import pytest
import trio

from workers.trio_executor import (
    TrioCancellationToken,
    ProcessManager,
    TrioJobExecutor,
    AsyncioTrioBridge,
    trio_job_context,
    check_cancellation_async,
    cleanup_temp_files_async,
    get_trio_executor,
    initialize_trio_executor
)
from api.models import JobStatus


class TestTrioCancellationToken:
    """Test Trio cancellation token functionality"""
    
    @pytest.mark.trio
    async def test_cancellation_token_creation(self):
        """Test creating a cancellation token"""
        with trio.CancelScope() as cancel_scope:
            token = TrioCancellationToken("test_job", cancel_scope)
            
            assert token.job_id == "test_job"
            assert token.cancel_scope == cancel_scope
            assert not token.is_cancelled()
    
    @pytest.mark.trio
    async def test_cancellation_token_cancel(self):
        """Test cancelling a token"""
        with trio.CancelScope() as cancel_scope:
            token = TrioCancellationToken("test_job", cancel_scope)
            
            token.cancel()
            
            assert token.is_cancelled()
            assert cancel_scope.cancel_called
    
    @pytest.mark.trio
    async def test_cleanup_callbacks(self):
        """Test cleanup callback execution"""
        with trio.CancelScope() as cancel_scope:
            token = TrioCancellationToken("test_job", cancel_scope)
            
            callback_called = False
            
            def cleanup_callback():
                nonlocal callback_called
                callback_called = True
            
            token.add_cleanup_callback(cleanup_callback)
            token.cleanup()
            
            assert callback_called
    
    @pytest.mark.trio
    async def test_cleanup_callback_exception_handling(self):
        """Test that cleanup callback exceptions don't break cleanup"""
        with trio.CancelScope() as cancel_scope:
            token = TrioCancellationToken("test_job", cancel_scope)
            
            def failing_callback():
                raise Exception("Callback failed")
            
            def working_callback():
                nonlocal callback_called
                callback_called = True
            
            callback_called = False
            
            token.add_cleanup_callback(failing_callback)
            token.add_cleanup_callback(working_callback)
            
            # Should not raise exception
            token.cleanup()
            
            # Working callback should still be called
            assert callback_called


class TestProcessManager:
    """Test process manager functionality"""
    
    @pytest.mark.trio
    async def test_process_manager_creation(self):
        """Test creating a process manager"""
        with trio.CancelScope() as cancel_scope:
            token = TrioCancellationToken("test_job", cancel_scope)
            manager = ProcessManager("test_job", token)
            
            assert manager.job_id == "test_job"
            assert manager.cancellation_token == token
            assert manager.processes == []
    
    @pytest.mark.trio
    async def test_start_process_success(self):
        """Test starting a successful process"""
        with trio.CancelScope() as cancel_scope:
            token = TrioCancellationToken("test_job", cancel_scope)
            manager = ProcessManager("test_job", token)
            
            # Use a simple command that should succeed
            process = await manager.start_process(["echo", "hello"], timeout=5.0)
            
            assert process.returncode == 0
            assert len(manager.processes) == 1
    
    @pytest.mark.trio
    async def test_start_process_cancellation(self):
        """Test process cancellation"""
        with trio.CancelScope() as cancel_scope:
            token = TrioCancellationToken("test_job", cancel_scope)
            manager = ProcessManager("test_job", token)
            
            # Cancel before starting process
            token.cancel()
            
            with pytest.raises(trio.Cancelled):
                await manager.start_process(["sleep", "10"], timeout=5.0)
    
    @pytest.mark.trio
    async def test_process_timeout(self):
        """Test process timeout handling"""
        with trio.CancelScope() as cancel_scope:
            token = TrioCancellationToken("test_job", cancel_scope)
            manager = ProcessManager("test_job", token)
            
            with pytest.raises(Exception):  # Should raise TimeoutExpired or similar
                await manager.start_process(["sleep", "10"], timeout=0.1)
    
    @pytest.mark.trio
    async def test_cleanup_all_processes(self):
        """Test cleaning up all processes"""
        with trio.CancelScope() as cancel_scope:
            token = TrioCancellationToken("test_job", cancel_scope)
            manager = ProcessManager("test_job", token)
            
            # Mock a process
            mock_process = Mock()
            mock_process.poll.return_value = None  # Still running
            manager.processes.append(mock_process)
            
            manager.cleanup_all_processes()
            
            # Should have called terminate
            mock_process.terminate.assert_called_once()
            assert len(manager.processes) == 0


class TestTrioJobExecutor:
    """Test Trio job executor functionality"""
    
    def test_executor_creation(self):
        """Test creating a job executor"""
        executor = TrioJobExecutor()
        
        assert executor.active_jobs == {}
    
    @pytest.mark.trio
    async def test_execute_job_success(self):
        """Test successful job execution"""
        executor = TrioJobExecutor()
        
        async def mock_job_function(job_id, token, manager):
            return f"result_for_{job_id}"
        
        result = await executor.execute_job("test_job", mock_job_function, timeout=5.0)
        
        assert result == "result_for_test_job"
        assert "test_job" not in executor.active_jobs  # Should be cleaned up
    
    @pytest.mark.trio
    async def test_execute_job_cancellation(self):
        """Test job cancellation during execution"""
        executor = TrioJobExecutor()
        
        async def mock_job_function(job_id, token, manager):
            # Simulate some work
            await trio.sleep(0.1)
            if token.is_cancelled():
                raise trio.Cancelled()
            return f"result_for_{job_id}"
        
        # Start job and cancel it
        async with trio.open_nursery() as nursery:
            nursery.start_soon(executor.execute_job, "test_job", mock_job_function, 5.0)
            
            # Give job time to start
            await trio.sleep(0.05)
            
            # Cancel the job
            success = executor.cancel_job("test_job")
            assert success
            
            # Wait for cancellation to propagate
            await trio.sleep(0.2)
    
    @pytest.mark.trio
    async def test_execute_job_failure(self):
        """Test job execution failure"""
        executor = TrioJobExecutor()
        
        async def failing_job_function(job_id, token, manager):
            raise ValueError("Job failed")
        
        with patch('workers.trio_executor.get_job_queue') as mock_get_queue:
            mock_queue = Mock()
            mock_get_queue.return_value = mock_queue
            
            with pytest.raises(ValueError, match="Job failed"):
                await executor.execute_job("test_job", failing_job_function, timeout=5.0)
            
            # Should have updated job status to failed
            mock_queue.update_job_status.assert_called()
    
    def test_cancel_nonexistent_job(self):
        """Test cancelling a job that doesn't exist"""
        executor = TrioJobExecutor()
        
        result = executor.cancel_job("nonexistent_job")
        
        assert not result
    
    @pytest.mark.trio
    async def test_get_active_jobs(self):
        """Test getting active jobs list"""
        executor = TrioJobExecutor()
        
        # Add some mock active jobs
        with trio.CancelScope() as scope1, trio.CancelScope() as scope2:
            executor.active_jobs["job1"] = TrioCancellationToken("job1", scope1)
            executor.active_jobs["job2"] = TrioCancellationToken("job2", scope2)
            
            active_jobs = executor.get_active_jobs()
            
            assert set(active_jobs) == {"job1", "job2"}


class TestAsyncioTrioBridge:
    """Test asyncio-Trio bridge functionality"""
    
    @pytest.mark.trio
    async def test_run_sync_function(self):
        """Test running a sync function"""
        def sync_function(x, y):
            return x + y
        
        result = await AsyncioTrioBridge.run_asyncio_function(sync_function, 2, 3)
        
        assert result == 5
    
    @pytest.mark.trio
    async def test_run_async_function(self):
        """Test running an async function"""
        async def async_function(x, y):
            await asyncio.sleep(0.01)
            return x * y
        
        result = await AsyncioTrioBridge.run_asyncio_function(async_function, 3, 4)
        
        assert result == 12


class TestTrioJobContext:
    """Test Trio job context manager"""
    
    @pytest.mark.trio
    async def test_job_context_creation(self):
        """Test creating a job context"""
        async with trio_job_context("test_job") as (token, manager):
            assert isinstance(token, TrioCancellationToken)
            assert isinstance(manager, ProcessManager)
            assert token.job_id == "test_job"
            assert manager.job_id == "test_job"
    
    @pytest.mark.trio
    async def test_job_context_cleanup(self):
        """Test job context cleanup"""
        cleanup_called = False
        
        async with trio_job_context("test_job") as (token, manager):
            def cleanup_callback():
                nonlocal cleanup_called
                cleanup_called = True
            
            token.add_cleanup_callback(cleanup_callback)
        
        # Cleanup should have been called
        assert cleanup_called


class TestUtilityFunctions:
    """Test utility functions"""
    
    @pytest.mark.trio
    async def test_check_cancellation_async(self):
        """Test async cancellation check"""
        with trio.CancelScope() as cancel_scope:
            token = TrioCancellationToken("test_job", cancel_scope)
            
            # Not cancelled initially
            result = await check_cancellation_async(token, "test_job")
            assert not result
            
            # Cancel and check again
            token.cancel()
            
            with patch('workers.trio_executor.get_job_queue') as mock_get_queue:
                mock_queue = Mock()
                mock_get_queue.return_value = mock_queue
                
                result = await check_cancellation_async(token, "test_job")
                assert result
                
                # Should have updated job status
                mock_queue.update_job_status.assert_called_with("test_job", JobStatus.CANCELED)
    
    @pytest.mark.trio
    async def test_cleanup_temp_files_async(self):
        """Test async temp file cleanup"""
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Verify it exists
        assert os.path.exists(temp_dir)
        
        # Clean it up
        await cleanup_temp_files_async(temp_dir)
        
        # Verify it's gone
        assert not os.path.exists(temp_dir)


class TestGlobalExecutor:
    """Test global executor management"""
    
    def test_get_trio_executor(self):
        """Test getting global executor"""
        executor = get_trio_executor()
        
        assert isinstance(executor, TrioJobExecutor)
        
        # Should return same instance
        executor2 = get_trio_executor()
        assert executor is executor2
    
    def test_initialize_trio_executor(self):
        """Test initializing global executor"""
        executor = initialize_trio_executor()
        
        assert isinstance(executor, TrioJobExecutor)
        
        # Should be the same as get_trio_executor
        executor2 = get_trio_executor()
        assert executor is executor2


class TestIntegration:
    """Integration tests for Trio executor"""
    
    @pytest.mark.trio
    async def test_full_job_lifecycle(self):
        """Test complete job lifecycle with Trio"""
        executor = TrioJobExecutor()
        
        job_started = False
        job_completed = False
        
        async def test_job_function(job_id, token, manager):
            nonlocal job_started, job_completed
            job_started = True
            
            # Simulate some work with cancellation checks
            for i in range(5):
                if token.is_cancelled():
                    raise trio.Cancelled()
                await trio.sleep(0.01)
            
            job_completed = True
            return f"completed_{job_id}"
        
        with patch('workers.trio_executor.get_job_queue') as mock_get_queue:
            mock_queue = Mock()
            mock_get_queue.return_value = mock_queue
            
            result = await executor.execute_job("test_job", test_job_function, timeout=5.0)
            
            assert job_started
            assert job_completed
            assert result == "completed_test_job"
    
    @pytest.mark.trio
    async def test_job_cancellation_cleanup(self):
        """Test that cancellation properly cleans up resources"""
        executor = TrioJobExecutor()
        
        cleanup_called = False
        
        async def test_job_function(job_id, token, manager):
            # Add cleanup callback
            def cleanup():
                nonlocal cleanup_called
                cleanup_called = True
            
            token.add_cleanup_callback(cleanup)
            
            # Wait for cancellation
            while not token.is_cancelled():
                await trio.sleep(0.01)
            
            raise trio.Cancelled()
        
        with patch('workers.trio_executor.get_job_queue') as mock_get_queue:
            mock_queue = Mock()
            mock_get_queue.return_value = mock_queue
            
            async with trio.open_nursery() as nursery:
                nursery.start_soon(executor.execute_job, "test_job", test_job_function, 5.0)
                
                # Give job time to start
                await trio.sleep(0.05)
                
                # Cancel the job
                executor.cancel_job("test_job")
                
                # Wait for cleanup
                await trio.sleep(0.1)
        
        # Cleanup should have been called
        assert cleanup_called