"""
Redis job queue infrastructure for the API server.

This module provides Redis connection management, RQ (Redis Queue) integration,
and job state management with proper status transitions.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager

import redis
from rq import Queue, Worker
from rq.job import Job
from rq.exceptions import NoSuchJobError
from rq.job import JobStatus as RQJobStatus

from .models import JobData, JobStatus, JobCreateRequest
from .config import RedisConfig

logger = logging.getLogger(__name__)


class RedisConnectionManager:
    """Manages Redis connections and connection pooling"""
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self._pool = None
        self._connection = None
    
    def get_connection_pool(self) -> redis.ConnectionPool:
        """Get or create Redis connection pool"""
        if self._pool is None:
            self._pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            logger.info(f"Created Redis connection pool for {self.config.host}:{self.config.port}")
        
        return self._pool
    
    def get_connection(self) -> redis.Redis:
        """Get Redis connection from pool"""
        if self._connection is None:
            pool = self.get_connection_pool()
            self._connection = redis.Redis(connection_pool=pool)
        
        return self._connection
    
    def test_connection(self) -> bool:
        """Test Redis connection"""
        try:
            conn = self.get_connection()
            conn.ping()
            logger.info("Redis connection test successful")
            return True
        except Exception as e:
            logger.error(f"Redis connection test failed: {e}")
            return False
    
    def close(self):
        """Close Redis connections"""
        if self._connection:
            self._connection.close()
            self._connection = None
        
        if self._pool:
            self._pool.disconnect()
            self._pool = None
        
        logger.info("Redis connections closed")


class JobQueue:
    """Redis-based job queue using RQ"""
    
    def __init__(self, redis_manager: RedisConnectionManager, queue_name: str = "video_generation"):
        self.redis_manager = redis_manager
        self.queue_name = queue_name
        self._queue = None
        self._redis = None
    
    @property
    def queue(self) -> Queue:
        """Get RQ queue instance"""
        if self._queue is None:
            connection = self.redis_manager.get_connection()
            # Set result_ttl to 7 days (604800 seconds) to prevent automatic deletion
            self._queue = Queue(
                self.queue_name, 
                connection=connection,
                result_ttl=604800  # 7 days in seconds
            )
        return self._queue
    
    @property
    def redis(self) -> redis.Redis:
        """Get Redis connection for direct operations"""
        if self._redis is None:
            self._redis = self.redis_manager.get_connection()
        return self._redis
    
    def enqueue_job(
        self, 
        request: JobCreateRequest, 
        effective_config: Dict[str, Any],
        job_timeout: int = 3600
    ) -> JobData:
        """
        Enqueue a new video generation job
        
        Args:
            request: Job creation request with prompt
            effective_config: Merged configuration for the job
            job_timeout: Job timeout in seconds
            
        Returns:
            JobData instance with job details
            
        Raises:
            Exception: If job enqueueing fails
        """
        job_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        
        # Create job data
        job_data = JobData(
            id=job_id,
            status=JobStatus.QUEUED,
            created_at=created_at,
            prompt=request.prompt,
            config=effective_config
        )
        
        try:
            # Store job metadata in Redis
            self._store_job_data(job_data)
            
            # Enqueue the job with RQ
            rq_job = self.queue.enqueue(
                'workers.video_worker.process_video_job',  # Worker function
                job_id,
                job_timeout=job_timeout,
                job_id=job_id  # Use our UUID as RQ job ID
            )
            
            logger.info(f"Job {job_id} enqueued successfully")
            return job_data
            
        except Exception as e:
            logger.error(f"Failed to enqueue job {job_id}: {e}")
            # Clean up job data if enqueueing failed
            self._delete_job_data(job_id)
            raise
    
    def get_job(self, job_id: str) -> Optional[JobData]:
        """
        Get job data by ID
        
        Args:
            job_id: Job identifier
            
        Returns:
            JobData instance or None if not found
        """
        try:
            job_key = self._get_job_key(job_id)
            job_json = self.redis.get(job_key)
            
            if job_json is None:
                return None
            
            job_dict = json.loads(job_json)
            return JobData(**job_dict)
            
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            return None
    
    def list_jobs(self, limit: int = 100, offset: int = 0) -> List[JobData]:
        """
        List jobs with pagination, ordered by creation time (newest first)
        
        Args:
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip
            
        Returns:
            List of JobData instances
        """
        try:
            pattern = self._get_job_key("*")
            job_keys = self.redis.keys(pattern)
            
            jobs = []
            for job_key in job_keys:
                try:
                    job_json = self.redis.get(job_key)
                    if job_json:
                        job_dict = json.loads(job_json)
                        jobs.append(JobData(**job_dict))
                except Exception as e:
                    logger.warning(f"Failed to parse job data for key {job_key}: {e}")
                    continue
            
            # Sort by creation time (newest first)
            jobs.sort(key=lambda x: x.created_at, reverse=True)
            
            # Apply pagination
            start_idx = offset
            end_idx = offset + limit
            
            return jobs[start_idx:end_idx]
            
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            return []
    
    def update_job_status(
        self, 
        job_id: str, 
        status: JobStatus, 
        progress: Optional[int] = None,
        error: Optional[str] = None,
        gcs_uri: Optional[str] = None
    ) -> bool:
        """
        Update job status and related fields
        
        Args:
            job_id: Job identifier
            status: New job status
            progress: Progress percentage (0-100)
            error: Error message (for failed jobs)
            gcs_uri: GCS URI (for completed jobs)
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            job_data = self.get_job(job_id)
            if job_data is None:
                logger.warning(f"Cannot update non-existent job {job_id}")
                return False
            
            # Update status
            old_status = job_data.status
            job_data.status = status
            
            # Update timestamps based on status transitions
            now = datetime.now(timezone.utc)
            
            if status == JobStatus.STARTED and old_status == JobStatus.QUEUED:
                job_data.started_at = now
            elif status in [JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELED]:
                if job_data.finished_at is None:
                    job_data.finished_at = now
            
            # Update optional fields
            if progress is not None:
                job_data.progress = max(0, min(100, progress))
            
            if error is not None:
                job_data.error = error
            
            if gcs_uri is not None:
                job_data.gcs_uri = gcs_uri
            
            # Store updated job data
            self._store_job_data(job_data)
            
            logger.info(f"Job {job_id} status updated: {old_status} -> {status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update job {job_id} status: {e}")
            return False
    
    def add_job_log(self, job_id: str, message: str) -> bool:
        """
        Add a log message to a job
        
        Args:
            job_id: Job identifier
            message: Log message
            
        Returns:
            True if log added successfully, False otherwise
        """
        try:
            job_data = self.get_job(job_id)
            if job_data is None:
                logger.warning(f"Cannot add log to non-existent job {job_id}")
                return False
            
            job_data.add_log(message)
            self._store_job_data(job_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add log to job {job_id}: {e}")
            return False
    
    def get_job_logs(self, job_id: str, tail: Optional[int] = None) -> List[str]:
        """
        Get job logs
        
        Args:
            job_id: Job identifier
            tail: Number of recent log lines to return (None for all)
            
        Returns:
            List of log lines
        """
        try:
            job_data = self.get_job(job_id)
            if job_data is None:
                return []
            
            logs = job_data.logs
            if tail is not None and tail > 0:
                logs = logs[-tail:]
            
            return logs
            
        except Exception as e:
            logger.error(f"Failed to get logs for job {job_id}: {e}")
            return []
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job (cooperative cancellation)
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if cancellation initiated, False otherwise
        """
        try:
            job_data = self.get_job(job_id)
            if job_data is None:
                logger.warning(f"Cannot cancel non-existent job {job_id}")
                return False
            
            # Only cancel jobs that are not already finished
            if job_data.status in [JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELED]:
                logger.warning(f"Cannot cancel job {job_id} in status {job_data.status}")
                return False
            
            # Set cancellation flag for worker processes (threading-based)
            try:
                from workers.video_worker import set_job_cancellation_flag
                set_job_cancellation_flag(job_id)
                logger.info(f"Threading cancellation flag set for job {job_id}")
            except ImportError:
                logger.warning("Could not import worker cancellation function")
            except Exception as e:
                logger.warning(f"Failed to set threading cancellation flag for job {job_id}: {e}")
            
            # Set cancellation flag for Trio-based workers
            try:
                from workers.trio_executor import get_trio_executor
                trio_executor = get_trio_executor()
                if trio_executor.cancel_job(job_id):
                    logger.info(f"Trio cancellation initiated for job {job_id}")
            except ImportError:
                logger.debug("Trio executor not available for cancellation")
            except Exception as e:
                logger.warning(f"Failed to cancel Trio job {job_id}: {e}")
            
            # Try to cancel RQ job
            try:
                rq_job = Job.fetch(job_id, connection=self.redis)
                rq_job.cancel()
                logger.info(f"RQ job {job_id} cancelled")
            except NoSuchJobError:
                logger.warning(f"RQ job {job_id} not found, updating status only")
            except Exception as e:
                logger.warning(f"Failed to cancel RQ job {job_id}: {e}")
            
            # Update job status to canceled
            self.update_job_status(job_id, JobStatus.CANCELED)
            self.add_job_log(job_id, "Job cancellation requested")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    def get_queue_stats(self) -> Dict[str, int]:
        """
        Get queue statistics
        
        Returns:
            Dictionary with queue statistics
        """
        try:
            queue = self.queue
            
            return {
                "queued_jobs": len(queue),
                "started_jobs": len(queue.started_job_registry),
                "finished_jobs": len(queue.finished_job_registry),
                "failed_jobs": len(queue.failed_job_registry),
                "deferred_jobs": len(queue.deferred_job_registry),
                "scheduled_jobs": len(queue.scheduled_job_registry)
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {}
    
    def cleanup_old_jobs(self, max_age_days: int = 7) -> int:
        """
        Clean up old job data from Redis
        
        Args:
            max_age_days: Maximum age of jobs to keep
            
        Returns:
            Number of jobs cleaned up
        """
        try:
            pattern = self._get_job_key("*")
            job_keys = self.redis.keys(pattern)
            
            cleaned_count = 0
            cutoff_time = datetime.now(timezone.utc).timestamp() - (max_age_days * 24 * 3600)
            
            for job_key in job_keys:
                try:
                    job_json = self.redis.get(job_key)
                    if job_json:
                        job_dict = json.loads(job_json)
                        created_at = datetime.fromisoformat(job_dict['created_at'].replace('Z', '+00:00'))
                        
                        if created_at.timestamp() < cutoff_time:
                            self.redis.delete(job_key)
                            cleaned_count += 1
                            
                except Exception as e:
                    logger.warning(f"Failed to process job key {job_key} during cleanup: {e}")
                    continue
            
            logger.info(f"Cleaned up {cleaned_count} old jobs")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old jobs: {e}")
            return 0
    
    def _store_job_data(self, job_data: JobData):
        """Store job data in Redis"""
        job_key = self._get_job_key(job_data.id)
        job_json = job_data.model_dump_json()
        
        # Store with expiration (30 days)
        self.redis.setex(job_key, 30 * 24 * 3600, job_json)
    
    def _delete_job_data(self, job_id: str):
        """Delete job data from Redis"""
        job_key = self._get_job_key(job_id)
        self.redis.delete(job_key)
    
    def _get_job_key(self, job_id: str) -> str:
        """Get Redis key for job data"""
        return f"job:{job_id}"


class JobStateManager:
    """Manages job state transitions and validation"""
    
    # Valid state transitions
    VALID_TRANSITIONS = {
        JobStatus.QUEUED: [JobStatus.STARTED, JobStatus.CANCELED],
        JobStatus.STARTED: [JobStatus.PROGRESS, JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELED],
        JobStatus.PROGRESS: [JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELED],
        JobStatus.FINISHED: [],  # Terminal state
        JobStatus.FAILED: [],    # Terminal state
        JobStatus.CANCELED: []   # Terminal state
    }
    
    @classmethod
    def is_valid_transition(cls, from_status: JobStatus, to_status: JobStatus) -> bool:
        """
        Check if a status transition is valid
        
        Args:
            from_status: Current job status
            to_status: Target job status
            
        Returns:
            True if transition is valid, False otherwise
        """
        return to_status in cls.VALID_TRANSITIONS.get(from_status, [])
    
    @classmethod
    def is_terminal_status(cls, status: JobStatus) -> bool:
        """
        Check if a status is terminal (no further transitions allowed)
        
        Args:
            status: Job status to check
            
        Returns:
            True if status is terminal, False otherwise
        """
        return len(cls.VALID_TRANSITIONS.get(status, [])) == 0
    
    @classmethod
    def get_valid_next_states(cls, current_status: JobStatus) -> List[JobStatus]:
        """
        Get list of valid next states for a given status
        
        Args:
            current_status: Current job status
            
        Returns:
            List of valid next statuses
        """
        return cls.VALID_TRANSITIONS.get(current_status, [])


# Global instances (will be initialized by the application)
redis_manager: Optional[RedisConnectionManager] = None
job_queue: Optional[JobQueue] = None


def initialize_queue_infrastructure(redis_config: RedisConfig) -> tuple[RedisConnectionManager, JobQueue]:
    """
    Initialize Redis connection manager and job queue
    
    Args:
        redis_config: Redis configuration
        
    Returns:
        Tuple of (RedisConnectionManager, JobQueue)
        
    Raises:
        Exception: If initialization fails
    """
    global redis_manager, job_queue
    
    try:
        # Initialize Redis connection manager
        redis_manager = RedisConnectionManager(redis_config)
        
        # Test connection
        if not redis_manager.test_connection():
            raise Exception("Redis connection test failed")
        
        # Initialize job queue
        job_queue = JobQueue(redis_manager)
        
        logger.info("Queue infrastructure initialized successfully")
        return redis_manager, job_queue
        
    except Exception as e:
        logger.error(f"Failed to initialize queue infrastructure: {e}")
        raise


def get_job_queue() -> JobQueue:
    """
    Get the global job queue instance
    
    Returns:
        JobQueue instance
        
    Raises:
        RuntimeError: If queue infrastructure not initialized
    """
    if job_queue is None:
        raise RuntimeError("Queue infrastructure not initialized. Call initialize_queue_infrastructure() first.")
    
    return job_queue


def get_redis_manager() -> RedisConnectionManager:
    """
    Get the global Redis manager instance
    
    Returns:
        RedisConnectionManager instance
        
    Raises:
        RuntimeError: If queue infrastructure not initialized
    """
    if redis_manager is None:
        raise RuntimeError("Queue infrastructure not initialized. Call initialize_queue_infrastructure() first.")
    
    return redis_manager


@asynccontextmanager
async def queue_lifespan():
    """
    Async context manager for queue lifecycle management
    
    Usage:
        async with queue_lifespan():
            # Use queue infrastructure
            pass
    """
    try:
        yield
    finally:
        # Cleanup on exit
        if redis_manager:
            redis_manager.close()
            logger.info("Queue infrastructure cleaned up")