"""
Mock Redis manager and job queue for integration testing.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from unittest.mock import Mock

from api.models import JobData, JobStatus, JobCreateRequest


class MockRedisManager:
    """Mock Redis connection manager"""
    
    def __init__(self, config):
        self.config = config
        self.connected = True
        self.connection_count = 0
        
    def get_connection(self):
        """Mock Redis connection"""
        self.connection_count += 1
        return MockRedisConnection()
    
    def test_connection(self) -> bool:
        """Mock connection test"""
        return self.connected
    
    def close(self):
        """Mock connection close"""
        self.connected = False


class MockRedisConnection:
    """Mock Redis connection"""
    
    def __init__(self):
        self.data = {}
        
    def ping(self):
        """Mock ping"""
        return True
    
    def get(self, key: str):
        """Mock get"""
        return self.data.get(key)
    
    def set(self, key: str, value: str):
        """Mock set"""
        self.data[key] = value
    
    def setex(self, key: str, ttl: int, value: str):
        """Mock setex"""
        self.data[key] = value
    
    def delete(self, key: str):
        """Mock delete"""
        if key in self.data:
            del self.data[key]
            return 1
        return 0
    
    def keys(self, pattern: str):
        """Mock keys"""
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return [key for key in self.data.keys() if key.startswith(prefix)]
        return [key for key in self.data.keys() if key == pattern]


class MockJobQueue:
    """Mock job queue that simulates Redis/RQ behavior"""
    
    def __init__(self, redis_manager: MockRedisManager):
        self.redis_manager = redis_manager
        self.jobs = {}  # job_id -> JobData
        self.queue_stats = {
            "queued_jobs": 0,
            "started_jobs": 0,
            "finished_jobs": 0,
            "failed_jobs": 0
        }
        self.enqueue_count = 0
        
    def enqueue_job(self, request: JobCreateRequest, effective_config: Dict[str, Any], 
                   job_timeout: int = 3600) -> JobData:
        """Mock job enqueueing"""
        job_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        
        job_data = JobData(
            id=job_id,
            status=JobStatus.QUEUED,
            created_at=created_at,
            prompt=request.prompt,
            config=effective_config
        )
        
        self.jobs[job_id] = job_data
        self.queue_stats["queued_jobs"] += 1
        self.enqueue_count += 1
        
        return job_data
    
    def get_job(self, job_id: str) -> Optional[JobData]:
        """Mock job retrieval"""
        return self.jobs.get(job_id)
    
    def update_job_status(self, job_id: str, status: JobStatus, 
                         progress: Optional[int] = None,
                         error: Optional[str] = None,
                         gcs_uri: Optional[str] = None) -> bool:
        """Mock job status update"""
        if job_id not in self.jobs:
            return False
        
        job_data = self.jobs[job_id]
        old_status = job_data.status
        
        # Update status
        job_data.status = status
        
        # Update timestamps
        now = datetime.now(timezone.utc)
        if status == JobStatus.STARTED and old_status == JobStatus.QUEUED:
            job_data.started_at = now
            self.queue_stats["queued_jobs"] -= 1
            self.queue_stats["started_jobs"] += 1
        elif status in [JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELED]:
            if job_data.finished_at is None:
                job_data.finished_at = now
                if old_status == JobStatus.STARTED or old_status == JobStatus.PROGRESS:
                    self.queue_stats["started_jobs"] -= 1
                elif old_status == JobStatus.QUEUED:
                    self.queue_stats["queued_jobs"] -= 1
                
                if status == JobStatus.FINISHED:
                    self.queue_stats["finished_jobs"] += 1
                elif status == JobStatus.FAILED:
                    self.queue_stats["failed_jobs"] += 1
        
        # Update optional fields
        if progress is not None:
            job_data.progress = max(0, min(100, progress))
        
        if error is not None:
            job_data.error = error
        
        if gcs_uri is not None:
            job_data.gcs_uri = gcs_uri
        
        return True
    
    def add_job_log(self, job_id: str, message: str) -> bool:
        """Mock job log addition"""
        if job_id not in self.jobs:
            return False
        
        job_data = self.jobs[job_id]
        job_data.add_log(message)
        return True
    
    def get_job_logs(self, job_id: str, tail: Optional[int] = None) -> List[str]:
        """Mock job log retrieval"""
        if job_id not in self.jobs:
            return []
        
        job_data = self.jobs[job_id]
        logs = job_data.logs
        
        if tail is not None and tail > 0:
            logs = logs[-tail:]
        
        return logs
    
    def cancel_job(self, job_id: str) -> bool:
        """Mock job cancellation"""
        if job_id not in self.jobs:
            return False
        
        job_data = self.jobs[job_id]
        
        # Only cancel jobs that are not already finished
        if job_data.status in [JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELED]:
            return False
        
        # Update status to canceled
        self.update_job_status(job_id, JobStatus.CANCELED)
        self.add_job_log(job_id, "Job cancellation requested")
        
        return True
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Mock queue statistics"""
        return self.queue_stats.copy()
    
    def cleanup_old_jobs(self, max_age_days: int = 7) -> int:
        """Mock job cleanup"""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (max_age_days * 24 * 3600)
        
        jobs_to_remove = []
        for job_id, job_data in self.jobs.items():
            if job_data.created_at.timestamp() < cutoff_time:
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.jobs[job_id]
        
        return len(jobs_to_remove)
    
    def get_all_jobs(self) -> Dict[str, JobData]:
        """Get all jobs (for testing)"""
        return self.jobs.copy()
    
    def reset(self):
        """Reset mock state"""
        self.jobs.clear()
        self.queue_stats = {
            "queued_jobs": 0,
            "started_jobs": 0,
            "finished_jobs": 0,
            "failed_jobs": 0
        }
        self.enqueue_count = 0


class MockJobStateManager:
    """Mock job state manager for testing state transitions"""
    
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
        """Check if a status transition is valid"""
        return to_status in cls.VALID_TRANSITIONS.get(from_status, [])
    
    @classmethod
    def is_terminal_status(cls, status: JobStatus) -> bool:
        """Check if a status is terminal"""
        return len(cls.VALID_TRANSITIONS.get(status, [])) == 0


def create_mock_job_with_progress(job_id: str, progress_steps: List[tuple]) -> JobData:
    """Create a mock job with predefined progress steps"""
    job_data = JobData(
        id=job_id,
        status=JobStatus.QUEUED,
        created_at=datetime.now(timezone.utc),
        prompt="Test job with progress"
    )
    
    # Add progress logs
    for progress, message in progress_steps:
        job_data.add_log(f"Progress {progress}%: {message}")
    
    return job_data


def create_mock_failed_job(job_id: str, error_message: str) -> JobData:
    """Create a mock failed job"""
    job_data = JobData(
        id=job_id,
        status=JobStatus.FAILED,
        created_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        prompt="Test job that failed",
        error=error_message
    )
    
    job_data.add_log("Job started")
    job_data.add_log(f"Error occurred: {error_message}")
    
    return job_data


def create_mock_completed_job(job_id: str, gcs_uri: str) -> JobData:
    """Create a mock completed job"""
    job_data = JobData(
        id=job_id,
        status=JobStatus.FINISHED,
        progress=100,
        created_at=datetime.now(timezone.utc),
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        prompt="Test job that completed",
        gcs_uri=gcs_uri
    )
    
    job_data.add_log("Job started")
    job_data.add_log("Processing video...")
    job_data.add_log("Video generation complete")
    job_data.add_log(f"Uploaded to {gcs_uri}")
    
    return job_data