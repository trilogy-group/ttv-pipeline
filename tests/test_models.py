"""
Tests for Pydantic models and data validation.
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from api.models import (
    JobStatus, JobCreateRequest, JobCreateResponse, JobStatusResponse,
    ArtifactResponse, LogsResponse, JobCancelResponse, ErrorResponse,
    HealthResponse, MetricsResponse, JobData
)


class TestJobModels:
    """Test job-related Pydantic models"""
    
    def test_job_create_request_valid(self):
        """Test valid job creation request"""
        request = JobCreateRequest(prompt="Generate a video of a sunset")
        
        assert request.prompt == "Generate a video of a sunset"
    
    def test_job_create_request_whitespace_cleanup(self):
        """Test that excessive whitespace is cleaned up"""
        request = JobCreateRequest(prompt="  Multiple   spaces   here  ")
        
        assert request.prompt == "Multiple spaces here"
    
    def test_job_create_request_empty_prompt(self):
        """Test that empty prompts are rejected"""
        with pytest.raises(ValidationError):
            JobCreateRequest(prompt="")
        
        with pytest.raises(ValidationError):
            JobCreateRequest(prompt="   ")  # Only whitespace
    
    def test_job_create_request_too_long(self):
        """Test that overly long prompts are rejected"""
        long_prompt = "x" * 2001  # Exceeds 2000 character limit
        
        with pytest.raises(ValidationError):
            JobCreateRequest(prompt=long_prompt)
    
    def test_job_create_response(self):
        """Test job creation response model"""
        now = datetime.now(timezone.utc)
        response = JobCreateResponse(
            id="job_123",
            status=JobStatus.QUEUED,
            created_at=now
        )
        
        assert response.id == "job_123"
        assert response.status == JobStatus.QUEUED
        assert response.created_at == now
    
    def test_job_status_response_complete(self):
        """Test complete job status response"""
        now = datetime.now(timezone.utc)
        response = JobStatusResponse(
            id="job_123",
            status=JobStatus.FINISHED,
            progress=100,
            created_at=now,
            started_at=now,
            finished_at=now,
            gcs_uri="gs://bucket/path/video.mp4"
        )
        
        assert response.id == "job_123"
        assert response.status == JobStatus.FINISHED
        assert response.progress == 100
        assert response.gcs_uri == "gs://bucket/path/video.mp4"
    
    def test_job_status_response_invalid_progress(self):
        """Test that invalid progress values are rejected"""
        now = datetime.now(timezone.utc)
        
        with pytest.raises(ValidationError):
            JobStatusResponse(
                id="job_123",
                status=JobStatus.PROGRESS,
                progress=101,  # > 100
                created_at=now
            )
        
        with pytest.raises(ValidationError):
            JobStatusResponse(
                id="job_123",
                status=JobStatus.PROGRESS,
                progress=-1,  # < 0
                created_at=now
            )


class TestArtifactAndLogsModels:
    """Test artifact and logs response models"""
    
    def test_artifact_response(self):
        """Test artifact response model"""
        response = ArtifactResponse(
            gcs_uri="gs://bucket/path/video.mp4",
            url="https://storage.googleapis.com/signed-url",
            expires_in=3600
        )
        
        assert response.gcs_uri == "gs://bucket/path/video.mp4"
        assert response.url == "https://storage.googleapis.com/signed-url"
        assert response.expires_in == 3600
    
    def test_logs_response(self):
        """Test logs response model"""
        logs = [
            "[2025-01-01T12:00:00] Starting job",
            "[2025-01-01T12:01:00] Processing segment 1",
            "[2025-01-01T12:02:00] Job completed"
        ]
        
        response = LogsResponse(lines=logs)
        
        assert response.lines == logs
    
    def test_logs_response_truncation(self):
        """Test that logs are truncated when too many lines"""
        # Create more than 1000 log lines
        logs = [f"Log line {i}" for i in range(1500)]
        
        response = LogsResponse(lines=logs)
        
        # Should be truncated to last 1000 lines
        assert len(response.lines) == 1000
        assert response.lines[0] == "Log line 500"  # First of last 1000
        assert response.lines[-1] == "Log line 1499"  # Last line
    
    def test_job_cancel_response(self):
        """Test job cancellation response"""
        response = JobCancelResponse(
            id="job_123",
            status=JobStatus.CANCELED,
            message="Job cancellation requested"
        )
        
        assert response.id == "job_123"
        assert response.status == JobStatus.CANCELED
        assert response.message == "Job cancellation requested"


class TestSystemModels:
    """Test system health and metrics models"""
    
    def test_error_response(self):
        """Test error response model"""
        now = datetime.now(timezone.utc)
        response = ErrorResponse(
            error="ValidationError",
            message="Invalid input provided",
            details={"field": "prompt", "issue": "too_short"},
            timestamp=now,
            request_id="req_123"
        )
        
        assert response.error == "ValidationError"
        assert response.message == "Invalid input provided"
        assert response.details["field"] == "prompt"
        assert response.timestamp == now
        assert response.request_id == "req_123"
    
    def test_health_response(self):
        """Test health response model"""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            components={
                "api": "healthy",
                "redis": "healthy",
                "gcs": "healthy"
            }
        )
        
        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert response.components["api"] == "healthy"
    
    def test_metrics_response(self):
        """Test metrics response model"""
        response = MetricsResponse(
            active_jobs=5,
            queued_jobs=10,
            total_jobs_processed=100,
            average_processing_time=45.5,
            uptime_seconds=3600
        )
        
        assert response.active_jobs == 5
        assert response.queued_jobs == 10
        assert response.total_jobs_processed == 100
        assert response.average_processing_time == 45.5
        assert response.uptime_seconds == 3600


class TestJobData:
    """Test internal JobData model"""
    
    def test_job_data_creation(self):
        """Test JobData model creation"""
        now = datetime.now(timezone.utc)
        job_data = JobData(
            id="job_123",
            status=JobStatus.QUEUED,
            created_at=now,
            prompt="Test prompt",
            config={"backend": "veo3"}
        )
        
        assert job_data.id == "job_123"
        assert job_data.status == JobStatus.QUEUED
        assert job_data.progress == 0  # Default value
        assert job_data.prompt == "Test prompt"
        assert job_data.config["backend"] == "veo3"
        assert len(job_data.logs) == 0  # Default empty list
    
    def test_job_data_to_status_response(self):
        """Test converting JobData to JobStatusResponse"""
        now = datetime.now(timezone.utc)
        job_data = JobData(
            id="job_123",
            status=JobStatus.FINISHED,
            progress=100,
            created_at=now,
            started_at=now,
            finished_at=now,
            prompt="Test prompt",
            gcs_uri="gs://bucket/video.mp4"
        )
        
        response = job_data.to_status_response()
        
        assert isinstance(response, JobStatusResponse)
        assert response.id == "job_123"
        assert response.status == JobStatus.FINISHED
        assert response.progress == 100
        assert response.gcs_uri == "gs://bucket/video.mp4"
    
    def test_job_data_add_log(self):
        """Test adding log entries to JobData"""
        job_data = JobData(
            id="job_123",
            status=JobStatus.QUEUED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        job_data.add_log("Starting job")
        job_data.add_log("Processing segment 1")
        
        assert len(job_data.logs) == 2
        assert "Starting job" in job_data.logs[0]
        assert "Processing segment 1" in job_data.logs[1]
        # Check that timestamps are added
        assert job_data.logs[0].startswith("[")
        assert "]" in job_data.logs[0]
    
    def test_job_data_log_truncation(self):
        """Test that logs are truncated when max_logs is exceeded"""
        job_data = JobData(
            id="job_123",
            status=JobStatus.QUEUED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        # Add more logs than the limit
        for i in range(15):
            job_data.add_log(f"Log message {i}", max_logs=10)
        
        # Should only keep the last 10 logs
        assert len(job_data.logs) == 10
        assert "Log message 5" in job_data.logs[0]  # First of last 10
        assert "Log message 14" in job_data.logs[-1]  # Last log