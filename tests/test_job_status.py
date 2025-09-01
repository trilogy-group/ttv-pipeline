"""
Tests for job status and polling endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from api.main import create_app
from api.models import JobData, JobStatus


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = MagicMock()
    config.server.host = "localhost"
    config.server.port = 8000
    config.server.quic_port = 8443
    config.redis.host = "localhost"
    config.redis.port = 6379
    config.gcs.bucket = "test-bucket"
    config.server.cors_origins = ["*"]
    config.security.rate_limit_per_minute = 60
    return config


@pytest.fixture
def client(mock_config):
    """Test client with mocked configuration"""
    with patch('api.main.get_config_from_env', return_value=mock_config), \
         patch('api.queue.initialize_queue_infrastructure') as mock_init_queue:
        
        # Mock the queue infrastructure initialization
        mock_redis_manager = MagicMock()
        mock_job_queue = MagicMock()
        mock_init_queue.return_value = (mock_redis_manager, mock_job_queue)
        
        app = create_app()
        app.state.config = mock_config
        
        return TestClient(app)


class TestJobStatusEndpoint:
    """Test job status retrieval endpoint"""
    
    @patch('api.queue.get_job_queue')
    def test_get_job_status_success(self, mock_get_queue, client):
        """Test successful job status retrieval"""
        # Mock job queue
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        
        # Create mock job data
        job_data = JobData(
            id="test-job-123",
            status=JobStatus.PROGRESS,
            progress=45,
            created_at=datetime(2025, 8, 31, 12, 0, 0, tzinfo=timezone.utc),
            started_at=datetime(2025, 8, 31, 12, 1, 0, tzinfo=timezone.utc),
            prompt="A test prompt",
            gcs_uri=None,
            error=None
        )
        mock_queue.get_job.return_value = job_data
        
        # Make request
        response = client.get("/v1/jobs/test-job-123")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-job-123"
        assert data["status"] == "progress"
        assert data["progress"] == 45
        assert data["created_at"] == "2025-08-31T12:00:00Z"
        assert data["started_at"] == "2025-08-31T12:01:00Z"
        assert data["finished_at"] is None
        assert data["gcs_uri"] is None
        assert data["error"] is None
        
        # Verify queue was called correctly
        mock_queue.get_job.assert_called_once_with("test-job-123")
    
    @patch('api.queue.get_job_queue')
    def test_get_job_status_finished_with_gcs_uri(self, mock_get_queue, client):
        """Test job status retrieval for finished job with GCS URI"""
        # Mock job queue
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        
        # Create mock finished job data
        job_data = JobData(
            id="test-job-456",
            status=JobStatus.FINISHED,
            progress=100,
            created_at=datetime(2025, 8, 31, 12, 0, 0, tzinfo=timezone.utc),
            started_at=datetime(2025, 8, 31, 12, 1, 0, tzinfo=timezone.utc),
            finished_at=datetime(2025, 8, 31, 12, 15, 30, tzinfo=timezone.utc),
            prompt="A finished test prompt",
            gcs_uri="gs://test-bucket/ttv-api/2025-08/test-job-456/final_video.mp4",
            error=None
        )
        mock_queue.get_job.return_value = job_data
        
        # Make request
        response = client.get("/v1/jobs/test-job-456")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-job-456"
        assert data["status"] == "finished"
        assert data["progress"] == 100
        assert data["finished_at"] == "2025-08-31T12:15:30Z"
        assert data["gcs_uri"] == "gs://test-bucket/ttv-api/2025-08/test-job-456/final_video.mp4"
        assert data["error"] is None
    
    @patch('api.queue.get_job_queue')
    def test_get_job_status_failed_with_error(self, mock_get_queue, client):
        """Test job status retrieval for failed job with error message"""
        # Mock job queue
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        
        # Create mock failed job data
        job_data = JobData(
            id="test-job-789",
            status=JobStatus.FAILED,
            progress=25,
            created_at=datetime(2025, 8, 31, 12, 0, 0, tzinfo=timezone.utc),
            started_at=datetime(2025, 8, 31, 12, 1, 0, tzinfo=timezone.utc),
            finished_at=datetime(2025, 8, 31, 12, 5, 15, tzinfo=timezone.utc),
            prompt="A failed test prompt",
            gcs_uri=None,
            error="Pipeline execution failed: Invalid model configuration"
        )
        mock_queue.get_job.return_value = job_data
        
        # Make request
        response = client.get("/v1/jobs/test-job-789")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-job-789"
        assert data["status"] == "failed"
        assert data["progress"] == 25
        assert data["finished_at"] == "2025-08-31T12:05:15Z"
        assert data["gcs_uri"] is None
        assert data["error"] == "Pipeline execution failed: Invalid model configuration"
    
    @patch('api.queue.get_job_queue')
    def test_get_job_status_not_found(self, mock_get_queue, client):
        """Test job status retrieval for non-existent job"""
        # Mock job queue
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        mock_queue.get_job.return_value = None
        
        # Make request
        response = client.get("/v1/jobs/nonexistent-job")
        
        # Verify response
        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "JobNotFoundError"
        assert "nonexistent-job" in data["message"]
        assert "request_id" in data
        
        # Verify queue was called correctly
        mock_queue.get_job.assert_called_once_with("nonexistent-job")
    
    @patch('api.queue.get_job_queue')
    def test_get_job_status_redis_error(self, mock_get_queue, client):
        """Test job status retrieval when Redis fails"""
        # Mock job queue
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        mock_queue.get_job.side_effect = Exception("Redis connection timeout")
        
        # Make request
        response = client.get("/v1/jobs/test-job-123")
        
        # Verify response
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "RedisConnectionError"
        assert "Redis" in data["message"]
        assert "request_id" in data
    
    @patch('api.queue.get_job_queue')
    def test_get_job_status_queue_not_initialized(self, mock_get_queue, client):
        """Test job status retrieval when queue is not initialized"""
        mock_get_queue.side_effect = RuntimeError("Queue infrastructure not initialized")
        
        # Make request
        response = client.get("/v1/jobs/test-job-123")
        
        # Verify response
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "APIException"
        assert "request_id" in data


class TestJobLogsEndpoint:
    """Test job logs retrieval endpoint"""
    
    @patch('api.queue.get_job_queue')
    def test_get_job_logs_success(self, mock_get_queue, client):
        """Test successful job logs retrieval"""
        # Mock job queue
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        
        # Mock job data (to verify job exists)
        job_data = JobData(
            id="test-job-123",
            status=JobStatus.PROGRESS,
            created_at=datetime.now(timezone.utc),
            prompt="A test prompt"
        )
        mock_queue.get_job.return_value = job_data
        
        # Mock log lines
        log_lines = [
            "[2025-08-31T12:01:00Z] Job started",
            "[2025-08-31T12:01:15Z] Loading model configuration",
            "[2025-08-31T12:02:00Z] Generating keyframes",
            "[2025-08-31T12:03:30Z] Processing video segments"
        ]
        mock_queue.get_job_logs.return_value = log_lines
        
        # Make request
        response = client.get("/v1/jobs/test-job-123/logs")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["lines"] == log_lines
        
        # Verify queue was called correctly
        mock_queue.get_job.assert_called_once_with("test-job-123")
        mock_queue.get_job_logs.assert_called_once_with("test-job-123", tail=100)
    
    @patch('api.queue.get_job_queue')
    def test_get_job_logs_with_tail_parameter(self, mock_get_queue, client):
        """Test job logs retrieval with tail parameter"""
        # Mock job queue
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        
        # Mock job data
        job_data = JobData(
            id="test-job-123",
            status=JobStatus.PROGRESS,
            created_at=datetime.now(timezone.utc),
            prompt="A test prompt"
        )
        mock_queue.get_job.return_value = job_data
        
        # Mock log lines (last 50)
        log_lines = [f"[2025-08-31T12:01:{i:02d}Z] Log line {i}" for i in range(50)]
        mock_queue.get_job_logs.return_value = log_lines
        
        # Make request with tail parameter
        response = client.get("/v1/jobs/test-job-123/logs?tail=50")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert len(data["lines"]) == 50
        
        # Verify queue was called with correct tail parameter
        mock_queue.get_job_logs.assert_called_once_with("test-job-123", tail=50)
    
    @patch('api.queue.get_job_queue')
    def test_get_job_logs_tail_parameter_validation(self, mock_get_queue, client):
        """Test job logs retrieval with invalid tail parameter"""
        # Test negative tail parameter
        response = client.get("/v1/jobs/test-job-123/logs?tail=-1")
        
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "ValidationError"
        assert "non-negative" in data["message"]
    
    @patch('api.queue.get_job_queue')
    def test_get_job_logs_tail_parameter_limit(self, mock_get_queue, client):
        """Test job logs retrieval with tail parameter exceeding limit"""
        # Mock job queue
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        
        # Mock job data
        job_data = JobData(
            id="test-job-123",
            status=JobStatus.PROGRESS,
            created_at=datetime.now(timezone.utc),
            prompt="A test prompt"
        )
        mock_queue.get_job.return_value = job_data
        
        # Mock log lines
        mock_queue.get_job_logs.return_value = []
        
        # Make request with very large tail parameter
        response = client.get("/v1/jobs/test-job-123/logs?tail=50000")
        
        # Verify response is successful (tail should be limited to 10000)
        assert response.status_code == 200
        
        # Verify queue was called with limited tail parameter
        mock_queue.get_job_logs.assert_called_once_with("test-job-123", tail=10000)
    
    @patch('api.queue.get_job_queue')
    def test_get_job_logs_job_not_found(self, mock_get_queue, client):
        """Test job logs retrieval for non-existent job"""
        # Mock job queue
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        mock_queue.get_job.return_value = None
        
        # Make request
        response = client.get("/v1/jobs/nonexistent-job/logs")
        
        # Verify response
        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "JobNotFoundError"
        assert "nonexistent-job" in data["message"]
    
    @patch('api.queue.get_job_queue')
    def test_get_job_logs_empty_logs(self, mock_get_queue, client):
        """Test job logs retrieval for job with no logs"""
        # Mock job queue
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        
        # Mock job data
        job_data = JobData(
            id="test-job-123",
            status=JobStatus.QUEUED,
            created_at=datetime.now(timezone.utc),
            prompt="A test prompt"
        )
        mock_queue.get_job.return_value = job_data
        
        # Mock empty log lines
        mock_queue.get_job_logs.return_value = []
        
        # Make request
        response = client.get("/v1/jobs/test-job-123/logs")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["lines"] == []
    
    @patch('api.queue.get_job_queue')
    def test_get_job_logs_redis_error(self, mock_get_queue, client):
        """Test job logs retrieval when Redis fails"""
        # Mock job queue
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        mock_queue.get_job.side_effect = Exception("Redis connection failed")
        
        # Make request
        response = client.get("/v1/jobs/test-job-123/logs")
        
        # Verify response
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "RedisConnectionError"
        assert "Redis" in data["message"]


class TestJobStatusIntegration:
    """Integration tests for job status functionality"""
    
    @patch('api.queue.get_job_queue')
    def test_job_status_progression(self, mock_get_queue, client):
        """Test job status progression through different states"""
        # Mock job queue
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        
        base_time = datetime(2025, 8, 31, 12, 0, 0, tzinfo=timezone.utc)
        
        # Test queued status
        queued_job = JobData(
            id="test-job-123",
            status=JobStatus.QUEUED,
            progress=0,
            created_at=base_time,
            prompt="A test prompt"
        )
        mock_queue.get_job.return_value = queued_job
        
        response = client.get("/v1/jobs/test-job-123")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert data["progress"] == 0
        assert data["started_at"] is None
        
        # Test started status
        started_job = JobData(
            id="test-job-123",
            status=JobStatus.STARTED,
            progress=5,
            created_at=base_time,
            started_at=base_time.replace(minute=1),
            prompt="A test prompt"
        )
        mock_queue.get_job.return_value = started_job
        
        response = client.get("/v1/jobs/test-job-123")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["progress"] == 5
        assert data["started_at"] == "2025-08-31T12:01:00Z"
        
        # Test progress status
        progress_job = JobData(
            id="test-job-123",
            status=JobStatus.PROGRESS,
            progress=50,
            created_at=base_time,
            started_at=base_time.replace(minute=1),
            prompt="A test prompt"
        )
        mock_queue.get_job.return_value = progress_job
        
        response = client.get("/v1/jobs/test-job-123")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "progress"
        assert data["progress"] == 50
        
        # Test finished status
        finished_job = JobData(
            id="test-job-123",
            status=JobStatus.FINISHED,
            progress=100,
            created_at=base_time,
            started_at=base_time.replace(minute=1),
            finished_at=base_time.replace(minute=15),
            prompt="A test prompt",
            gcs_uri="gs://test-bucket/ttv-api/2025-08/test-job-123/final_video.mp4"
        )
        mock_queue.get_job.return_value = finished_job
        
        response = client.get("/v1/jobs/test-job-123")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "finished"
        assert data["progress"] == 100
        assert data["finished_at"] == "2025-08-31T12:15:00Z"
        assert data["gcs_uri"] == "gs://test-bucket/ttv-api/2025-08/test-job-123/final_video.mp4"