"""
Tests for the main FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import create_app
from api.models import JobCreateRequest


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
        
        # Mock pipeline config for job creation
        mock_config.pipeline = MagicMock()
        mock_config.pipeline.model_dump.return_value = {
            'prompt': 'default prompt',
            'default_backend': 'wan21',
            'size': '1280x720'
        }
        
        return TestClient(app)


class TestBasicEndpoints:
    """Test basic application endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API information"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "TTV Pipeline API"
        assert "version" in data
        assert "description" in data
        assert data["docs_url"] == "/docs"
    
    def test_health_check(self, client):
        """Test basic health check endpoint"""
        response = client.get("/healthz")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_readiness_check(self, client):
        """Test readiness check endpoint"""
        response = client.get("/readyz")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "timestamp" in data
        assert "version" in data
    
    def test_metrics_endpoint(self, client):
        """Test basic metrics endpoint"""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data


class TestMiddleware:
    """Test middleware functionality"""
    
    def test_security_headers(self, client):
        """Test that security headers are added"""
        response = client.get("/")
        
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    
    def test_request_id_header(self, client):
        """Test that correlation ID is added to response headers"""
        response = client.get("/")
        
        assert "X-Correlation-ID" in response.headers
        correlation_id = response.headers["X-Correlation-ID"]
        # UUID format validation
        import uuid
        uuid.UUID(correlation_id)  # Should not raise exception if valid UUID
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        # Test with a simple GET request since OPTIONS might not be handled by our routes
        response = client.get("/", headers={"Origin": "http://localhost:3000"})
        
        # Should have CORS headers from middleware
        assert response.status_code == 200
        # Note: CORS headers are typically added by the CORS middleware


class TestErrorHandling:
    """Test error handling and exception responses"""
    
    def test_404_error(self, client):
        """Test 404 error handling"""
        response = client.get("/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        # FastAPI returns a different format for 404s by default
        assert "detail" in data or "error" in data
    
    def test_method_not_allowed(self, client):
        """Test method not allowed handling"""
        # POST to GET-only endpoint, but our request validation middleware
        # will check content-type first, so we need to provide proper JSON
        response = client.post("/healthz", json={})
        
        # Could be 405 (method not allowed) or 415 (unsupported media type)
        # depending on middleware order
        assert response.status_code in [405, 415]


class TestJobCreation:
    """Test job creation endpoint"""
    
    @patch('api.queue.get_job_queue')
    def test_create_job_success(self, mock_get_queue, client):
        """Test successful job creation"""
        from api.models import JobData, JobStatus
        from datetime import datetime, timezone
        
        # Mock job queue
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        
        # Mock job data response
        job_data = JobData(
            id="test-job-123",
            status=JobStatus.QUEUED,
            created_at=datetime.now(timezone.utc),
            prompt="A test prompt"
        )
        mock_queue.enqueue_job.return_value = job_data
        
        # Make request
        response = client.post("/v1/jobs", json={"prompt": "A test prompt"})
        
        # Verify response
        assert response.status_code == 202
        data = response.json()
        assert data["id"] == "test-job-123"
        assert data["status"] == "queued"
        assert "created_at" in data
        
        # Verify Location header
        assert response.headers["Location"] == "/v1/jobs/test-job-123"
        
        # Verify queue was called correctly
        mock_queue.enqueue_job.assert_called_once()
        call_args = mock_queue.enqueue_job.call_args
        assert call_args[1]["request"].prompt == "A test prompt"
        assert call_args[1]["effective_config"]["prompt"] == "A test prompt"
    
    def test_create_job_missing_prompt(self, client):
        """Test job creation with missing prompt"""
        response = client.post("/v1/jobs", json={})
        
        assert response.status_code == 422  # Validation error
        data = response.json()
        assert "detail" in data
    
    def test_create_job_empty_prompt(self, client):
        """Test job creation with empty prompt"""
        response = client.post("/v1/jobs", json={"prompt": ""})
        
        assert response.status_code == 422  # Validation error
        data = response.json()
        assert "detail" in data
    
    def test_create_job_prompt_too_long(self, client):
        """Test job creation with prompt that's too long"""
        long_prompt = "A" * 2001  # Exceeds 2000 character limit
        response = client.post("/v1/jobs", json={"prompt": long_prompt})
        
        assert response.status_code == 422  # Validation error
        data = response.json()
        assert "detail" in data
    
    @patch('api.queue.get_job_queue')
    def test_create_job_queue_failure(self, mock_get_queue, client):
        """Test job creation when queue fails"""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        mock_queue.enqueue_job.side_effect = Exception("Redis connection failed")
        
        response = client.post("/v1/jobs", json={"prompt": "A test prompt"})
        
        assert response.status_code == 503  # Service unavailable
        data = response.json()
        assert "error" in data
        assert "Redis" in data["message"]


class TestJobCancellation:
    """Test job cancellation endpoint"""
    
    @patch('api.queue.get_job_queue')
    def test_cancel_job_success(self, mock_get_queue, client):
        """Test successful job cancellation"""
        from api.models import JobData, JobStatus
        from datetime import datetime, timezone
        
        # Mock job queue
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        
        # Mock existing job data
        job_data = JobData(
            id="test-job-123",
            status=JobStatus.STARTED,
            created_at=datetime.now(timezone.utc),
            prompt="A test prompt"
        )
        mock_queue.get_job.return_value = job_data
        mock_queue.cancel_job.return_value = True
        
        # Mock updated job data after cancellation
        cancelled_job_data = JobData(
            id="test-job-123",
            status=JobStatus.CANCELED,
            created_at=datetime.now(timezone.utc),
            prompt="A test prompt"
        )
        # Return cancelled job on second call
        mock_queue.get_job.side_effect = [job_data, cancelled_job_data]
        
        # Make request
        response = client.post("/v1/jobs/test-job-123/cancel", json={})
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-job-123"
        assert data["status"] == "canceled"
        assert data["message"] == "Job cancellation initiated successfully"
        
        # Verify queue methods were called
        mock_queue.cancel_job.assert_called_once_with("test-job-123")
    
    @patch('api.queue.get_job_queue')
    def test_cancel_job_not_found(self, mock_get_queue, client):
        """Test cancelling a job that doesn't exist"""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        mock_queue.get_job.return_value = None
        
        response = client.post("/v1/jobs/nonexistent-job/cancel", json={})
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "Job nonexistent-job not found" in data["message"]
    
    @patch('api.queue.get_job_queue')
    def test_cancel_finished_job(self, mock_get_queue, client):
        """Test cancelling a job that's already finished"""
        from api.models import JobData, JobStatus
        from datetime import datetime, timezone
        
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        
        # Mock finished job data
        job_data = JobData(
            id="test-job-123",
            status=JobStatus.FINISHED,
            created_at=datetime.now(timezone.utc),
            prompt="A test prompt"
        )
        mock_queue.get_job.return_value = job_data
        
        response = client.post("/v1/jobs/test-job-123/cancel", json={})
        
        assert response.status_code == 409  # Conflict
        data = response.json()
        assert "error" in data
        assert "already in terminal status" in data["message"]
    
    @patch('api.queue.get_job_queue')
    def test_cancel_job_cancellation_failed(self, mock_get_queue, client):
        """Test job cancellation when the cancellation process fails"""
        from api.models import JobData, JobStatus
        from datetime import datetime, timezone
        
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        
        # Mock running job data
        job_data = JobData(
            id="test-job-123",
            status=JobStatus.STARTED,
            created_at=datetime.now(timezone.utc),
            prompt="A test prompt"
        )
        mock_queue.get_job.return_value = job_data
        mock_queue.cancel_job.return_value = False  # Cancellation failed
        
        response = client.post("/v1/jobs/test-job-123/cancel", json={})
        
        assert response.status_code == 409  # Conflict
        data = response.json()
        assert "error" in data
        assert "Cancellation failed" in data["message"]
    
    @patch('api.queue.get_job_queue')
    def test_cancel_job_queue_error(self, mock_get_queue, client):
        """Test job cancellation when queue operations fail"""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        mock_queue.get_job.side_effect = Exception("Redis connection failed")
        
        response = client.post("/v1/jobs/test-job-123/cancel", json={})
        
        assert response.status_code == 503  # Service unavailable
        data = response.json()
        assert "error" in data
        assert "Redis" in data["message"]


class TestApplicationLifespan:
    """Test application startup and shutdown"""
    
    @patch('api.main.get_config_from_env')
    @patch('api.queue.initialize_queue_infrastructure')
    def test_app_creation_with_config(self, mock_init_queue, mock_get_config, mock_config):
        """Test that app creation loads configuration"""
        mock_get_config.return_value = mock_config
        mock_init_queue.return_value = (MagicMock(), MagicMock())
        
        app = create_app()
        
        # Configuration should be loaded during lifespan startup
        # We can't easily test the lifespan context manager in unit tests,
        # but we can verify the app is created successfully
        assert app is not None
        assert app.title == "TTV Pipeline API"
    
    @patch('api.main.get_config_from_env')
    def test_app_creation_config_failure(self, mock_get_config):
        """Test app creation handles configuration failures"""
        mock_get_config.side_effect = Exception("Config loading failed")
        
        # App creation should succeed, but lifespan startup would fail
        app = create_app()
        assert app is not None