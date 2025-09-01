"""
Tests for the artifact URL endpoint.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from api.main import create_app
from api.models import JobData, JobStatus
from api.gcs_client import GCSClientError


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
    config.gcs.signed_url_expiration = 3600
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


@pytest.fixture
def finished_job_data():
    """Mock finished job data with GCS URI"""
    return JobData(
        id="test-job-123",
        status=JobStatus.FINISHED,
        created_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        prompt="A test prompt",
        gcs_uri="gs://test-bucket/test-prefix/2025-08/test-job-123/final_video.mp4"
    )


@pytest.fixture
def running_job_data():
    """Mock running job data without GCS URI"""
    return JobData(
        id="test-job-456",
        status=JobStatus.STARTED,
        created_at=datetime.now(timezone.utc),
        started_at=datetime.now(timezone.utc),
        prompt="A test prompt"
    )


class TestArtifactEndpoint:
    """Test artifact URL generation endpoint"""
    
    @patch('api.queue.get_job_queue')
    @patch('api.gcs_client.create_gcs_client')
    def test_get_artifact_success(self, mock_create_gcs_client, mock_get_queue, client, finished_job_data):
        """Test successful artifact URL generation"""
        # Mock job queue
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        mock_queue.get_job.return_value = finished_job_data
        
        # Mock GCS client
        mock_gcs_client = MagicMock()
        mock_create_gcs_client.return_value = mock_gcs_client
        mock_gcs_client.generate_signed_url.return_value = "https://storage.googleapis.com/signed-url"
        
        # Make request
        response = client.get("/v1/jobs/test-job-123/artifact")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["gcs_uri"] == finished_job_data.gcs_uri
        assert data["url"] == "https://storage.googleapis.com/signed-url"
        assert data["expires_in"] == 3600
        
        # Verify GCS client was called correctly
        mock_gcs_client.generate_signed_url.assert_called_once_with(
            gcs_uri=finished_job_data.gcs_uri,
            expiration_seconds=None
        )
    
    @patch('api.queue.get_job_queue')
    @patch('api.gcs_client.create_gcs_client')
    def test_get_artifact_with_custom_expiration(self, mock_create_gcs_client, mock_get_queue, client, finished_job_data):
        """Test artifact URL generation with custom expiration"""
        # Mock job queue
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        mock_queue.get_job.return_value = finished_job_data
        
        # Mock GCS client
        mock_gcs_client = MagicMock()
        mock_create_gcs_client.return_value = mock_gcs_client
        mock_gcs_client.generate_signed_url.return_value = "https://storage.googleapis.com/signed-url"
        
        # Make request with custom expiration
        response = client.get("/v1/jobs/test-job-123/artifact?expires_in=7200")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["expires_in"] == 7200
        
        # Verify GCS client was called with custom expiration
        mock_gcs_client.generate_signed_url.assert_called_once_with(
            gcs_uri=finished_job_data.gcs_uri,
            expiration_seconds=7200
        )
    
    @patch('api.queue.get_job_queue')
    @patch('api.gcs_client.create_gcs_client')
    def test_get_artifact_with_redirect(self, mock_create_gcs_client, mock_get_queue, client, finished_job_data):
        """Test artifact URL generation with redirect option"""
        # Mock job queue
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        mock_queue.get_job.return_value = finished_job_data
        
        # Mock GCS client
        mock_gcs_client = MagicMock()
        mock_create_gcs_client.return_value = mock_gcs_client
        mock_gcs_client.generate_signed_url.return_value = "https://storage.googleapis.com/signed-url"
        
        # Make request with redirect option
        response = client.get("/v1/jobs/test-job-123/artifact?redirect=true", follow_redirects=False)
        
        # Verify redirect response
        assert response.status_code == 302
        assert response.headers["location"] == "https://storage.googleapis.com/signed-url"
    
    @patch('api.queue.get_job_queue')
    def test_get_artifact_job_not_found(self, mock_get_queue, client):
        """Test artifact request for non-existent job"""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        mock_queue.get_job.return_value = None
        
        response = client.get("/v1/jobs/nonexistent-job/artifact")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "Job nonexistent-job not found" in data["message"]
    
    @patch('api.queue.get_job_queue')
    def test_get_artifact_job_not_finished(self, mock_get_queue, client, running_job_data):
        """Test artifact request for job that's not finished"""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        mock_queue.get_job.return_value = running_job_data
        
        response = client.get("/v1/jobs/test-job-456/artifact")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "Artifact for job test-job-456 not ready" in data["message"]
        assert data["details"]["current_status"] == "started"
    
    @patch('api.queue.get_job_queue')
    def test_get_artifact_finished_but_no_gcs_uri(self, mock_get_queue, client):
        """Test artifact request for finished job without GCS URI"""
        # Create finished job without GCS URI
        job_data = JobData(
            id="test-job-789",
            status=JobStatus.FINISHED,
            created_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            prompt="A test prompt"
            # No gcs_uri
        )
        
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        mock_queue.get_job.return_value = job_data
        
        response = client.get("/v1/jobs/test-job-789/artifact")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "finished but no artifact available" in data["message"]
    
    @patch('api.queue.get_job_queue')
    @patch('api.gcs_client.create_gcs_client')
    def test_get_artifact_gcs_client_init_failure(self, mock_create_gcs_client, mock_get_queue, client, finished_job_data):
        """Test artifact request when GCS client initialization fails"""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        mock_queue.get_job.return_value = finished_job_data
        
        # Mock GCS client initialization failure
        mock_create_gcs_client.side_effect = Exception("GCS credentials not found")
        
        response = client.get("/v1/jobs/test-job-123/artifact")
        
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "GCSError" in data["error"]
        assert "Failed to initialize storage client" in data["message"]
    
    @patch('api.queue.get_job_queue')
    @patch('api.gcs_client.create_gcs_client')
    def test_get_artifact_signed_url_generation_failure(self, mock_create_gcs_client, mock_get_queue, client, finished_job_data):
        """Test artifact request when signed URL generation fails"""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        mock_queue.get_job.return_value = finished_job_data
        
        # Mock GCS client with signed URL generation failure
        mock_gcs_client = MagicMock()
        mock_create_gcs_client.return_value = mock_gcs_client
        mock_gcs_client.generate_signed_url.side_effect = GCSClientError("Artifact not found")
        
        response = client.get("/v1/jobs/test-job-123/artifact")
        
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "GCSError" in data["error"]
        assert "Failed to generate download URL" in data["message"]
    
    @patch('api.queue.get_job_queue')
    def test_get_artifact_queue_connection_failure(self, mock_get_queue, client):
        """Test artifact request when Redis connection fails"""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        mock_queue.get_job.side_effect = Exception("Redis connection failed")
        
        response = client.get("/v1/jobs/test-job-123/artifact")
        
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "RedisConnectionError" in data["error"]
        assert "Failed to retrieve job data" in data["message"]


class TestArtifactValidation:
    """Test artifact endpoint parameter validation"""
    
    @patch('api.queue.get_job_queue')
    def test_get_artifact_expires_in_too_small(self, mock_get_queue, client):
        """Test artifact request with expires_in too small"""
        response = client.get("/v1/jobs/test-job-123/artifact?expires_in=30")
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "expires_in must be at least 60 seconds" in data["message"]
    
    @patch('api.queue.get_job_queue')
    def test_get_artifact_expires_in_too_large(self, mock_get_queue, client):
        """Test artifact request with expires_in too large"""
        response = client.get("/v1/jobs/test-job-123/artifact?expires_in=700000")
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "expires_in cannot exceed 604800 seconds" in data["message"]
    
    @patch('api.queue.get_job_queue')
    def test_get_artifact_invalid_expires_in_type(self, mock_get_queue, client):
        """Test artifact request with invalid expires_in type"""
        response = client.get("/v1/jobs/test-job-123/artifact?expires_in=invalid")
        
        assert response.status_code == 422  # FastAPI validation error
        data = response.json()
        assert "detail" in data
    
    @patch('api.queue.get_job_queue')
    def test_get_artifact_invalid_redirect_type(self, mock_get_queue, client):
        """Test artifact request with invalid redirect type"""
        response = client.get("/v1/jobs/test-job-123/artifact?redirect=invalid")
        
        assert response.status_code == 422  # FastAPI validation error
        data = response.json()
        assert "detail" in data


class TestArtifactEdgeCases:
    """Test edge cases for artifact endpoint"""
    
    @patch('api.queue.get_job_queue')
    @patch('api.gcs_client.create_gcs_client')
    def test_get_artifact_boundary_expiration_values(self, mock_create_gcs_client, mock_get_queue, client, finished_job_data):
        """Test artifact request with boundary expiration values"""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        mock_queue.get_job.return_value = finished_job_data
        
        mock_gcs_client = MagicMock()
        mock_create_gcs_client.return_value = mock_gcs_client
        mock_gcs_client.generate_signed_url.return_value = "https://storage.googleapis.com/signed-url"
        
        # Test minimum valid expiration (60 seconds)
        response = client.get("/v1/jobs/test-job-123/artifact?expires_in=60")
        assert response.status_code == 200
        data = response.json()
        assert data["expires_in"] == 60
        
        # Test maximum valid expiration (7 days)
        response = client.get("/v1/jobs/test-job-123/artifact?expires_in=604800")
        assert response.status_code == 200
        data = response.json()
        assert data["expires_in"] == 604800
    
    @patch('api.queue.get_job_queue')
    @patch('api.gcs_client.create_gcs_client')
    def test_get_artifact_with_failed_job(self, mock_create_gcs_client, mock_get_queue, client):
        """Test artifact request for failed job"""
        failed_job_data = JobData(
            id="test-job-failed",
            status=JobStatus.FAILED,
            created_at=datetime.now(timezone.utc),
            prompt="A test prompt",
            error="Pipeline execution failed"
        )
        
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        mock_queue.get_job.return_value = failed_job_data
        
        response = client.get("/v1/jobs/test-job-failed/artifact")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "Artifact for job test-job-failed not ready" in data["message"]
        assert data["details"]["current_status"] == "failed"
    
    @patch('api.queue.get_job_queue')
    @patch('api.gcs_client.create_gcs_client')
    def test_get_artifact_with_canceled_job(self, mock_create_gcs_client, mock_get_queue, client):
        """Test artifact request for canceled job"""
        canceled_job_data = JobData(
            id="test-job-canceled",
            status=JobStatus.CANCELED,
            created_at=datetime.now(timezone.utc),
            prompt="A test prompt"
        )
        
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        mock_queue.get_job.return_value = canceled_job_data
        
        response = client.get("/v1/jobs/test-job-canceled/artifact")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "Artifact for job test-job-canceled not ready" in data["message"]
        assert data["details"]["current_status"] == "canceled"
    
    @patch('api.queue.get_job_queue')
    @patch('api.gcs_client.create_gcs_client')
    def test_get_artifact_empty_job_id(self, mock_create_gcs_client, mock_get_queue, client):
        """Test artifact request with empty job ID"""
        # FastAPI will handle this as a 404 since the route won't match
        response = client.get("/v1/jobs//artifact")
        
        assert response.status_code == 404
    
    @patch('api.queue.get_job_queue')
    @patch('api.gcs_client.create_gcs_client')
    def test_get_artifact_special_characters_job_id(self, mock_create_gcs_client, mock_get_queue, client):
        """Test artifact request with special characters in job ID"""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        mock_queue.get_job.return_value = None  # Job not found
        
        # Test with URL-encoded special characters
        response = client.get("/v1/jobs/job%20with%20spaces/artifact")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "Job job with spaces not found" in data["message"]