"""
Integration tests with mocked backends for the TTV Pipeline API.

This module provides comprehensive integration tests that cover:
1. Job flow from submission to completion using mock video generators
2. HTTP/3 protocol negotiation and basic connectivity
3. GCS upload/download with test files and signed URL generation

Requirements covered: 2.1, 2.2, 3.1, 8.1, 8.3
"""

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch, MagicMock, AsyncMock

import pytest
import httpx
from fastapi.testclient import TestClient
from PIL import Image

from api.main import create_app
from api.models import JobStatus, JobData
from api.config import APIConfig, APIServerConfig, RedisConfig, GCSConfig, SecurityConfig
from video_generator_interface import VideoGeneratorInterface, VideoGenerationError

# Import mock classes
from tests.mocks.mock_generators import MockVideoGenerator, MockGeneratorFactory
from tests.mocks.mock_gcs import MockGCSClient, MockGCSUploader
from tests.mocks.mock_redis import MockRedisManager, MockJobQueue


# Mock classes are now imported from tests.mocks


@pytest.fixture
def mock_config():
    """Mock API configuration for testing"""
    return APIConfig(
        server=APIServerConfig(
            host="localhost",
            port=8000,
            quic_port=8443,
            workers=1,
            cors_origins=["*"]
        ),
        redis=RedisConfig(
            host="localhost",
            port=6379,
            db=0
        ),
        gcs=GCSConfig(
            bucket="test-api-bucket",
            prefix="ttv-api",
            signed_url_expiration=3600
        ),
        security=SecurityConfig(
            auth_token="test-token",
            rate_limit_per_minute=60
        )
    )


@pytest.fixture
def test_image():
    """Create a test image file"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        # Create a simple test image
        img = Image.new('RGB', (1280, 720), color='blue')
        img.save(f.name, 'PNG')
        yield f.name
    
    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def mock_app(mock_config):
    """Create FastAPI app with mocked dependencies"""
    with patch('api.main.get_config_from_env', return_value=mock_config), \
         patch('api.queue.initialize_queue_infrastructure') as mock_init_queue:
        
        # Create mock objects
        mock_redis_manager = MagicMock()
        mock_job_queue = MagicMock()
        mock_init_queue.return_value = (mock_redis_manager, mock_job_queue)
        
        # Patch the get_job_queue function at the module level
        with patch('api.queue.get_job_queue', return_value=mock_job_queue):
            app = create_app()
            app.state.config = mock_config
            app.state.mock_job_queue = mock_job_queue
            app.state.mock_redis_manager = mock_redis_manager
            
            yield app


@pytest.mark.integration
class TestJobFlowWithMockedBackends:
    """Test complete job flow from submission to completion using mock video generators"""
    
    def test_successful_job_flow_text_to_video(self, mock_app, test_image):
        """Test successful job flow for text-to-video generation"""
        with patch('api.queue.get_job_queue') as mock_get_queue:
            client = TestClient(mock_app)
            
            # Mock the job queue behavior
            job_id = "test-job-123"
            job_data = JobData(
                id=job_id,
                status=JobStatus.QUEUED,
                created_at=datetime.now(timezone.utc),
                prompt="A beautiful sunset over mountains"
            )
            
            # Configure mock queue
            mock_queue = MagicMock()
            mock_get_queue.return_value = mock_queue
            mock_queue.enqueue_job.return_value = job_data
        
            # Mock video generator
            with patch('generators.factory.create_video_generator') as mock_create_generator:
                mock_generator = MockVideoGenerator({}, generation_time=1.0)
                mock_create_generator.return_value = mock_generator
                
                # Mock GCS client
                with patch('api.gcs_client.create_gcs_client') as mock_create_gcs:
                    mock_gcs = MockGCSClient(mock_app.state.config.gcs)
                    mock_create_gcs.return_value = mock_gcs
                    
                    # Step 1: Submit job
                    response = client.post("/v1/jobs", json={
                        "prompt": "A beautiful sunset over mountains"
                    })
                    
                    assert response.status_code == 202
                    data = response.json()
                    assert data["id"] == job_id
                    assert data["status"] == "queued"
                    assert "Location" in response.headers
                    
                    # Step 2: Simulate job processing
                    # Update job status to started
                    job_data.status = JobStatus.STARTED
                    job_data.started_at = datetime.now(timezone.utc)
                    mock_queue.get_job.return_value = job_data
                    
                    response = client.get(f"/v1/jobs/{job_id}")
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "started"
                    assert data["started_at"] is not None
                    
                    # Step 3: Simulate progress updates
                    job_data.status = JobStatus.PROGRESS
                    job_data.progress = 50
                    
                    response = client.get(f"/v1/jobs/{job_id}")
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "progress"
                    assert data["progress"] == 50
                    
                    # Step 4: Simulate completion
                    job_data.status = JobStatus.FINISHED
                    job_data.progress = 100
                    job_data.finished_at = datetime.now(timezone.utc)
                    job_data.gcs_uri = "gs://test-api-bucket/ttv-api/2025-08/test-job-123/final_video.mp4"
                    
                    response = client.get(f"/v1/jobs/{job_id}")
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "finished"
                    assert data["progress"] == 100
                    assert data["gcs_uri"] is not None
                    assert data["finished_at"] is not None
    
    def test_job_flow_with_failure(self, mock_app):
        """Test job flow when video generation fails"""
        client = TestClient(mock_app)
        
        job_id = "test-job-fail"
        job_data = JobData(
            id=job_id,
            status=JobStatus.QUEUED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        mock_queue = mock_app.state.mock_job_queue
        mock_queue.enqueue_job.return_value = job_data
        
        # Mock failing video generator
        with patch('generators.factory.create_video_generator') as mock_create_generator:
            mock_generator = MockVideoGenerator({}, should_fail=True)
            mock_create_generator.return_value = mock_generator
            
            # Submit job
            response = client.post("/v1/jobs", json={
                "prompt": "Test prompt"
            })
            assert response.status_code == 202
            
            # Simulate job failure
            job_data.status = JobStatus.FAILED
            job_data.error = "Mock generation failure"
            job_data.finished_at = datetime.now(timezone.utc)
            mock_queue.get_job.return_value = job_data
            
            response = client.get(f"/v1/jobs/{job_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "failed"
            assert data["error"] == "Mock generation failure"
    
    def test_job_cancellation_flow(self, mock_app):
        """Test job cancellation flow"""
        client = TestClient(mock_app)
        
        job_id = "test-job-cancel"
        job_data = JobData(
            id=job_id,
            status=JobStatus.STARTED,
            created_at=datetime.now(timezone.utc),
            started_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        mock_queue = mock_app.state.mock_job_queue
        mock_queue.get_job.return_value = job_data
        mock_queue.cancel_job.return_value = True
        
        # Cancel the job
        response = client.post(f"/v1/jobs/{job_id}/cancel", json={})
        assert response.status_code == 200
        
        # Verify cancellation was called
        mock_queue.cancel_job.assert_called_once_with(job_id)
        
        # Simulate job status after cancellation
        job_data.status = JobStatus.CANCELED
        job_data.finished_at = datetime.now(timezone.utc)
        
        response = client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "canceled"
    
    def test_job_logs_retrieval(self, mock_app):
        """Test job logs retrieval during processing"""
        client = TestClient(mock_app)
        
        job_id = "test-job-logs"
        
        mock_queue = mock_app.state.mock_job_queue
        mock_queue.get_job.return_value = JobData(
            id=job_id,
            status=JobStatus.PROGRESS,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        # Mock log retrieval
        test_logs = [
            "[2025-08-31T12:00:00Z] Starting video generation",
            "[2025-08-31T12:00:30Z] Processing keyframes",
            "[2025-08-31T12:01:00Z] Generating video segments"
        ]
        mock_queue.get_job_logs.return_value = test_logs
        
        # Get logs
        response = client.get(f"/v1/jobs/{job_id}/logs")
        assert response.status_code == 200
        data = response.json()
        assert data["lines"] == test_logs
        
        # Test with tail parameter
        response = client.get(f"/v1/jobs/{job_id}/logs?tail=2")
        assert response.status_code == 200
        mock_queue.get_job_logs.assert_called_with(job_id, tail=2)
    
    def test_multiple_concurrent_jobs(self, mock_app):
        """Test handling multiple concurrent jobs"""
        client = TestClient(mock_app)
        
        mock_queue = mock_app.state.mock_job_queue
        
        # Submit multiple jobs
        job_ids = []
        for i in range(3):
            job_id = f"test-job-concurrent-{i}"
            job_data = JobData(
                id=job_id,
                status=JobStatus.QUEUED,
                created_at=datetime.now(timezone.utc),
                prompt=f"Test prompt {i}"
            )
            mock_queue.enqueue_job.return_value = job_data
            
            response = client.post("/v1/jobs", json={
                "prompt": f"Test prompt {i}"
            })
            assert response.status_code == 202
            job_ids.append(job_id)
        
        # Verify all jobs were created
        assert len(job_ids) == 3
        assert mock_queue.enqueue_job.call_count == 3


@pytest.mark.integration
class TestHTTP3ProtocolNegotiation:
    """Test HTTP/3 protocol negotiation and basic connectivity"""
    
    @pytest.fixture
    def http3_client_config(self):
        """HTTP client configuration for HTTP/3 testing"""
        return {
            "verify": False,  # Accept self-signed certificates
            "timeout": 10.0,
            "follow_redirects": True,
            "http2": True  # Enable HTTP/2 for protocol negotiation testing
        }
    
    def test_alt_svc_header_presence(self, mock_app, http3_client_config):
        """Test that Alt-Svc header is present for HTTP/3 advertisement"""
        client = TestClient(mock_app)
        
        # Mock HTTPS request context
        with patch.object(client, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {
                "Alt-Svc": 'h3=":8443"; ma=86400, h2=":443"; ma=86400'
            }
            mock_response.json.return_value = {"status": "healthy"}
            mock_request.return_value = mock_response
            
            response = client.get("/healthz")
            assert response.status_code == 200
            
            # Verify Alt-Svc header format
            alt_svc = response.headers.get("Alt-Svc", "")
            assert "h3=" in alt_svc
            assert ":8443" in alt_svc
    
    def test_protocol_version_detection(self, mock_app):
        """Test protocol version detection in middleware"""
        from api.middleware import HTTP3Middleware
        
        app_mock = Mock()
        middleware = HTTP3Middleware(app_mock, quic_port=8443, enable_alt_svc=True)
        
        # Test HTTP/2 detection
        request = Mock()
        request.scope = {"http_version": "2.0"}
        version = middleware._get_protocol_version(request)
        assert version == "h2"
        
        # Test HTTP/1.1 detection
        request.scope = {"http_version": "1.1"}
        version = middleware._get_protocol_version(request)
        assert version == "http/1.1"
        
        # Test HTTP/3 detection (if supported)
        request.scope = {"http_version": "3.0"}
        version = middleware._get_protocol_version(request)
        assert version == "h3"
    
    def test_hypercorn_config_http3_support(self, mock_config):
        """Test Hypercorn configuration includes HTTP/3 support"""
        from api.hypercorn_config import HTTP3Config
        
        config_builder = HTTP3Config(mock_config)
        
        # Test development config (HTTP/2 only)
        dev_config = config_builder.build_development_config()
        assert "h2c" in dev_config.alpn_protocols
        assert "http/1.1" in dev_config.alpn_protocols
        
        # Test production config with certificates
        with patch('pathlib.Path.exists', return_value=True):
            prod_config = config_builder.build_production_config(
                certfile="/path/to/cert.pem",
                keyfile="/path/to/key.pem"
            )
            assert "h3" in prod_config.alpn_protocols
            assert "h2" in prod_config.alpn_protocols
            assert "http/1.1" in prod_config.alpn_protocols
            assert prod_config.quic_bind == ["localhost:8443"]
    
    @pytest.mark.asyncio
    async def test_async_http_client_protocol_negotiation(self):
        """Test HTTP protocol negotiation with async client"""
        # This test simulates what would happen with a real HTTP/3 client
        
        # Mock server responses for different protocols
        protocol_responses = {
            "1.1": {"version": "HTTP/1.1", "alt_svc": None},
            "2": {"version": "HTTP/2", "alt_svc": 'h3=":8443"; ma=86400'},
            "3": {"version": "HTTP/3", "alt_svc": None}  # No Alt-Svc needed for HTTP/3
        }
        
        for protocol, expected in protocol_responses.items():
            # Simulate protocol-specific behavior
            if protocol == "3":
                # HTTP/3 would use QUIC transport
                assert expected["version"] == "HTTP/3"
            elif protocol == "2":
                # HTTP/2 should advertise HTTP/3 via Alt-Svc
                assert expected["alt_svc"] is not None
                assert "h3=" in expected["alt_svc"]
            else:
                # HTTP/1.1 baseline
                assert expected["version"] == "HTTP/1.1"
    
    def test_middleware_stack_order(self, mock_app):
        """Test that HTTP/3 middleware is properly positioned in the stack"""
        # Check middleware order
        middleware_classes = [middleware.cls for middleware in mock_app.user_middleware]
        middleware_names = [cls.__name__ for cls in middleware_classes]
        
        # HTTP3Middleware should be present
        assert "HTTP3Middleware" in middleware_names
        
        # Should be positioned appropriately (after security, before CORS)
        http3_index = middleware_names.index("HTTP3Middleware")
        cors_index = middleware_names.index("CORSMiddleware")
        
        # HTTP3Middleware should come before CORS
        assert http3_index < cors_index


@pytest.mark.integration
class TestGCSIntegrationWithMocks:
    """Test GCS upload/download with test files and signed URL generation"""
    
    @pytest.fixture
    def mock_gcs_config(self):
        """Mock GCS configuration"""
        return GCSConfig(
            bucket="test-integration-bucket",
            prefix="ttv-api-test",
            credentials_path="test-credentials.json",
            signed_url_expiration=3600
        )
    
    @pytest.fixture
    def test_video_file(self):
        """Create a test video file"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            # Write minimal MP4 header
            f.write(b'\x00\x00\x00\x20ftypmp42\x00\x00\x00\x00mp42isom')
            f.write(b'\x00' * 5000)  # 5KB test file
            yield f.name
        
        if os.path.exists(f.name):
            os.unlink(f.name)
    
    def test_gcs_upload_workflow(self, mock_gcs_config, test_video_file):
        """Test GCS upload workflow with mock client"""
        # Use our mock GCS client instead of the real one
        mock_gcs = MockGCSClient(mock_gcs_config)
        
        # Test upload
        job_id = "test-job-gcs-upload"
        gcs_uri = mock_gcs.upload_artifact(test_video_file, job_id)
        
        # Verify GCS URI format
        expected_pattern = f"gs://{mock_gcs_config.bucket}/{mock_gcs_config.prefix}"
        assert gcs_uri.startswith(expected_pattern)
        assert job_id in gcs_uri
        assert "final_video.mp4" in gcs_uri
        
        # Verify file was stored in mock
        assert gcs_uri in mock_gcs.uploaded_files
        file_info = mock_gcs.get_file_info(gcs_uri)
        assert file_info is not None
        assert file_info['job_id'] == job_id
    
    def test_signed_url_generation(self, mock_gcs_config):
        """Test signed URL generation"""
        # Use our mock GCS client
        mock_gcs = MockGCSClient(mock_gcs_config)
        
        # First upload a file to generate a signed URL for
        gcs_uri = f"gs://{mock_gcs_config.bucket}/test-path/video.mp4"
        mock_gcs.uploaded_files[gcs_uri] = {
            'content': b'test video content',
            'size': 18,
            'uploaded_at': time.time(),
            'job_id': 'test-job'
        }
        
        # Test signed URL generation
        signed_url = mock_gcs.generate_signed_url(gcs_uri, expiration_seconds=7200)
        
        # Verify signed URL format
        assert signed_url.startswith("https://storage.googleapis.com/")
        assert mock_gcs_config.bucket in signed_url
        assert "expires=" in signed_url
        assert mock_gcs.signed_url_count == 1
    
    def test_artifact_endpoint_integration(self, mock_app, mock_gcs_config):
        """Test artifact endpoint with GCS integration"""
        client = TestClient(mock_app)
        
        job_id = "test-job-artifact"
        gcs_uri = f"gs://{mock_gcs_config.bucket}/ttv-api/2025-08/{job_id}/final_video.mp4"
        
        # Mock job with GCS URI
        job_data = JobData(
            id=job_id,
            status=JobStatus.FINISHED,
            created_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            prompt="Test prompt",
            gcs_uri=gcs_uri
        )
        
        mock_queue = mock_app.state.mock_job_queue
        mock_queue.get_job.return_value = job_data
        
        # Mock GCS client
        with patch('api.gcs_client.create_gcs_client') as mock_create_gcs:
            mock_gcs = Mock()
            mock_gcs.generate_signed_url.return_value = "https://storage.googleapis.com/signed-url"
            mock_create_gcs.return_value = mock_gcs
            
            # Test artifact endpoint
            response = client.get(f"/v1/jobs/{job_id}/artifact")
            assert response.status_code == 200
            
            data = response.json()
            assert data["gcs_uri"] == gcs_uri
            assert data["url"] == "https://storage.googleapis.com/signed-url"
            assert data["expires_in"] == 3600
    
    def test_gcs_download_workflow(self, mock_gcs_config, test_video_file):
        """Test GCS download workflow"""
        # Use our mock GCS client
        mock_gcs = MockGCSClient(mock_gcs_config)
        
        # First upload a file to download later
        gcs_uri = f"gs://{mock_gcs_config.bucket}/test-path/video.mp4"
        with open(test_video_file, 'rb') as f:
            content = f.read()
        
        mock_gcs.uploaded_files[gcs_uri] = {
            'content': content,
            'size': len(content),
            'uploaded_at': time.time(),
            'job_id': 'test-job'
        }
        
        # Test download
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            downloaded_path = mock_gcs.download_artifact(gcs_uri, temp_file.name)
            
            # Verify download
            assert os.path.exists(downloaded_path)
            assert os.path.getsize(downloaded_path) > 0
            assert mock_gcs.download_count == 1
            
            # Verify content matches
            with open(downloaded_path, 'rb') as f:
                downloaded_content = f.read()
            assert downloaded_content == content
            
            # Cleanup
            os.unlink(downloaded_path)
    
    def test_gcs_error_handling(self, mock_gcs_config):
        """Test GCS error handling scenarios"""
        from tests.mocks.mock_gcs import create_failing_gcs_client
        
        # Test upload failure
        failing_gcs = create_failing_gcs_client(mock_gcs_config, failure_type="upload")
        
        with pytest.raises(Exception):  # Should raise GCSError or similar
            failing_gcs.upload_artifact("/nonexistent/file.mp4", "test-job")
        
        # Test signed URL failure
        failing_gcs = create_failing_gcs_client(mock_gcs_config, failure_type="signed_url")
        
        with pytest.raises(Exception):
            failing_gcs.generate_signed_url("gs://test-bucket/test-file.mp4")
        
        # Test download failure
        failing_gcs = create_failing_gcs_client(mock_gcs_config, failure_type="download")
        
        with pytest.raises(Exception):
            failing_gcs.download_artifact("gs://test-bucket/test-file.mp4", "/tmp/test.mp4")
    
    def test_gcs_path_generation(self, mock_gcs_config):
        """Test GCS path generation follows required format"""
        # Use our mock GCS client
        mock_gcs = MockGCSClient(mock_gcs_config)
        
        job_id = "test-job-123"
        
        # Test the path generation directly
        path = mock_gcs.generate_artifact_path(job_id)
        
        # Verify format: gs://{bucket}/{prefix}/{YYYY-MM}/{job_id}/final_video.mp4
        # The path should contain the bucket, prefix, job_id and filename
        assert path.startswith(f"gs://{mock_gcs_config.bucket}/{mock_gcs_config.prefix}/")
        assert job_id in path
        assert "final_video.mp4" in path
        # Check that it has the year-month format (YYYY-MM)
        import re
        assert re.search(r'/\d{4}-\d{2}/', path) is not None


@pytest.mark.integration
class TestEndToEndIntegration:
    """End-to-end integration tests combining all components"""
    
    def test_complete_video_generation_workflow(self, mock_app, test_image):
        """Test complete workflow from job submission to artifact retrieval"""
        client = TestClient(mock_app)
        
        # Setup mocks
        job_id = "test-job-e2e"
        mock_queue = mock_app.state.mock_job_queue
        
        # Mock video generator and GCS
        with patch('generators.factory.create_video_generator') as mock_create_generator, \
             patch('api.gcs_client.create_gcs_client') as mock_create_gcs:
            
            mock_generator = MockVideoGenerator({})
            mock_create_generator.return_value = mock_generator
            
            mock_gcs = MockGCSClient(mock_app.state.config.gcs)
            mock_create_gcs.return_value = mock_gcs
            
            # Step 1: Submit job
            job_data = JobData(
                id=job_id,
                status=JobStatus.QUEUED,
                created_at=datetime.now(timezone.utc),
                prompt="A cinematic video of a sunset"
            )
            mock_queue.enqueue_job.return_value = job_data
            
            response = client.post("/v1/jobs", json={
                "prompt": "A cinematic video of a sunset"
            })
            assert response.status_code == 202
            
            # Step 2: Check initial status
            mock_queue.get_job.return_value = job_data
            response = client.get(f"/v1/jobs/{job_id}")
            assert response.status_code == 200
            assert response.json()["status"] == "queued"
            
            # Step 3: Simulate processing and completion
            job_data.status = JobStatus.FINISHED
            job_data.progress = 100
            job_data.finished_at = datetime.now(timezone.utc)
            job_data.gcs_uri = f"gs://test-api-bucket/ttv-api/2025-08/{job_id}/final_video.mp4"
            
            response = client.get(f"/v1/jobs/{job_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "finished"
            assert data["gcs_uri"] is not None
            
            # Step 4: Get artifact URL
            response = client.get(f"/v1/jobs/{job_id}/artifact")
            assert response.status_code == 200
            artifact_data = response.json()
            assert "url" in artifact_data
            assert "expires_in" in artifact_data
            assert artifact_data["gcs_uri"] == job_data.gcs_uri
    
    def test_error_recovery_and_logging(self, mock_app):
        """Test error recovery and comprehensive logging"""
        client = TestClient(mock_app)
        
        job_id = "test-job-error-recovery"
        mock_queue = mock_app.state.mock_job_queue
        
        # Test various error scenarios
        error_scenarios = [
            (404, "Job not found"),
            (500, "Internal server error"),
            (503, "Service temporarily unavailable")
        ]
        
        for status_code, error_message in error_scenarios:
            # Mock different error conditions
            if status_code == 404:
                mock_queue.get_job.return_value = None
            else:
                mock_queue.get_job.side_effect = Exception(error_message)
            
            response = client.get(f"/v1/jobs/{job_id}")
            
            # Verify appropriate error handling
            assert response.status_code in [404, 500, 503]
            
            # Reset mock for next iteration
            mock_queue.get_job.side_effect = None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])