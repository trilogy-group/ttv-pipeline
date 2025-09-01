"""
Tests for health and monitoring endpoints.

This module tests the health check, readiness check, and metrics endpoints
to ensure proper monitoring and observability functionality.
"""

import json
import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.config import APIConfig, APIServerConfig, RedisConfig, GCSConfig, SecurityConfig
from api.routes.health import reset_metrics, record_job_processed, record_request


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing"""
    return APIConfig(
        server=APIServerConfig(
            host="localhost",
            port=8000,
            quic_port=8443,
            workers=2,
            cors_origins=["*"]
        ),
        redis=RedisConfig(
            host="localhost",
            port=6379,
            db=0
        ),
        gcs=GCSConfig(
            bucket="test-bucket",
            prefix="test-prefix",
            credentials_path=None,
            signed_url_expiration=3600
        ),
        security=SecurityConfig(
            auth_token=None,
            rate_limit_per_minute=60
        ),
        pipeline_config={}
    )


@pytest.fixture
def app_with_config(mock_config):
    """Create FastAPI app with mock configuration"""
    app = create_app()
    app.state.config = mock_config
    return app


@pytest.fixture
def client(app_with_config):
    """Create test client"""
    return TestClient(app_with_config)


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_healthz_basic(self, client):
        """Test basic health check endpoint"""
        response = client.get("/healthz")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"
        assert "components" in data
        assert data["components"]["api"] == "healthy"
        
        # Verify timestamp format
        timestamp = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
        assert isinstance(timestamp, datetime)
    
    def test_healthz_always_succeeds(self, client):
        """Test that health check always returns success"""
        # Make multiple requests to ensure consistency
        for _ in range(5):
            response = client.get("/healthz")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"
    
    @patch('api.routes.health.get_redis_manager')
    @patch('api.routes.health.GCSClient')
    @patch('api.routes.health.get_job_queue')
    def test_readyz_all_healthy(self, mock_get_job_queue, mock_gcs_client, mock_get_redis_manager, client):
        """Test readiness check when all components are healthy"""
        # Mock Redis manager
        mock_redis_manager = Mock()
        mock_redis_manager.test_connection.return_value = True
        mock_get_redis_manager.return_value = mock_redis_manager
        
        # Mock GCS client
        mock_gcs_instance = Mock()
        mock_gcs_instance.test_connection = AsyncMock(return_value=True)
        mock_gcs_client.return_value = mock_gcs_instance
        
        # Mock job queue
        mock_job_queue = Mock()
        mock_job_queue.get_queue_stats.return_value = {"queued_jobs": 0, "started_jobs": 1}
        mock_get_job_queue.return_value = mock_job_queue
        
        response = client.get("/readyz")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "ready"
        assert data["version"] == "1.0.0"
        assert data["components"]["redis"] == "healthy"
        assert data["components"]["gcs"] == "healthy"
        assert data["components"]["workers"] == "healthy"
        assert data["components"]["api"] == "healthy"
    
    @patch('api.routes.health.get_redis_manager')
    @patch('api.routes.health.GCSClient')
    @patch('api.routes.health.get_job_queue')
    def test_readyz_redis_unhealthy(self, mock_get_job_queue, mock_gcs_client, mock_get_redis_manager, client):
        """Test readiness check when Redis is unhealthy"""
        # Mock Redis manager (unhealthy)
        mock_redis_manager = Mock()
        mock_redis_manager.test_connection.return_value = False
        mock_get_redis_manager.return_value = mock_redis_manager
        
        # Mock GCS client (healthy)
        mock_gcs_instance = Mock()
        mock_gcs_instance.test_connection = AsyncMock(return_value=True)
        mock_gcs_client.return_value = mock_gcs_instance
        
        # Mock job queue (healthy)
        mock_job_queue = Mock()
        mock_job_queue.get_queue_stats.return_value = {"queued_jobs": 0}
        mock_get_job_queue.return_value = mock_job_queue
        
        response = client.get("/readyz")
        
        assert response.status_code == 503
        data = response.json()
        
        assert data["status"] == "not_ready"
        assert data["components"]["redis"] == "unhealthy"
        assert data["components"]["gcs"] == "healthy"
        assert data["components"]["workers"] == "healthy"
    
    @patch('api.routes.health.get_redis_manager')
    @patch('api.routes.health.GCSClient')
    @patch('api.routes.health.get_job_queue')
    def test_readyz_gcs_unhealthy(self, mock_get_job_queue, mock_gcs_client, mock_get_redis_manager, client):
        """Test readiness check when GCS is unhealthy"""
        # Mock Redis manager (healthy)
        mock_redis_manager = Mock()
        mock_redis_manager.test_connection.return_value = True
        mock_get_redis_manager.return_value = mock_redis_manager
        
        # Mock GCS client (unhealthy)
        mock_gcs_instance = Mock()
        mock_gcs_instance.test_connection = AsyncMock(return_value=False)
        mock_gcs_client.return_value = mock_gcs_instance
        
        # Mock job queue (healthy)
        mock_job_queue = Mock()
        mock_job_queue.get_queue_stats.return_value = {"queued_jobs": 0}
        mock_get_job_queue.return_value = mock_job_queue
        
        response = client.get("/readyz")
        
        assert response.status_code == 503
        data = response.json()
        
        assert data["status"] == "not_ready"
        assert data["components"]["redis"] == "healthy"
        assert data["components"]["gcs"] == "unhealthy"
        assert data["components"]["workers"] == "healthy"
    
    @patch('api.routes.health.get_redis_manager')
    @patch('api.routes.health.GCSClient')
    @patch('api.routes.health.get_job_queue')
    def test_readyz_workers_unhealthy(self, mock_get_job_queue, mock_gcs_client, mock_get_redis_manager, client):
        """Test readiness check when workers are unhealthy"""
        # Mock Redis manager (healthy)
        mock_redis_manager = Mock()
        mock_redis_manager.test_connection.return_value = True
        mock_get_redis_manager.return_value = mock_redis_manager
        
        # Mock GCS client (healthy)
        mock_gcs_instance = Mock()
        mock_gcs_instance.test_connection = AsyncMock(return_value=True)
        mock_gcs_client.return_value = mock_gcs_instance
        
        # Mock job queue (unhealthy - returns None)
        mock_get_job_queue.return_value.get_queue_stats.return_value = None
        
        response = client.get("/readyz")
        
        assert response.status_code == 503
        data = response.json()
        
        assert data["status"] == "not_ready"
        assert data["components"]["redis"] == "healthy"
        assert data["components"]["gcs"] == "healthy"
        assert data["components"]["workers"] == "unhealthy"
    
    @patch('api.routes.health.get_redis_manager')
    def test_readyz_redis_exception(self, mock_get_redis_manager, client):
        """Test readiness check when Redis check raises exception"""
        # Mock Redis manager to raise exception
        mock_get_redis_manager.side_effect = Exception("Redis connection failed")
        
        response = client.get("/readyz")
        
        assert response.status_code == 503
        data = response.json()
        
        assert data["status"] == "not_ready"
        assert data["components"]["redis"] == "unhealthy"


class TestMetricsEndpoints:
    """Test metrics endpoints"""
    
    def setup_method(self):
        """Reset metrics before each test"""
        reset_metrics()
    
    @patch('api.routes.health.get_job_queue')
    def test_metrics_json_basic(self, mock_get_job_queue, client):
        """Test JSON metrics endpoint with basic data"""
        # Mock job queue stats
        mock_job_queue = Mock()
        mock_job_queue.get_queue_stats.return_value = {
            "queued_jobs": 5,
            "started_jobs": 2,
            "finished_jobs": 10,
            "failed_jobs": 1
        }
        mock_get_job_queue.return_value = mock_job_queue
        
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["active_jobs"] == 2  # started_jobs
        assert data["queued_jobs"] == 5
        assert data["total_jobs_processed"] == 0  # No jobs recorded yet
        assert data["average_processing_time"] == 0.0
        assert data["uptime_seconds"] >= 0
    
    @patch('api.routes.health.get_job_queue')
    def test_metrics_json_with_processed_jobs(self, mock_get_job_queue, client):
        """Test JSON metrics endpoint with processed jobs"""
        # Record some job processing times
        record_job_processed(10.5)
        record_job_processed(15.2)
        record_job_processed(8.3)
        
        # Mock job queue stats
        mock_job_queue = Mock()
        mock_job_queue.get_queue_stats.return_value = {
            "queued_jobs": 3,
            "started_jobs": 1,
        }
        mock_get_job_queue.return_value = mock_job_queue
        
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["active_jobs"] == 1
        assert data["queued_jobs"] == 3
        assert data["total_jobs_processed"] == 3
        
        # Check average processing time
        expected_avg = (10.5 + 15.2 + 8.3) / 3
        assert abs(data["average_processing_time"] - expected_avg) < 0.01
    
    @patch('api.routes.health.get_job_queue')
    def test_metrics_json_queue_unavailable(self, mock_get_job_queue, client):
        """Test JSON metrics endpoint when queue is unavailable"""
        # Mock job queue to raise exception
        mock_get_job_queue.side_effect = Exception("Queue unavailable")
        
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return basic metrics even when queue is unavailable
        assert data["active_jobs"] == 0
        assert data["queued_jobs"] == 0
        assert data["total_jobs_processed"] == 0
        assert data["average_processing_time"] == 0.0
        assert data["uptime_seconds"] >= 0
    
    @patch('api.routes.health.get_job_queue')
    @patch('api.routes.health.get_redis_manager')
    @patch('api.routes.health.GCSClient')
    def test_metrics_prometheus_format(self, mock_gcs_client, mock_get_redis_manager, mock_get_job_queue, client):
        """Test Prometheus metrics endpoint format"""
        # Record some metrics
        record_job_processed(12.0)
        record_request()
        record_request()
        
        # Mock dependencies
        mock_redis_manager = Mock()
        mock_redis_manager.test_connection.return_value = True
        mock_get_redis_manager.return_value = mock_redis_manager
        
        mock_gcs_instance = Mock()
        mock_gcs_instance.test_connection = AsyncMock(return_value=True)
        mock_gcs_client.return_value = mock_gcs_instance
        
        mock_job_queue = Mock()
        mock_job_queue.get_queue_stats.return_value = {
            "queued_jobs": 2,
            "started_jobs": 1,
            "finished_jobs": 5,
            "failed_jobs": 0
        }
        mock_get_job_queue.return_value = mock_job_queue
        
        response = client.get("/metrics/prometheus")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        
        content = response.text
        
        # Check for required Prometheus metrics
        assert "ttv_api_info" in content
        assert 'version="1.0.0"' in content
        assert "ttv_api_uptime_seconds" in content
        assert "ttv_api_jobs_total 1" in content  # One job processed
        assert "ttv_api_requests_total 2" in content  # Two requests recorded
        assert "ttv_api_job_processing_time_seconds 12.00" in content
        
        # Check queue metrics
        assert 'ttv_api_queue_jobs{status="queued"} 2' in content
        assert 'ttv_api_queue_jobs{status="started"} 1' in content
        assert 'ttv_api_queue_jobs{status="finished"} 5' in content
        assert 'ttv_api_queue_jobs{status="failed"} 0' in content
        
        # Check health metrics
        assert 'ttv_api_health_status{component="redis"} 1' in content
        assert 'ttv_api_health_status{component="gcs"} 1' in content
        assert 'ttv_api_health_status{component="api"} 1' in content
    
    @patch('api.routes.health.get_job_queue')
    def test_metrics_prometheus_error_handling(self, mock_get_job_queue, client):
        """Test Prometheus metrics endpoint error handling"""
        # Mock job queue to raise exception
        mock_get_job_queue.side_effect = Exception("Queue error")
        
        response = client.get("/metrics/prometheus")
        
        assert response.status_code == 200
        content = response.text
        
        # Should return minimal metrics even on error
        assert "ttv_api_uptime_seconds" in content
        assert 'ttv_api_health_status{component="api"} 1' in content
    
    def test_record_job_processed(self):
        """Test job processing metrics recording"""
        reset_metrics()
        
        # Record some jobs
        record_job_processed(5.0)
        record_job_processed(10.0)
        record_job_processed(15.0)
        
        # Check internal metrics storage
        from api.routes.health import _metrics_storage
        
        assert _metrics_storage["total_jobs_processed"] == 3
        assert _metrics_storage["total_processing_time"] == 30.0
    
    def test_record_request(self):
        """Test request metrics recording"""
        reset_metrics()
        
        # Record some requests
        record_request()
        record_request()
        record_request()
        
        # Check internal metrics storage
        from api.routes.health import _metrics_storage
        
        assert _metrics_storage["total_request_count"] == 3


class TestMetricsUtilities:
    """Test metrics utility functions"""
    
    def test_reset_metrics(self):
        """Test metrics reset functionality"""
        # Record some data
        record_job_processed(10.0)
        record_request()
        
        # Verify data exists
        from api.routes.health import _metrics_storage
        assert _metrics_storage["total_jobs_processed"] > 0
        assert _metrics_storage["total_request_count"] > 0
        
        # Reset and verify
        reset_metrics()
        assert _metrics_storage["total_jobs_processed"] == 0
        assert _metrics_storage["total_request_count"] == 0
        assert _metrics_storage["total_processing_time"] == 0.0
    
    def test_startup_time_tracking(self):
        """Test that startup time is tracked correctly"""
        from api.routes.health import _startup_time
        
        # Startup time should be set and reasonable
        assert _startup_time > 0
        assert _startup_time <= time.time()
        
        # Uptime should be positive
        uptime = time.time() - _startup_time
        assert uptime >= 0


if __name__ == "__main__":
    pytest.main([__file__])