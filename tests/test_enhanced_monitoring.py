"""
Tests for enhanced observability and monitoring functionality.

This module tests the enhanced metrics collection, system health monitoring,
and comprehensive observability features added to the API server.
"""

import json
import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.config import APIConfig, APIServerConfig, RedisConfig, GCSConfig, SecurityConfig
from api.routes.health import (
    reset_metrics, record_job_processed, record_request_latency,
    get_queue_depth_metrics, get_system_health_metrics,
    _calculate_percentile, _calculate_error_rate
)


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


class TestEnhancedMetrics:
    """Test enhanced metrics collection and reporting"""
    
    def setup_method(self):
        """Reset metrics before each test"""
        reset_metrics()
    
    @patch('api.routes.health.get_job_queue')
    def test_enhanced_metrics_json_format(self, mock_get_job_queue, client):
        """Test enhanced JSON metrics endpoint with detailed data"""
        # Record some enhanced metrics
        record_job_processed(15.5)
        record_job_processed(22.3)
        record_request_latency(0.150, 200)
        record_request_latency(0.250, 200)
        record_request_latency(1.500, 500)
        
        # Mock job queue stats
        mock_job_queue = Mock()
        mock_job_queue.get_queue_stats.return_value = {
            "queued_jobs": 8,
            "started_jobs": 3,
            "finished_jobs": 15,
            "failed_jobs": 2
        }
        mock_get_job_queue.return_value = mock_job_queue
        
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check enhanced metrics
        assert data["active_jobs"] == 3
        assert data["queued_jobs"] == 8
        assert data["total_jobs_processed"] == 2
        assert data["total_requests"] == 3
        assert data["queue_depth"] == 11  # queued + active
        
        # Check calculated metrics
        expected_avg_job_time = (15.5 + 22.3) / 2
        assert abs(data["average_processing_time"] - expected_avg_job_time) < 0.01
        
        expected_avg_latency = (0.150 + 0.250 + 1.500) / 3
        assert abs(data["average_request_latency"] - expected_avg_latency) < 0.001
        
        # Check throughput metrics
        assert data["requests_per_second"] >= 0
        assert data["jobs_per_hour"] >= 0
        
        # Check status code distribution
        assert "200" in data["status_code_distribution"]
        assert "500" in data["status_code_distribution"]
        assert data["status_code_distribution"]["200"] == 2
        assert data["status_code_distribution"]["500"] == 1
    
    @patch('api.routes.health.get_job_queue')
    @patch('api.routes.health.get_redis_manager')
    @patch('api.routes.health.GCSClient')
    def test_enhanced_prometheus_metrics(self, mock_gcs_client, mock_get_redis_manager, mock_get_job_queue, client):
        """Test enhanced Prometheus metrics with histogram data"""
        # Record metrics with various latencies and durations
        record_request_latency(0.05, 200)   # < 0.1s bucket
        record_request_latency(0.3, 200)    # < 0.5s bucket
        record_request_latency(1.2, 200)    # < 2.5s bucket
        record_request_latency(8.0, 500)    # < 10s bucket
        record_request_latency(15.0, 500)   # +Inf bucket
        
        record_job_processed(25.0)   # < 30s bucket
        record_job_processed(120.0)  # < 300s bucket
        record_job_processed(1200.0) # < 1800s bucket
        record_job_processed(4000.0) # +Inf bucket
        
        # Mock dependencies
        mock_redis_manager = Mock()
        mock_redis_manager.test_connection.return_value = True
        mock_get_redis_manager.return_value = mock_redis_manager
        
        mock_gcs_instance = Mock()
        mock_gcs_instance.test_connection = AsyncMock(return_value=True)
        mock_gcs_client.return_value = mock_gcs_instance
        
        mock_job_queue = Mock()
        mock_job_queue.get_queue_stats.return_value = {
            "queued_jobs": 5,
            "started_jobs": 2,
            "finished_jobs": 10
        }
        mock_get_job_queue.return_value = mock_job_queue
        
        response = client.get("/metrics/prometheus")
        
        assert response.status_code == 200
        content = response.text
        
        # Check enhanced metrics
        assert "ttv_api_request_latency_seconds" in content
        assert "ttv_api_requests_per_second" in content
        assert "ttv_api_jobs_per_hour" in content
        
        # Check histogram buckets (cumulative)
        assert 'ttv_api_request_duration_seconds_bucket{le="0.1"} 1' in content  # 0.05s
        assert 'ttv_api_request_duration_seconds_bucket{le="0.5"} 2' in content  # 0.05s + 0.3s
        assert 'ttv_api_request_duration_seconds_bucket{le="2.5"} 3' in content  # + 1.2s
        assert 'ttv_api_request_duration_seconds_bucket{le="10.0"} 4' in content # + 8.0s
        assert 'ttv_api_request_duration_seconds_bucket{le="+Inf"} 5' in content # + 15.0s
        
        assert 'ttv_api_job_duration_seconds_bucket{le="30"} 1' in content    # 25.0s
        assert 'ttv_api_job_duration_seconds_bucket{le="300"} 2' in content   # + 120.0s
        assert 'ttv_api_job_duration_seconds_bucket{le="1800"} 3' in content  # + 1200.0s
        assert 'ttv_api_job_duration_seconds_bucket{le="+Inf"} 4' in content  # + 4000.0s
        
        # Check queue depth metric
        assert "ttv_api_queue_depth_total 7" in content  # 5 queued + 2 started
        
        # Check HTTP status code metrics
        assert 'ttv_api_http_requests_total{code="200"} 3' in content
        assert 'ttv_api_http_requests_total{code="500"} 2' in content
    
    @patch('api.routes.health.get_job_queue')
    @patch('api.routes.health.get_redis_manager')
    @patch('api.routes.health.GCSClient')
    def test_system_metrics_endpoint(self, mock_gcs_client, mock_get_redis_manager, mock_get_job_queue, client):
        """Test comprehensive system metrics endpoint"""
        # Record some metrics
        record_job_processed(30.0)
        record_request_latency(0.2, 200)
        record_request_latency(0.8, 200)
        record_request_latency(2.0, 404)
        
        # Mock dependencies
        mock_redis_manager = Mock()
        mock_redis_manager.test_connection.return_value = True
        mock_get_redis_manager.return_value = mock_redis_manager
        
        mock_gcs_instance = Mock()
        mock_gcs_instance.test_connection = AsyncMock(return_value=True)
        mock_gcs_client.return_value = mock_gcs_instance
        
        mock_job_queue = Mock()
        mock_job_queue.get_queue_stats.return_value = {
            "queued_jobs": 3,
            "started_jobs": 1,
            "finished_jobs": 8,
            "failed_jobs": 1,
            "deferred_jobs": 0,
            "scheduled_jobs": 0
        }
        mock_get_job_queue.return_value = mock_job_queue
        
        response = client.get("/metrics/system")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check structure
        assert "timestamp" in data
        assert "version" in data
        assert "system_health" in data
        assert "queue_metrics" in data
        assert "histogram_data" in data
        assert "throughput_metrics" in data
        assert "latency_metrics" in data
        assert "error_metrics" in data
        
        # Check system health
        system_health = data["system_health"]
        assert "components" in system_health
        assert "performance" in system_health
        assert "capacity" in system_health
        
        # Check queue metrics
        queue_metrics = data["queue_metrics"]
        assert queue_metrics["total_depth"] == 4  # 3 queued + 1 started
        assert queue_metrics["queued"] == 3
        assert queue_metrics["active"] == 1
        assert queue_metrics["failed"] == 1
        
        # Check histogram data
        histogram_data = data["histogram_data"]
        assert "request_latency_distribution" in histogram_data
        assert "job_duration_distribution" in histogram_data
        
        # Check latency metrics
        latency_metrics = data["latency_metrics"]
        assert "average_request_latency" in latency_metrics
        assert "average_job_duration" in latency_metrics
        assert "p95_request_latency" in latency_metrics
        assert "p99_request_latency" in latency_metrics
        
        # Check error metrics
        error_metrics = data["error_metrics"]
        assert "status_code_distribution" in error_metrics
        assert "error_rate" in error_metrics
        assert error_metrics["error_rate"] == 33.33  # 1 out of 3 requests was 404


class TestMetricsUtilities:
    """Test metrics utility functions"""
    
    def setup_method(self):
        """Reset metrics before each test"""
        reset_metrics()
    
    def test_percentile_calculation(self):
        """Test histogram percentile calculation"""
        # Create test histogram data (cumulative counts)
        buckets = {
            "0.1": 10,   # 10 requests <= 100ms
            "0.5": 30,   # 30 requests <= 500ms (cumulative)
            "1.0": 45,   # 45 requests <= 1s (cumulative)
            "2.5": 55,   # 55 requests <= 2.5s (cumulative)
            "5.0": 60,   # 60 requests <= 5s (cumulative)
            "10.0": 63,  # 63 requests <= 10s (cumulative)
            "+Inf": 65   # 65 requests total (cumulative)
        }
        
        # Test various percentiles
        p50 = _calculate_percentile(buckets, 0.5)  # 50th percentile
        p95 = _calculate_percentile(buckets, 0.95)  # 95th percentile
        p99 = _calculate_percentile(buckets, 0.99)  # 99th percentile
        
        # 50th percentile: 32.5 out of 65 -> should be in 1.0 bucket (30 < 32.5 <= 45)
        assert p50 == 1.0
        
        # 95th percentile: 61.75 out of 65 -> should be in 10.0 bucket (60 < 61.75 <= 63)
        assert p95 == 10.0
        
        # 99th percentile: 64.35 out of 65 -> should be in +Inf bucket (63 < 64.35 <= 65)
        assert p99 == 10.0  # Capped at 10.0 for display
    
    def test_error_rate_calculation(self):
        """Test error rate calculation from status codes"""
        # Record various status codes
        record_request_latency(0.1, 200)  # Success
        record_request_latency(0.2, 200)  # Success
        record_request_latency(0.3, 201)  # Success
        record_request_latency(0.4, 404)  # Client error
        record_request_latency(0.5, 500)  # Server error
        
        error_rate = _calculate_error_rate()
        
        # 2 errors out of 5 requests = 40%
        assert error_rate == 40.0
    
    @patch('api.routes.health.get_job_queue')
    def test_queue_depth_metrics(self, mock_get_job_queue):
        """Test queue depth metrics collection"""
        mock_job_queue = Mock()
        mock_job_queue.get_queue_stats.return_value = {
            "queued_jobs": 12,
            "started_jobs": 5,
            "finished_jobs": 100,
            "failed_jobs": 8,
            "deferred_jobs": 2,
            "scheduled_jobs": 3
        }
        mock_get_job_queue.return_value = mock_job_queue
        
        metrics = get_queue_depth_metrics()
        
        assert metrics["total_depth"] == 17  # 12 queued + 5 started
        assert metrics["queued"] == 12
        assert metrics["active"] == 5
        assert metrics["finished"] == 100
        assert metrics["failed"] == 8
        assert metrics["deferred"] == 2
        assert metrics["scheduled"] == 3
    
    @patch('api.routes.health.get_job_queue')
    def test_queue_depth_metrics_error_handling(self, mock_get_job_queue):
        """Test queue depth metrics error handling"""
        mock_get_job_queue.side_effect = Exception("Queue unavailable")
        
        metrics = get_queue_depth_metrics()
        
        # Should return zero values on error
        assert metrics["total_depth"] == 0
        assert metrics["queued"] == 0
        assert metrics["active"] == 0
    
    @patch('api.routes.health.get_redis_manager')
    def test_system_health_metrics(self, mock_get_redis_manager):
        """Test system health metrics collection"""
        # Record some metrics first
        record_job_processed(45.0)
        record_request_latency(0.3, 200)
        
        mock_redis_manager = Mock()
        mock_redis_manager.test_connection.return_value = True
        mock_get_redis_manager.return_value = mock_redis_manager
        
        health_metrics = get_system_health_metrics()
        
        assert "components" in health_metrics
        assert "performance" in health_metrics
        assert "capacity" in health_metrics
        
        # Check component health
        components = health_metrics["components"]
        assert components["redis"]["healthy"] is True
        assert "last_check" in components["redis"]
        
        # Check performance metrics
        performance = health_metrics["performance"]
        assert performance["uptime_seconds"] >= 0
        assert performance["average_job_duration"] == 45.0
        assert performance["average_request_latency"] == 0.3


class TestMetricsIntegration:
    """Test metrics integration with middleware and workers"""
    
    def setup_method(self):
        """Reset metrics before each test"""
        reset_metrics()
    
    def test_request_latency_recording(self):
        """Test request latency recording functionality"""
        # Record various request latencies
        record_request_latency(0.05, 200)
        record_request_latency(0.15, 200)
        record_request_latency(0.75, 404)
        record_request_latency(2.5, 500)
        
        # Check internal metrics storage
        from api.routes.health import _metrics_storage
        
        assert _metrics_storage["total_request_count"] == 4
        assert _metrics_storage["total_request_duration"] == 3.45
        
        # Check histogram buckets
        latency_buckets = _metrics_storage["request_latency_buckets"]
        assert latency_buckets["0.1"] == 1   # 0.05s
        assert latency_buckets["0.5"] == 2   # 0.15s (cumulative)
        assert latency_buckets["1.0"] == 3   # 0.75s (cumulative)
        assert latency_buckets["5.0"] == 4   # 2.5s (cumulative)
        
        # Check status code distribution
        status_counts = _metrics_storage["status_code_counts"]
        assert status_counts["200"] == 2
        assert status_counts["404"] == 1
        assert status_counts["500"] == 1
    
    def test_job_processing_metrics_recording(self):
        """Test job processing metrics recording"""
        # Record various job processing times
        record_job_processed(25.0)   # < 30s
        record_job_processed(120.0)  # < 300s
        record_job_processed(450.0)  # < 600s
        record_job_processed(2000.0) # > 1800s (+Inf)
        
        # Check internal metrics storage
        from api.routes.health import _metrics_storage
        
        assert _metrics_storage["total_jobs_processed"] == 4
        assert _metrics_storage["total_processing_time"] == 2595.0
        
        # Check histogram buckets
        job_buckets = _metrics_storage["job_duration_buckets"]
        assert job_buckets["30"] == 1     # 25.0s
        assert job_buckets["300"] == 2    # 120.0s (cumulative)
        assert job_buckets["600"] == 3    # 450.0s (cumulative)
        assert job_buckets["+Inf"] == 4   # 2000.0s (cumulative)


if __name__ == "__main__":
    pytest.main([__file__])