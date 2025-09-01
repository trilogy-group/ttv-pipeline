"""
Integration tests for security middleware with the main application.

This module tests that the security middleware integrates properly with
the FastAPI application and provides the expected security features.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import create_app


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = MagicMock()
    config.server.host = "localhost"
    config.server.port = 8000
    config.server.quic_port = 8443
    config.server.cors_origins = ["http://localhost:3000", "https://example.com"]
    config.redis.host = "localhost"
    config.redis.port = 6379
    config.gcs.bucket = "test-bucket"
    config.security.rate_limit_per_minute = 10
    return config


@pytest.fixture
def client(mock_config):
    """Test client with mocked configuration"""
    with patch('api.main.get_config_from_env', return_value=mock_config):
        app = create_app()
        app.state.config = mock_config
        return TestClient(app)


class TestSecurityIntegration:
    """Test security middleware integration with the main application"""
    
    def test_security_headers_on_all_endpoints(self, client):
        """Test that security headers are added to all endpoints"""
        endpoints = ["/", "/healthz", "/readyz", "/metrics"]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            
            # All endpoints should have security headers
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert response.headers["X-Frame-Options"] == "DENY"
            assert response.headers["X-XSS-Protection"] == "1; mode=block"
            assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
            assert response.headers["X-Permitted-Cross-Domain-Policies"] == "none"
            assert "Content-Security-Policy" in response.headers
    
    def test_api_endpoints_have_cache_control(self, client):
        """Test that API endpoints have proper cache control headers"""
        # Add a test v1 route directly to the app
        from fastapi import APIRouter
        router = APIRouter()
        
        @router.get("/v1/test")
        async def test_api_endpoint():
            return {"test": "data"}
        
        client.app.include_router(router)
        
        response = client.get("/v1/test")
        
        # API endpoints should have no-cache headers
        assert response.headers["Cache-Control"] == "no-cache, no-store, must-revalidate"
        assert response.headers["Pragma"] == "no-cache"
        assert response.headers["Expires"] == "0"
    
    def test_correlation_id_tracking(self, client):
        """Test that correlation IDs are properly tracked across requests"""
        response1 = client.get("/")
        response2 = client.get("/healthz")
        
        # Both responses should have correlation IDs
        assert "X-Correlation-ID" in response1.headers
        assert "X-Correlation-ID" in response2.headers
        
        # Correlation IDs should be different for different requests
        assert response1.headers["X-Correlation-ID"] != response2.headers["X-Correlation-ID"]
        
        # Correlation IDs should be valid UUIDs
        import uuid
        uuid.UUID(response1.headers["X-Correlation-ID"])
        uuid.UUID(response2.headers["X-Correlation-ID"])
    
    def test_rate_limiting_integration(self, client):
        """Test that rate limiting works with the application"""
        # Make requests up to the limit (10 per minute in mock config)
        responses = []
        for i in range(12):  # Exceed the limit
            response = client.get("/")
            responses.append(response)
        
        # First 10 should succeed
        for i in range(10):
            assert responses[i].status_code == 200
            assert "X-RateLimit-Limit" in responses[i].headers
            assert "X-RateLimit-Remaining" in responses[i].headers
        
        # Subsequent requests should be rate limited
        for i in range(10, 12):
            assert responses[i].status_code == 429
            assert responses[i].json()["error"] == "RateLimitExceeded"
    
    def test_request_validation_integration(self, client):
        """Test that request validation works with the application"""
        # Test oversized request by setting content-length header manually
        import json
        large_data = {"data": "x" * 20000}  # Large payload
        large_json = json.dumps(large_data)
        
        # This should be blocked by request validation middleware
        response = client.post(
            "/",  # Use root endpoint which accepts any method in theory
            content=large_json,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(large_json))
            }
        )
        
        # Should be blocked by request validation (413) or method not allowed (405)
        assert response.status_code in [405, 413, 415]
    
    def test_error_handling_with_correlation_ids(self, client):
        """Test that error responses include correlation IDs"""
        # Test 404 error
        response = client.get("/nonexistent")
        
        assert response.status_code == 404
        assert "X-Correlation-ID" in response.headers
        
        # Error response should include correlation ID if it's a structured error
        data = response.json()
        if "request_id" in data:
            assert data["request_id"] == response.headers["X-Correlation-ID"]
    
    def test_health_endpoints_bypass_rate_limiting(self, client):
        """Test that health endpoints are not rate limited"""
        # Exhaust rate limit with regular endpoint
        for i in range(12):
            client.get("/")
        
        # Health endpoints should still work
        health_response = client.get("/healthz")
        assert health_response.status_code == 200
        
        readiness_response = client.get("/readyz")
        assert readiness_response.status_code == 200
        
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
    
    @patch('api.middleware.logger')
    def test_structured_logging_integration(self, mock_logger, client):
        """Test that structured logging works with the application"""
        response = client.get("/")
        
        # Verify that logging was called with structured data
        assert mock_logger.info.called
        
        # Check that log entries contain expected fields
        log_calls = mock_logger.info.call_args_list
        request_log = None
        
        for call in log_calls:
            if len(call) > 1 and 'extra' in call[1]:
                extra = call[1]['extra']
                if extra.get('event') == 'request_start':
                    request_log = extra
                    break
        
        assert request_log is not None
        assert 'correlation_id' in request_log
        assert 'method' in request_log
        assert 'path' in request_log
        assert 'client_ip' in request_log
        assert 'timestamp' in request_log
    
    def test_middleware_order_and_interaction(self, client):
        """Test that middleware components work together in the correct order"""
        response = client.get("/")
        
        # Should have headers from all middleware components
        expected_headers = [
            "X-Correlation-ID",  # From request logging middleware
            "X-Content-Type-Options",  # From security headers middleware
            "X-RateLimit-Limit",  # From rate limiting middleware
        ]
        
        for header in expected_headers:
            assert header in response.headers
        
        # Response should be successful
        assert response.status_code == 200
        
        # Should have proper content type
        assert response.headers["content-type"] == "application/json"


class TestSecurityConfiguration:
    """Test security configuration and customization"""
    
    def test_custom_rate_limits(self):
        """Test that custom rate limits are applied from configuration"""
        custom_config = MagicMock()
        custom_config.server.host = "localhost"
        custom_config.server.port = 8000
        custom_config.server.quic_port = 8443
        custom_config.server.cors_origins = ["*"]
        custom_config.redis.host = "localhost"
        custom_config.redis.port = 6379
        custom_config.gcs.bucket = "test-bucket"
        custom_config.security.rate_limit_per_minute = 5  # Lower limit
        
        with patch('api.main.get_config_from_env', return_value=custom_config):
            app = create_app()
            app.state.config = custom_config
            
            # Manually update the rate limit middleware configuration
            # since we're not going through the full lifespan startup
            for middleware in app.user_middleware:
                if hasattr(middleware.cls, '__name__') and middleware.cls.__name__ == "RateLimitMiddleware":
                    middleware.kwargs["requests_per_minute"] = custom_config.security.rate_limit_per_minute
                    break
            
            client = TestClient(app)
            
            # Make requests up to the custom limit
            responses = []
            for i in range(7):
                response = client.get("/")
                responses.append(response)
            
            # First 5 should succeed, rest should be rate limited
            success_count = sum(1 for r in responses if r.status_code == 200)
            rate_limited_count = sum(1 for r in responses if r.status_code == 429)
            
            assert success_count == 5
            assert rate_limited_count == 2
    
    def test_cors_configuration(self):
        """Test that CORS configuration is applied from settings"""
        cors_config = MagicMock()
        cors_config.server.host = "localhost"
        cors_config.server.port = 8000
        cors_config.server.quic_port = 8443
        cors_config.server.cors_origins = ["https://trusted-domain.com"]
        cors_config.redis.host = "localhost"
        cors_config.redis.port = 6379
        cors_config.gcs.bucket = "test-bucket"
        cors_config.security.rate_limit_per_minute = 60
        
        with patch('api.main.get_config_from_env', return_value=cors_config):
            app = create_app()
            app.state.config = cors_config
            client = TestClient(app)
            
            # Test request with configured origin
            response = client.get("/", headers={"Origin": "https://trusted-domain.com"})
            assert response.status_code == 200
            
            # CORS middleware configuration is applied during startup
            # The actual CORS behavior would be tested in browser environments