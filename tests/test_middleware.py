"""
Tests for security middleware components.

This module tests the security headers, request logging, rate limiting,
and request validation middleware.
"""

import json
import time
from unittest.mock import Mock, patch
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import Response

from api.middleware import (
    SecurityHeadersMiddleware,
    RequestLoggingMiddleware,
    RateLimitMiddleware,
    RequestValidationMiddleware
)


class TestSecurityHeadersMiddleware:
    """Test security headers middleware"""
    
    def test_adds_security_headers(self):
        """Test that security headers are added to responses"""
        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        client = TestClient(app)
        response = client.get("/test")
        
        # Check security headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert response.headers["X-Permitted-Cross-Domain-Policies"] == "none"
        assert "Content-Security-Policy" in response.headers
    
    def test_cache_control_for_api_endpoints(self):
        """Test that cache control headers are added for API endpoints"""
        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)
        
        @app.get("/v1/test")
        async def api_endpoint():
            return {"message": "api"}
        
        @app.get("/docs")
        async def docs_endpoint():
            return {"message": "docs"}
        
        client = TestClient(app)
        
        # API endpoint should have no-cache headers
        api_response = client.get("/v1/test")
        assert api_response.headers["Cache-Control"] == "no-cache, no-store, must-revalidate"
        assert api_response.headers["Pragma"] == "no-cache"
        assert api_response.headers["Expires"] == "0"
        
        # Non-API endpoint should not have cache control
        docs_response = client.get("/docs")
        assert "Cache-Control" not in docs_response.headers or \
               docs_response.headers["Cache-Control"] != "no-cache, no-store, must-revalidate"


class TestRequestLoggingMiddleware:
    """Test request logging middleware"""
    
    @patch('api.middleware.logger')
    def test_logs_request_and_response(self, mock_logger):
        """Test that requests and responses are logged with correlation IDs"""
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        client = TestClient(app)
        response = client.get("/test")
        
        # Check that correlation ID is in response headers
        assert "X-Correlation-ID" in response.headers
        correlation_id = response.headers["X-Correlation-ID"]
        
        # Check that logging was called
        assert mock_logger.info.call_count >= 2  # Request start and complete
        
        # Verify log structure
        log_calls = mock_logger.info.call_args_list
        request_log = log_calls[0][1]['extra']  # First call should be request start
        
        assert request_log['event'] == 'request_start'
        assert request_log['correlation_id'] == correlation_id
        assert request_log['method'] == 'GET'
        assert request_log['path'] == '/test'
    
    def test_redacts_sensitive_headers(self):
        """Test that sensitive headers are redacted in logs"""
        app = FastAPI()
        
        # Mock logger to capture log data
        logged_data = []
        
        def mock_log_info(message, extra=None):
            if extra:
                logged_data.append(extra)
        
        with patch('api.middleware.logger.info', side_effect=mock_log_info):
            app.add_middleware(RequestLoggingMiddleware)
            
            @app.get("/test")
            async def test_endpoint():
                return {"message": "test"}
            
            client = TestClient(app)
            response = client.get("/test", headers={"Authorization": "Bearer secret-token"})
            
            # Find the request log entry
            request_log = None
            for log_entry in logged_data:
                if log_entry.get('event') == 'request_start':
                    request_log = log_entry
                    break
            
            assert request_log is not None
            assert request_log['headers']['authorization'] == '[REDACTED]'
    
    @patch('api.middleware.logger')
    def test_logs_request_errors(self, mock_logger):
        """Test that request errors are logged properly"""
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)
        
        @app.get("/error")
        async def error_endpoint():
            raise ValueError("Test error")
        
        client = TestClient(app)
        
        # This should raise an exception, but we catch it to test logging
        try:
            client.get("/error")
        except:
            pass
        
        # Check that error was logged
        mock_logger.error.assert_called()
        error_call = mock_logger.error.call_args
        assert 'request_error' in str(error_call)


class TestRateLimitMiddleware:
    """Test rate limiting middleware"""
    
    def test_allows_requests_under_limit(self):
        """Test that requests under the limit are allowed"""
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, requests_per_minute=10, burst_limit=5)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        client = TestClient(app)
        
        # Make several requests under the limit
        for i in range(5):
            response = client.get("/test")
            assert response.status_code == 200
            assert "X-RateLimit-Limit" in response.headers
            assert "X-RateLimit-Remaining" in response.headers
    
    def test_blocks_requests_over_limit(self):
        """Test that requests over the limit are blocked"""
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, requests_per_minute=2, burst_limit=2)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        client = TestClient(app)
        
        # Make requests up to the limit
        response1 = client.get("/test")
        assert response1.status_code == 200
        
        response2 = client.get("/test")
        assert response2.status_code == 200
        
        # Next request should be rate limited
        response3 = client.get("/test")
        assert response3.status_code == 429
        assert response3.json()["error"] == "RateLimitExceeded"
        assert "Retry-After" in response3.headers
    
    def test_skips_health_endpoints(self):
        """Test that health check endpoints are not rate limited"""
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, requests_per_minute=1, burst_limit=1)
        
        @app.get("/healthz")
        async def health_endpoint():
            return {"status": "healthy"}
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        client = TestClient(app)
        
        # Exhaust rate limit with regular endpoint
        response1 = client.get("/test")
        assert response1.status_code == 200
        
        response2 = client.get("/test")
        assert response2.status_code == 429
        
        # Health endpoint should still work
        health_response = client.get("/healthz")
        assert health_response.status_code == 200


class TestRequestValidationMiddleware:
    """Test request validation middleware"""
    
    def test_blocks_oversized_requests(self):
        """Test that oversized requests are blocked"""
        app = FastAPI()
        app.add_middleware(RequestValidationMiddleware, max_content_length=100)
        
        @app.post("/test")
        async def test_endpoint(data: dict):
            return {"message": "test"}
        
        client = TestClient(app)
        
        # Create a large payload
        large_data = {"data": "x" * 200}
        
        response = client.post(
            "/test",
            json=large_data,
            headers={"Content-Length": str(len(json.dumps(large_data)))}
        )
        
        assert response.status_code == 413
        assert response.json()["error"] == "PayloadTooLarge"
    
    def test_validates_content_type(self):
        """Test that content type is validated for POST requests"""
        app = FastAPI()
        app.add_middleware(RequestValidationMiddleware)
        
        @app.post("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        client = TestClient(app)
        
        # POST with wrong content type
        response = client.post(
            "/test",
            content="plain text",
            headers={"Content-Type": "text/plain"}
        )
        
        assert response.status_code == 415
        assert response.json()["error"] == "UnsupportedMediaType"
    
    def test_allows_valid_requests(self):
        """Test that valid requests are allowed through"""
        app = FastAPI()
        app.add_middleware(RequestValidationMiddleware)
        
        @app.post("/test")
        async def test_endpoint(data: dict):
            return {"message": "received", "data": data}
        
        @app.get("/test")
        async def get_endpoint():
            return {"message": "get"}
        
        client = TestClient(app)
        
        # Valid POST request
        post_response = client.post("/test", json={"key": "value"})
        assert post_response.status_code == 200
        
        # Valid GET request
        get_response = client.get("/test")
        assert get_response.status_code == 200


class TestMiddlewareIntegration:
    """Test middleware integration and order"""
    
    def test_middleware_order_and_headers(self):
        """Test that all middleware work together and headers are properly set"""
        app = FastAPI()
        
        # Add middleware in reverse order (FastAPI applies them in reverse)
        app.add_middleware(SecurityHeadersMiddleware)
        app.add_middleware(RequestLoggingMiddleware)
        app.add_middleware(RateLimitMiddleware, requests_per_minute=10)
        app.add_middleware(RequestValidationMiddleware)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        client = TestClient(app)
        response = client.get("/test")
        
        # Check that all expected headers are present
        assert response.status_code == 200
        assert "X-Correlation-ID" in response.headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-RateLimit-Limit" in response.headers
        
        # Check response content
        assert response.json()["message"] == "test"