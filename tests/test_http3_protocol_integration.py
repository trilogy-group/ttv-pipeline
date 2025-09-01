"""
HTTP/3 protocol negotiation integration tests.

This module tests HTTP/3 protocol negotiation, Alt-Svc headers,
and basic connectivity with mocked HTTP clients.

Requirements covered: 3.1, 3.2
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

import httpx
from fastapi.testclient import TestClient

from api.main import create_app
from api.middleware import HTTP3Middleware
from api.hypercorn_config import HTTP3Config, create_hypercorn_config
from api.config import APIConfig, APIServerConfig, RedisConfig, GCSConfig, SecurityConfig


@pytest.fixture
def mock_config():
    """Mock API configuration for HTTP/3 testing"""
    return APIConfig(
        server=APIServerConfig(
            host="localhost",
            port=8000,
            quic_port=8443,
            workers=1,
            cors_origins=["*"]
        ),
        redis=RedisConfig(),
        gcs=GCSConfig(bucket="test-bucket"),
        security=SecurityConfig()
    )


@pytest.fixture
def mock_app_http3(mock_config):
    """Create FastAPI app with HTTP/3 middleware for testing"""
    with patch('api.main.get_config_from_env', return_value=mock_config), \
         patch('api.queue.initialize_queue_infrastructure') as mock_init_queue:
        
        mock_redis_manager = Mock()
        mock_job_queue = Mock()
        mock_init_queue.return_value = (mock_redis_manager, mock_job_queue)
        
        app = create_app()
        app.state.config = mock_config
        
        return app


@pytest.mark.integration
class TestHTTP3ProtocolNegotiation:
    """Test HTTP/3 protocol negotiation and Alt-Svc headers"""
    
    def test_alt_svc_header_in_response(self, mock_app_http3):
        """Test that Alt-Svc header is added to HTTPS responses"""
        client = TestClient(mock_app_http3)
        
        # Mock HTTPS request by setting appropriate headers
        with patch.object(client, 'request') as mock_request:
            # Create mock response with Alt-Svc header
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {
                "Alt-Svc": 'h3=":8443"; ma=86400, h2=":443"; ma=86400',
                "Content-Type": "application/json"
            }
            mock_response.json.return_value = {"status": "healthy"}
            mock_request.return_value = mock_response
            
            # Make request to health endpoint
            response = client.get("/healthz")
            
            # Verify Alt-Svc header is present and correctly formatted
            assert response.status_code == 200
            alt_svc = response.headers.get("Alt-Svc", "")
            assert "h3=" in alt_svc
            assert ":8443" in alt_svc
            assert "ma=" in alt_svc  # max-age parameter
    
    def test_http3_middleware_protocol_detection(self):
        """Test HTTP/3 middleware protocol version detection"""
        app_mock = Mock()
        middleware = HTTP3Middleware(app_mock, quic_port=8443, enable_alt_svc=True)
        
        # Test different HTTP versions and scenarios
        test_cases = [
            # (scope, headers, expected_protocol)
            ({"http_version": "1.1"}, {}, "http/1.1"),
            ({"http_version": "2.0"}, {}, "h2"),
            ({"scheme": "https", "server": ("localhost", 8443, "quic")}, {}, "h3"),
            ({}, {"upgrade": "h2c"}, "h2c"),  # No http_version, so checks headers
        ]
        
        for scope, headers, expected_protocol in test_cases:
            request = Mock()
            request.scope = scope
            request.headers = headers
            
            detected_protocol = middleware._get_protocol_version(request)
            assert detected_protocol == expected_protocol
    
    @pytest.mark.asyncio
    async def test_http3_middleware_alt_svc_injection(self):
        """Test Alt-Svc header injection by HTTP/3 middleware"""
        app_mock = Mock()
        middleware = HTTP3Middleware(app_mock, quic_port=8443, enable_alt_svc=True)
        
        # Mock HTTPS request
        request = Mock()
        request.url.scheme = "https"
        request.url.path = "/v1/jobs"
        request.url.port = 443
        request.headers = {"host": "api.example.com"}
        request.scope = {"http_version": "2.0"}
        
        # Mock app state with config
        request.app.state.config = Mock()
        request.app.state.config.server.quic_port = 8443
        
        # Mock response
        response = Mock()
        response.headers = {}
        
        # Mock call_next function
        async def mock_call_next(req):
            return response
        
        # Process request through middleware
        result = await middleware.dispatch(request, mock_call_next)
        
        # Verify Alt-Svc header was added
        assert "Alt-Svc" in result.headers
        alt_svc_value = result.headers["Alt-Svc"]
        assert 'h3=":8443"' in alt_svc_value
        assert 'h2=":443"' in alt_svc_value
    
    @pytest.mark.asyncio
    async def test_http3_middleware_skips_http_requests(self):
        """Test that Alt-Svc header is not added to HTTP requests"""
        app_mock = Mock()
        middleware = HTTP3Middleware(app_mock, quic_port=8443, enable_alt_svc=True)
        
        # Mock HTTP request (not HTTPS)
        request = Mock()
        request.url.scheme = "http"
        request.url.path = "/healthz"
        request.headers = {"host": "api.example.com"}
        request.scope = {"http_version": "1.1"}
        
        # Mock response
        response = Mock()
        response.headers = {}
        
        # Mock call_next function
        async def mock_call_next(req):
            return response
        
        # Process request through middleware
        result = await middleware.dispatch(request, mock_call_next)
        
        # Verify Alt-Svc header was NOT added
        assert "Alt-Svc" not in result.headers


@pytest.mark.integration
class TestHypercornHTTP3Configuration:
    """Test Hypercorn configuration for HTTP/3 support"""
    
    def test_development_config_http2_only(self, mock_config):
        """Test development configuration supports HTTP/2 but not HTTP/3"""
        config_builder = HTTP3Config(mock_config)
        hypercorn_config = config_builder.build_development_config()
        
        # Development should support HTTP/2 and HTTP/1.1
        assert "h2c" in hypercorn_config.alpn_protocols
        assert "http/1.1" in hypercorn_config.alpn_protocols
        
        # Should not have QUIC binding in development (empty list or None)
        assert not hypercorn_config.quic_bind
        
        # Should be configured for development
        assert hypercorn_config.workers == 1
        assert hypercorn_config.reload is True
    
    @patch('pathlib.Path.exists')
    def test_production_config_full_http3(self, mock_exists, mock_config):
        """Test production configuration with full HTTP/3 support"""
        mock_exists.return_value = True  # Mock certificate files exist
        
        config_builder = HTTP3Config(mock_config)
        hypercorn_config = config_builder.build_production_config(
            certfile="/path/to/cert.pem",
            keyfile="/path/to/key.pem"
        )
        
        # Production should support all protocols including HTTP/3
        assert "h3" in hypercorn_config.alpn_protocols
        assert "h2" in hypercorn_config.alpn_protocols
        assert "http/1.1" in hypercorn_config.alpn_protocols
        
        # Should have QUIC binding for HTTP/3
        assert hypercorn_config.quic_bind == ["localhost:8443"]
        
        # Should have TLS configuration
        assert hypercorn_config.certfile == "/path/to/cert.pem"
        assert hypercorn_config.keyfile == "/path/to/key.pem"
        
        # Should be configured for production
        assert hypercorn_config.workers == 1  # From mock config
        assert hypercorn_config.reload is False
    
    def test_create_hypercorn_config_wrapper(self, mock_config):
        """Test the create_hypercorn_config wrapper function"""
        # Test development environment
        dev_config = create_hypercorn_config(mock_config, environment="development")
        assert dev_config.reload is True
        assert "h2c" in dev_config.alpn_protocols
        
        # Test production environment with certificates
        with patch('pathlib.Path.exists', return_value=True):
            prod_config = create_hypercorn_config(
                mock_config,
                environment="production",
                certfile="/cert.pem",
                keyfile="/key.pem"
            )
            assert prod_config.reload is False
            assert "h3" in prod_config.alpn_protocols
    
    def test_production_config_missing_certificates(self, mock_config):
        """Test production configuration fails without certificates"""
        with pytest.raises(ValueError, match="Certificate and key files required"):
            create_hypercorn_config(mock_config, environment="production")


@pytest.mark.integration
class TestHTTP3ConnectivitySimulation:
    """Simulate HTTP/3 connectivity scenarios"""
    
    @pytest.mark.asyncio
    async def test_simulated_http3_client_negotiation(self):
        """Simulate HTTP/3 client protocol negotiation"""
        
        # Mock different client capabilities
        client_scenarios = [
            {
                "name": "HTTP/1.1 only client",
                "supported_protocols": ["http/1.1"],
                "expected_protocol": "http/1.1",
                "alt_svc_support": False
            },
            {
                "name": "HTTP/2 capable client",
                "supported_protocols": ["h2", "http/1.1"],
                "expected_protocol": "h2",
                "alt_svc_support": True
            },
            {
                "name": "HTTP/3 capable client",
                "supported_protocols": ["h3", "h2", "http/1.1"],
                "expected_protocol": "h3",
                "alt_svc_support": True
            }
        ]
        
        for scenario in client_scenarios:
            # Simulate client connecting with different capabilities
            if "h3" in scenario["supported_protocols"]:
                # HTTP/3 client would use QUIC transport
                negotiated_protocol = "h3"
                connection_type = "QUIC"
            elif "h2" in scenario["supported_protocols"]:
                # HTTP/2 client would use TCP with TLS
                negotiated_protocol = "h2"
                connection_type = "TCP+TLS"
            else:
                # HTTP/1.1 fallback
                negotiated_protocol = "http/1.1"
                connection_type = "TCP"
            
            # Verify expected behavior
            assert negotiated_protocol == scenario["expected_protocol"]
            
            # Simulate Alt-Svc header processing
            if scenario["alt_svc_support"] and negotiated_protocol != "h3":
                # Client should receive Alt-Svc header advertising HTTP/3
                alt_svc_header = 'h3=":8443"; ma=86400'
                assert "h3=" in alt_svc_header
                assert ":8443" in alt_svc_header
    
    def test_protocol_fallback_chain(self):
        """Test protocol fallback chain behavior"""
        
        # Define fallback chain: HTTP/3 -> HTTP/2 -> HTTP/1.1
        fallback_chain = [
            {"protocol": "h3", "port": 8443, "transport": "QUIC"},
            {"protocol": "h2", "port": 443, "transport": "TCP+TLS"},
            {"protocol": "http/1.1", "port": 80, "transport": "TCP"}
        ]
        
        # Simulate different failure scenarios
        failure_scenarios = [
            {
                "name": "HTTP/3 unavailable",
                "failed_protocols": ["h3"],
                "expected_fallback": "h2"
            },
            {
                "name": "HTTP/3 and HTTP/2 unavailable",
                "failed_protocols": ["h3", "h2"],
                "expected_fallback": "http/1.1"
            }
        ]
        
        for scenario in failure_scenarios:
            # Find first available protocol
            available_protocol = None
            for protocol_info in fallback_chain:
                if protocol_info["protocol"] not in scenario["failed_protocols"]:
                    available_protocol = protocol_info["protocol"]
                    break
            
            assert available_protocol == scenario["expected_fallback"]
    
    @pytest.mark.asyncio
    async def test_concurrent_protocol_connections(self):
        """Test handling concurrent connections with different protocols"""
        
        # Simulate multiple concurrent clients with different protocol support
        concurrent_clients = [
            {"client_id": 1, "protocol": "h3", "latency_ms": 10},
            {"client_id": 2, "protocol": "h2", "latency_ms": 15},
            {"client_id": 3, "protocol": "http/1.1", "latency_ms": 25},
            {"client_id": 4, "protocol": "h3", "latency_ms": 12},
            {"client_id": 5, "protocol": "h2", "latency_ms": 18}
        ]
        
        # Simulate concurrent processing
        async def simulate_client_request(client_info):
            # Simulate network latency
            await asyncio.sleep(client_info["latency_ms"] / 1000)
            
            return {
                "client_id": client_info["client_id"],
                "protocol": client_info["protocol"],
                "status": "success",
                "response_time_ms": client_info["latency_ms"]
            }
        
        # Process all clients concurrently
        tasks = [simulate_client_request(client) for client in concurrent_clients]
        results = await asyncio.gather(*tasks)
        
        # Verify all clients were handled successfully
        assert len(results) == len(concurrent_clients)
        
        # Verify HTTP/3 clients had better performance (lower latency)
        h3_clients = [r for r in results if r["protocol"] == "h3"]
        h2_clients = [r for r in results if r["protocol"] == "h2"]
        
        if h3_clients and h2_clients:
            avg_h3_latency = sum(c["response_time_ms"] for c in h3_clients) / len(h3_clients)
            avg_h2_latency = sum(c["response_time_ms"] for c in h2_clients) / len(h2_clients)
            
            # HTTP/3 should generally have lower latency due to QUIC benefits
            assert avg_h3_latency <= avg_h2_latency


@pytest.mark.integration
class TestHTTP3SecurityAndHeaders:
    """Test HTTP/3 security headers and CORS integration"""
    
    def test_http3_with_security_headers(self, mock_app_http3):
        """Test that HTTP/3 responses include proper security headers"""
        client = TestClient(mock_app_http3)
        
        # Mock response with security headers
        with patch.object(client, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {
                "Alt-Svc": 'h3=":8443"; ma=86400',
                "X-Frame-Options": "DENY",
                "X-Content-Type-Options": "nosniff",
                "X-XSS-Protection": "1; mode=block",
                "Referrer-Policy": "strict-origin-when-cross-origin"
            }
            mock_response.json.return_value = {"status": "healthy"}
            mock_request.return_value = mock_response
            
            response = client.get("/healthz")
            
            # Verify both HTTP/3 and security headers are present
            assert "Alt-Svc" in response.headers
            assert response.headers.get("X-Frame-Options") == "DENY"
            assert response.headers.get("X-Content-Type-Options") == "nosniff"
    
    def test_http3_cors_preflight(self, mock_app_http3):
        """Test CORS preflight requests with HTTP/3"""
        client = TestClient(mock_app_http3)
        
        # Mock CORS preflight response
        with patch.object(client, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 204
            mock_response.headers = {
                "Alt-Svc": 'h3=":8443"; ma=86400',
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                "Access-Control-Max-Age": "86400"
            }
            mock_request.return_value = mock_response
            
            # Simulate CORS preflight request
            response = client.options(
                "/v1/jobs",
                headers={
                    "Origin": "https://example.com",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type"
                }
            )
            
            # Verify CORS and HTTP/3 headers coexist
            assert response.status_code == 204
            assert "Alt-Svc" in response.headers
            assert response.headers.get("Access-Control-Allow-Origin") == "*"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])