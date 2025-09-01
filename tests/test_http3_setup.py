"""
Tests for HTTP/3 setup and configuration.

This module tests the HTTP/3 configuration, middleware, and server setup
to ensure proper protocol negotiation and Alt-Svc header handling.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from api.main import create_app
from api.config import APIConfig, APIServerConfig, RedisConfig, GCSConfig, SecurityConfig
from api.hypercorn_config import HTTP3Config, create_hypercorn_config
from api.middleware import HTTP3Middleware


class TestHTTP3Middleware:
    """Test HTTP/3 middleware functionality"""
    
    def test_middleware_initialization(self):
        """Test HTTP3Middleware initialization"""
        app = Mock()
        middleware = HTTP3Middleware(app, quic_port=8443, enable_alt_svc=True)
        
        assert middleware.quic_port == 8443
        assert middleware.enable_alt_svc is True
    
    @pytest.mark.asyncio
    async def test_alt_svc_header_added(self):
        """Test that Alt-Svc header is added for HTTPS API requests"""
        app = Mock()
        middleware = HTTP3Middleware(app, quic_port=8443, enable_alt_svc=True)
        
        # Mock request
        request = Mock()
        request.url.scheme = "https"
        request.url.path = "/v1/jobs"
        request.url.port = 443
        request.headers = {"host": "api.example.com"}
        request.app.state.config = Mock()
        request.app.state.config.server.quic_port = 8443
        request.scope = {"http_version": "2.0"}  # Mock scope properly
        
        # Mock response
        response = Mock()
        response.headers = {}
        
        # Mock call_next
        async def call_next(req):
            return response
        
        # Process request
        result = await middleware.dispatch(request, call_next)
        
        # Verify Alt-Svc header was added
        assert "Alt-Svc" in result.headers
        assert 'h3=":8443"' in result.headers["Alt-Svc"]
        assert 'h2=":443"' in result.headers["Alt-Svc"]
    
    @pytest.mark.asyncio
    async def test_alt_svc_not_added_for_http(self):
        """Test that Alt-Svc header is not added for HTTP requests"""
        app = Mock()
        middleware = HTTP3Middleware(app, quic_port=8443, enable_alt_svc=True)
        
        # Mock HTTP request
        request = Mock()
        request.url.scheme = "http"
        request.url.path = "/v1/jobs"
        request.headers = {"host": "api.example.com"}
        request.app.state.config = Mock()
        request.app.state.config.server.quic_port = 8443
        request.scope = {"http_version": "1.1"}  # Mock scope properly
        
        # Mock response
        response = Mock()
        response.headers = {}
        
        # Mock call_next
        async def call_next(req):
            return response
        
        # Process request
        result = await middleware.dispatch(request, call_next)
        
        # Verify Alt-Svc header was not added
        assert "Alt-Svc" not in result.headers
    
    def test_protocol_version_detection(self):
        """Test protocol version detection from request scope"""
        app = Mock()
        middleware = HTTP3Middleware(app)
        
        # Test HTTP/2 detection
        request = Mock()
        request.scope = {"http_version": "2.0"}
        
        version = middleware._get_protocol_version(request)
        assert version == "h2"
        
        # Test HTTP/1.1 detection
        request.scope = {"http_version": "1.1"}
        version = middleware._get_protocol_version(request)
        assert version == "http/1.1"


class TestHypercornConfig:
    """Test Hypercorn configuration for HTTP/3"""
    
    def test_http3_config_initialization(self):
        """Test HTTP3Config initialization"""
        api_config = APIConfig(
            server=APIServerConfig(host="0.0.0.0", port=8000, quic_port=8443),
            redis=RedisConfig(),
            gcs=GCSConfig(bucket="test-bucket"),
            security=SecurityConfig()
        )
        
        config_builder = HTTP3Config(api_config)
        assert config_builder.api_config == api_config
    
    def test_development_config(self):
        """Test development configuration"""
        api_config = APIConfig(
            server=APIServerConfig(host="127.0.0.1", port=8000, quic_port=8443),
            redis=RedisConfig(),
            gcs=GCSConfig(bucket="test-bucket"),
            security=SecurityConfig()
        )
        
        config_builder = HTTP3Config(api_config)
        hypercorn_config = config_builder.build_development_config()
        
        assert hypercorn_config.bind == ["127.0.0.1:8000"]
        assert hypercorn_config.workers == 1
        assert hypercorn_config.reload is True
        assert "h2c" in hypercorn_config.alpn_protocols
        assert "http/1.1" in hypercorn_config.alpn_protocols
    
    @patch('pathlib.Path.exists')
    def test_production_config(self, mock_exists):
        """Test production configuration with certificates"""
        mock_exists.return_value = True
        
        api_config = APIConfig(
            server=APIServerConfig(host="0.0.0.0", port=8000, quic_port=8443, workers=4),
            redis=RedisConfig(),
            gcs=GCSConfig(bucket="test-bucket"),
            security=SecurityConfig()
        )
        
        config_builder = HTTP3Config(api_config)
        hypercorn_config = config_builder.build_production_config(
            certfile="/path/to/cert.pem",
            keyfile="/path/to/key.pem"
        )
        
        assert hypercorn_config.bind == ["0.0.0.0:8000"]
        assert hypercorn_config.quic_bind == ["0.0.0.0:8443"]
        assert hypercorn_config.workers == 4
        assert hypercorn_config.reload is False
        assert "h3" in hypercorn_config.alpn_protocols
        assert "h2" in hypercorn_config.alpn_protocols
        assert "http/1.1" in hypercorn_config.alpn_protocols
        assert hypercorn_config.certfile == "/path/to/cert.pem"
        assert hypercorn_config.keyfile == "/path/to/key.pem"
    
    def test_create_hypercorn_config_development(self):
        """Test create_hypercorn_config for development"""
        api_config = APIConfig(
            server=APIServerConfig(),
            redis=RedisConfig(),
            gcs=GCSConfig(bucket="test-bucket"),
            security=SecurityConfig()
        )
        
        hypercorn_config = create_hypercorn_config(api_config, environment="development")
        
        assert hypercorn_config.workers == 1
        assert hypercorn_config.reload is True
    
    @patch('pathlib.Path.exists')
    def test_create_hypercorn_config_production(self, mock_exists):
        """Test create_hypercorn_config for production"""
        mock_exists.return_value = True
        
        api_config = APIConfig(
            server=APIServerConfig(workers=2),
            redis=RedisConfig(),
            gcs=GCSConfig(bucket="test-bucket"),
            security=SecurityConfig()
        )
        
        hypercorn_config = create_hypercorn_config(
            api_config,
            environment="production",
            certfile="/path/to/cert.pem",
            keyfile="/path/to/key.pem"
        )
        
        assert hypercorn_config.workers == 2
        assert hypercorn_config.reload is False
        assert hypercorn_config.certfile == "/path/to/cert.pem"
    
    def test_create_hypercorn_config_production_missing_certs(self):
        """Test create_hypercorn_config for production without certificates"""
        api_config = APIConfig(
            server=APIServerConfig(),
            redis=RedisConfig(),
            gcs=GCSConfig(bucket="test-bucket"),
            security=SecurityConfig()
        )
        
        with pytest.raises(ValueError, match="Certificate and key files required"):
            create_hypercorn_config(api_config, environment="production")


class TestFastAPIHTTP3Integration:
    """Test FastAPI integration with HTTP/3 middleware"""
    
    def test_app_includes_http3_middleware(self):
        """Test that FastAPI app includes HTTP3Middleware"""
        app = create_app()
        
        # Check that HTTP3Middleware is in the middleware stack
        middleware_classes = [middleware.cls for middleware in app.user_middleware]
        middleware_names = [cls.__name__ for cls in middleware_classes]
        
        assert "HTTP3Middleware" in middleware_names
    
    def test_app_startup_configures_http3(self):
        """Test that app startup configures HTTP/3 middleware"""
        # This would require a more complex integration test
        # For now, we verify the middleware is present
        app = create_app()
        
        # Find HTTP3Middleware
        http3_middleware = None
        for middleware in app.user_middleware:
            if middleware.cls.__name__ == "HTTP3Middleware":
                http3_middleware = middleware
                break
        
        assert http3_middleware is not None
        assert "quic_port" in http3_middleware.kwargs
        assert "enable_alt_svc" in http3_middleware.kwargs


@pytest.mark.integration
class TestHTTP3EndToEnd:
    """End-to-end tests for HTTP/3 functionality"""
    
    def test_middleware_stack_includes_http3(self):
        """Test that the middleware stack includes HTTP/3 middleware"""
        app = create_app()
        
        # Check that HTTP3Middleware is in the middleware stack
        middleware_classes = [middleware.cls for middleware in app.user_middleware]
        middleware_names = [cls.__name__ for cls in middleware_classes]
        
        assert "HTTP3Middleware" in middleware_names
    
    def test_hypercorn_config_creation(self):
        """Test Hypercorn configuration creation without Redis dependency"""
        from api.config import APIConfig, APIServerConfig, RedisConfig, GCSConfig, SecurityConfig
        from api.hypercorn_config import create_hypercorn_config
        
        api_config = APIConfig(
            server=APIServerConfig(host="0.0.0.0", port=8000, quic_port=8443),
            redis=RedisConfig(),
            gcs=GCSConfig(bucket="test-bucket"),
            security=SecurityConfig()
        )
        
        # Test development config
        dev_config = create_hypercorn_config(api_config, environment="development")
        assert dev_config.bind == ["0.0.0.0:8000"]
        assert dev_config.workers == 1
        assert dev_config.reload is True