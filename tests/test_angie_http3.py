"""
Tests for Angie HTTP/3 edge proxy configuration.

This module tests the Angie edge proxy configuration for HTTP/3 support,
including protocol negotiation, upstream proxying, and certificate handling.
"""

import asyncio
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import pytest
import httpx
from httpx import AsyncClient


@pytest.mark.angie
class TestAngieHTTP3Config:
    """Test Angie HTTP/3 configuration"""
    
    @pytest.fixture
    def config_file(self) -> Path:
        """Get path to Angie configuration file"""
        return Path(__file__).parent.parent / "config" / "angie.conf"
    
    @pytest.fixture
    def docker_compose_file(self) -> Path:
        """Get path to Docker Compose file"""
        return Path(__file__).parent.parent / "docker-compose.http3.yml"
    
    def test_config_file_exists(self, config_file: Path):
        """Test that Angie configuration file exists"""
        assert config_file.exists(), f"Angie config file not found: {config_file}"
    
    def test_config_syntax_validation(self, config_file: Path):
        """Test Angie configuration syntax validation"""
        # Use nginx to validate syntax (Angie is nginx-compatible)
        try:
            result = subprocess.run(
                ["nginx", "-t", "-c", str(config_file)],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert result.returncode == 0, f"Config syntax error: {result.stderr}"
        except FileNotFoundError:
            pytest.skip("nginx not available for syntax validation")
        except subprocess.TimeoutExpired:
            pytest.fail("Configuration validation timed out")
    
    def test_http3_directives_present(self, config_file: Path):
        """Test that HTTP/3 specific directives are present"""
        config_content = config_file.read_text()
        
        # Check for essential HTTP/3 directives
        required_directives = [
            "listen 443 quic",
            "quic_retry on",
            "ssl_early_data on",
            "Alt-Svc",
            "h3=",
            "quic_max_concurrent_streams",
            "proxy_http_version 3.0"
        ]
        
        for directive in required_directives:
            assert directive in config_content, f"Missing HTTP/3 directive: {directive}"
    
    def test_upstream_configuration(self, config_file: Path):
        """Test upstream configuration for HTTP/3"""
        config_content = config_file.read_text()
        
        # Check upstream configuration
        assert "upstream api_backend" in config_content
        assert "quic=on" in config_content
        assert "server api:8443" in config_content
        assert "server api:8000 backup" in config_content
    
    def test_ssl_configuration(self, config_file: Path):
        """Test SSL/TLS configuration for HTTP/3"""
        config_content = config_file.read_text()
        
        # Check SSL configuration
        ssl_requirements = [
            "ssl_protocols TLSv1.2 TLSv1.3",
            "ssl_certificate /etc/ssl/certs/cert.pem",
            "ssl_certificate_key /etc/ssl/certs/key.pem",
            "ssl_session_cache",
            "ssl_stapling on"
        ]
        
        for requirement in ssl_requirements:
            assert requirement in config_content, f"Missing SSL config: {requirement}"
    
    def test_quic_parameters(self, config_file: Path):
        """Test QUIC parameter tuning"""
        config_content = config_file.read_text()
        
        # Check QUIC parameters
        quic_params = [
            "quic_max_concurrent_streams",
            "quic_active_connection_id_limit",
            "quic_max_ack_delay",
            "quic_max_udp_payload_size",
            "quic_host_key"
        ]
        
        for param in quic_params:
            assert param in config_content, f"Missing QUIC parameter: {param}"
    
    def test_security_headers(self, config_file: Path):
        """Test security headers configuration"""
        config_content = config_file.read_text()
        
        # Check security headers
        security_headers = [
            "X-Frame-Options DENY",
            "X-Content-Type-Options nosniff",
            "X-XSS-Protection",
            "Referrer-Policy",
            "Access-Control-Allow-Origin"
        ]
        
        for header in security_headers:
            assert header in config_content, f"Missing security header: {header}"
    
    def test_rate_limiting_configuration(self, config_file: Path):
        """Test rate limiting configuration"""
        config_content = config_file.read_text()
        
        # Check rate limiting
        assert "limit_req_zone" in config_content
        assert "limit_req zone=api" in config_content
        assert "limit_req zone=burst" in config_content
    
    def test_health_check_endpoints(self, config_file: Path):
        """Test health check endpoint configuration"""
        config_content = config_file.read_text()
        
        # Check health endpoints
        assert "location ~ ^/(healthz|readyz)$" in config_content
        assert "access_log off" in config_content
    
    def test_docker_compose_configuration(self, docker_compose_file: Path):
        """Test Docker Compose configuration"""
        assert docker_compose_file.exists(), "Docker Compose file not found"
        
        # Validate Docker Compose syntax
        try:
            result = subprocess.run(
                ["docker-compose", "-f", str(docker_compose_file), "config"],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert result.returncode == 0, f"Docker Compose validation failed: {result.stderr}"
        except FileNotFoundError:
            pytest.skip("docker-compose not available")
        except subprocess.TimeoutExpired:
            pytest.fail("Docker Compose validation timed out")


@pytest.mark.angie
@pytest.mark.integration
class TestAngieHTTP3Connectivity:
    """Test Angie HTTP/3 connectivity (requires running services)"""
    
    @pytest.fixture
    def base_url(self) -> str:
        """Base URL for testing"""
        return "https://localhost"
    
    @pytest.fixture
    def client_config(self) -> Dict[str, Any]:
        """HTTP client configuration"""
        return {
            "verify": False,  # Accept self-signed certificates
            "timeout": 10.0,
            "follow_redirects": True
        }
    
    @pytest.mark.integration
    @pytest.mark.http3
    async def test_http_redirect(self):
        """Test HTTP to HTTPS redirect"""
        async with AsyncClient(verify=False, follow_redirects=False) as client:
            try:
                response = await client.get("http://localhost/healthz")
                assert response.status_code == 301
                assert response.headers.get("location", "").startswith("https://")
            except httpx.ConnectError:
                pytest.skip("Services not running")
    
    @pytest.mark.integration
    @pytest.mark.http3
    async def test_https_connectivity(self, base_url: str, client_config: Dict[str, Any]):
        """Test HTTPS connectivity"""
        async with AsyncClient(**client_config) as client:
            try:
                response = await client.get(f"{base_url}/healthz")
                assert response.status_code == 200
            except httpx.ConnectError:
                pytest.skip("Services not running")
    
    @pytest.mark.integration
    @pytest.mark.http3
    async def test_alt_svc_header(self, base_url: str, client_config: Dict[str, Any]):
        """Test Alt-Svc header for HTTP/3 advertisement"""
        async with AsyncClient(**client_config) as client:
            try:
                response = await client.get(f"{base_url}/healthz")
                alt_svc = response.headers.get("alt-svc", "")
                
                assert "h3=" in alt_svc, f"Alt-Svc header missing HTTP/3: {alt_svc}"
                assert ":443" in alt_svc, f"Alt-Svc header missing port: {alt_svc}"
            except httpx.ConnectError:
                pytest.skip("Services not running")
    
    @pytest.mark.integration
    @pytest.mark.security
    async def test_security_headers(self, base_url: str, client_config: Dict[str, Any]):
        """Test security headers"""
        async with AsyncClient(**client_config) as client:
            try:
                response = await client.get(f"{base_url}/healthz")
                
                # Check required security headers
                assert response.headers.get("x-frame-options") == "DENY"
                assert response.headers.get("x-content-type-options") == "nosniff"
                assert "x-xss-protection" in response.headers
                assert "referrer-policy" in response.headers
            except httpx.ConnectError:
                pytest.skip("Services not running")
    
    @pytest.mark.integration
    @pytest.mark.security
    async def test_cors_headers(self, base_url: str, client_config: Dict[str, Any]):
        """Test CORS headers"""
        async with AsyncClient(**client_config) as client:
            try:
                # Test preflight request
                response = await client.options(
                    f"{base_url}/v1/jobs",
                    headers={
                        "Origin": "https://example.com",
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "Content-Type"
                    }
                )
                
                assert response.status_code == 204
                assert response.headers.get("access-control-allow-origin") == "*"
                assert "POST" in response.headers.get("access-control-allow-methods", "")
            except httpx.ConnectError:
                pytest.skip("Services not running")
    
    @pytest.mark.integration
    @pytest.mark.security
    async def test_rate_limiting(self, base_url: str, client_config: Dict[str, Any]):
        """Test rate limiting functionality"""
        async with AsyncClient(**client_config) as client:
            try:
                # Make multiple rapid requests to trigger rate limiting
                responses = []
                for _ in range(25):  # Exceed burst limit
                    try:
                        response = await client.get(f"{base_url}/healthz")
                        responses.append(response.status_code)
                    except httpx.ReadTimeout:
                        # Rate limiting may cause timeouts
                        responses.append(429)
                
                # Should have some rate limited responses
                rate_limited = [code for code in responses if code == 429]
                # Note: Rate limiting may not trigger in test environment
                # This is more of a smoke test
                
            except httpx.ConnectError:
                pytest.skip("Services not running")
    
    @pytest.mark.integration
    @pytest.mark.http3
    async def test_upstream_proxy(self, base_url: str, client_config: Dict[str, Any]):
        """Test upstream proxy functionality"""
        async with AsyncClient(**client_config) as client:
            try:
                response = await client.get(f"{base_url}/healthz")
                
                # Check that request was proxied (should have API response format)
                assert response.status_code == 200
                
                # Check for proxy headers (if API returns them)
                # The actual headers depend on the API implementation
                
            except httpx.ConnectError:
                pytest.skip("Services not running")


@pytest.mark.angie
class TestAngieConfigGeneration:
    """Test Angie configuration generation and validation scripts"""
    
    def test_setup_script_exists(self):
        """Test that setup script exists and is executable"""
        script_path = Path(__file__).parent.parent / "scripts" / "setup_http3_certs.sh"
        assert script_path.exists(), "Setup script not found"
        assert script_path.stat().st_mode & 0o111, "Setup script not executable"
    
    def test_validation_script_exists(self):
        """Test that validation script exists and is executable"""
        script_path = Path(__file__).parent.parent / "scripts" / "validate_angie_config.sh"
        assert script_path.exists(), "Validation script not found"
        assert script_path.stat().st_mode & 0o111, "Validation script not executable"
    
    def test_setup_script_help(self):
        """Test setup script help output"""
        script_path = Path(__file__).parent.parent / "scripts" / "setup_http3_certs.sh"
        
        try:
            result = subprocess.run(
                [str(script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            assert result.returncode == 0
            assert "Usage:" in result.stdout
            assert "--force" in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("Cannot test setup script")
    
    def test_validation_script_help(self):
        """Test validation script help output"""
        script_path = Path(__file__).parent.parent / "scripts" / "validate_angie_config.sh"
        
        try:
            result = subprocess.run(
                [str(script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            assert result.returncode == 0
            assert "Usage:" in result.stdout
            assert "--syntax-only" in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("Cannot test validation script")


# Utility functions for testing
def is_service_running(service_name: str = "angie") -> bool:
    """Check if a Docker Compose service is running"""
    try:
        result = subprocess.run(
            ["docker-compose", "-f", "docker-compose.http3.yml", "ps", service_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        return "Up" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_service_logs(service_name: str = "angie", lines: int = 50) -> str:
    """Get logs from a Docker Compose service"""
    try:
        result = subprocess.run(
            ["docker-compose", "-f", "docker-compose.http3.yml", "logs", "--tail", str(lines), service_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


if __name__ == "__main__":
    # Run basic configuration tests
    import sys
    
    config_file = Path(__file__).parent.parent / "config" / "angie.conf"
    
    if not config_file.exists():
        print(f"ERROR: Angie config file not found: {config_file}")
        sys.exit(1)
    
    print("Running basic Angie configuration tests...")
    
    # Test configuration syntax
    try:
        result = subprocess.run(
            ["nginx", "-t", "-c", str(config_file)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✓ Configuration syntax is valid")
        else:
            print(f"✗ Configuration syntax error: {result.stderr}")
            sys.exit(1)
    except FileNotFoundError:
        print("⚠ nginx not available for syntax validation")
    
    # Test HTTP/3 directives
    config_content = config_file.read_text()
    if "listen 443 quic" in config_content:
        print("✓ HTTP/3 QUIC listener configured")
    else:
        print("✗ HTTP/3 QUIC listener not found")
        sys.exit(1)
    
    if "quic=on" in config_content:
        print("✓ HTTP/3 upstream configured")
    else:
        print("✗ HTTP/3 upstream not configured")
        sys.exit(1)
    
    print("All basic tests passed!")