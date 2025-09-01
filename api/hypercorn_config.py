"""
Hypercorn ASGI server configuration for HTTP/3 support.

This module provides configuration for running the FastAPI application
with Hypercorn ASGI server supporting HTTP/3 over QUIC with fallbacks
to HTTP/2 and HTTP/1.1.
"""

import logging
import ssl
from pathlib import Path
from typing import Optional, List, Dict, Any

from hypercorn.config import Config as HypercornConfig
from api.config import APIConfig

logger = logging.getLogger(__name__)


class HTTP3Config:
    """Configuration builder for Hypercorn with HTTP/3 support"""
    
    def __init__(self, api_config: APIConfig):
        self.api_config = api_config
        self.hypercorn_config = HypercornConfig()
    
    def build_config(
        self,
        certfile: Optional[str] = None,
        keyfile: Optional[str] = None,
        ca_certs: Optional[str] = None,
        verify_mode: Optional[ssl.VerifyMode] = None
    ) -> HypercornConfig:
        """
        Build Hypercorn configuration with HTTP/3 support.
        
        Args:
            certfile: Path to SSL certificate file
            keyfile: Path to SSL private key file
            ca_certs: Path to CA certificates file
            verify_mode: SSL verification mode
            
        Returns:
            Configured Hypercorn config instance
        """
        config = self.hypercorn_config
        
        # Basic server configuration
        config.bind = [
            f"{self.api_config.server.host}:{self.api_config.server.port}",  # HTTP/1.1, HTTP/2
        ]
        
        # Add QUIC binding for HTTP/3 if certificates are provided
        if certfile and keyfile:
            config.quic_bind = [
                f"{self.api_config.server.host}:{self.api_config.server.quic_port}"
            ]
            config.certfile = certfile
            config.keyfile = keyfile
            
            if ca_certs:
                config.ca_certs = ca_certs
            
            if verify_mode:
                config.verify_mode = verify_mode
            
            # HTTP/3 specific configuration
            config.alpn_protocols = ["h3", "h2", "http/1.1"]
            
            logger.info(f"HTTP/3 enabled on {self.api_config.server.host}:{self.api_config.server.quic_port}")
        else:
            # HTTP/2 and HTTP/1.1 only
            config.alpn_protocols = ["h2", "http/1.1"]
            logger.info("HTTP/3 disabled - no certificates provided")
        
        # Worker configuration
        config.workers = self.api_config.server.workers
        
        # Logging configuration
        config.loglevel = "info"
        config.access_log_format = (
            '%(h)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'
        )
        
        # Performance tuning
        config.keep_alive_timeout = 65
        config.max_requests = 1000
        config.max_requests_jitter = 100
        
        # HTTP/3 and QUIC specific settings
        if certfile and keyfile:
            # QUIC connection settings
            config.quic_max_concurrent_connections = 1000
            config.quic_stateless_reset_token_key = None  # Will be auto-generated
            
            # HTTP/3 settings
            config.h3_max_concurrent_streams = 100
            config.h3_max_header_list_size = 16384
            
        # Security headers
        config.server_names = ["*"]  # Accept any server name for internal use
        
        # Graceful shutdown
        config.graceful_timeout = 30
        
        logger.info(f"Hypercorn configured with {config.workers} workers")
        logger.info(f"Bind addresses: {config.bind}")
        if hasattr(config, 'quic_bind') and config.quic_bind:
            logger.info(f"QUIC bind addresses: {config.quic_bind}")
        logger.info(f"ALPN protocols: {config.alpn_protocols}")
        
        return config
    
    def build_development_config(self) -> HypercornConfig:
        """
        Build configuration for development (no HTTPS/HTTP3).
        
        Returns:
            Hypercorn config for development use
        """
        config = self.hypercorn_config
        
        # Development server configuration
        config.bind = [f"{self.api_config.server.host}:{self.api_config.server.port}"]
        config.workers = 1  # Single worker for development
        config.reload = True
        config.loglevel = "debug"
        
        # HTTP/1.1 and HTTP/2 without encryption
        config.alpn_protocols = ["h2c", "http/1.1"]  # h2c = HTTP/2 cleartext
        
        # Development-friendly settings
        config.keep_alive_timeout = 5
        config.graceful_timeout = 5
        
        logger.info("Development configuration - HTTP/3 disabled")
        logger.info(f"Bind address: {config.bind[0]}")
        
        return config
    
    def build_production_config(
        self,
        certfile: str,
        keyfile: str,
        ca_certs: Optional[str] = None
    ) -> HypercornConfig:
        """
        Build configuration for production with full HTTP/3 support.
        
        Args:
            certfile: Path to SSL certificate file
            keyfile: Path to SSL private key file
            ca_certs: Path to CA certificates file (optional)
            
        Returns:
            Hypercorn config for production use
        """
        # Validate certificate files
        if not Path(certfile).exists():
            raise FileNotFoundError(f"Certificate file not found: {certfile}")
        
        if not Path(keyfile).exists():
            raise FileNotFoundError(f"Private key file not found: {keyfile}")
        
        if ca_certs and not Path(ca_certs).exists():
            raise FileNotFoundError(f"CA certificates file not found: {ca_certs}")
        
        config = self.build_config(
            certfile=certfile,
            keyfile=keyfile,
            ca_certs=ca_certs,
            verify_mode=ssl.CERT_NONE  # Internal service, no client cert verification
        )
        
        # Production-specific settings
        config.workers = self.api_config.server.workers
        config.reload = False
        config.loglevel = "info"
        
        # Enhanced security for production
        config.ssl_handshake_timeout = 10
        config.ssl_shutdown_timeout = 5
        
        # Performance optimization
        config.max_requests = 10000
        config.max_requests_jitter = 1000
        config.keep_alive_timeout = 75
        
        logger.info("Production configuration with HTTP/3 support")
        
        return config


def create_hypercorn_config(
    api_config: APIConfig,
    environment: str = "development",
    certfile: Optional[str] = None,
    keyfile: Optional[str] = None,
    ca_certs: Optional[str] = None
) -> HypercornConfig:
    """
    Create Hypercorn configuration based on environment.
    
    Args:
        api_config: API configuration instance
        environment: Environment type ("development" or "production")
        certfile: Path to SSL certificate file
        keyfile: Path to SSL private key file
        ca_certs: Path to CA certificates file
        
    Returns:
        Configured Hypercorn config instance
    """
    config_builder = HTTP3Config(api_config)
    
    if environment == "production":
        if not certfile or not keyfile:
            raise ValueError("Certificate and key files required for production")
        return config_builder.build_production_config(certfile, keyfile, ca_certs)
    else:
        return config_builder.build_development_config()


def get_ssl_context(
    certfile: str,
    keyfile: str,
    ca_certs: Optional[str] = None
) -> ssl.SSLContext:
    """
    Create SSL context for HTTPS/HTTP3 support.
    
    Args:
        certfile: Path to SSL certificate file
        keyfile: Path to SSL private key file
        ca_certs: Path to CA certificates file
        
    Returns:
        Configured SSL context
    """
    # Create SSL context with TLS 1.3 support
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    
    # Load certificate and private key
    context.load_cert_chain(certfile, keyfile)
    
    # Load CA certificates if provided
    if ca_certs:
        context.load_verify_locations(ca_certs)
    
    # Configure for HTTP/3 and HTTP/2
    context.set_alpn_protocols(["h3", "h2", "http/1.1"])
    
    # Security settings
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.maximum_version = ssl.TLSVersion.TLSv1_3
    
    # Prefer TLS 1.3
    context.options |= ssl.OP_PREFER_TLS_1_3
    
    # Disable compression to prevent CRIME attacks
    context.options |= ssl.OP_NO_COMPRESSION
    
    logger.info("SSL context configured for HTTP/3 support")
    
    return context