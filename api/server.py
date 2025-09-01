"""
Server startup script for the TTV Pipeline API with HTTP/3 support.

This module provides the main entry point for running the API server
using Hypercorn ASGI server with HTTP/3 over QUIC support.
"""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional

import click
from hypercorn.asyncio import serve
from hypercorn.config import Config as HypercornConfig

from api.config import get_config_from_env, APIConfig
from api.hypercorn_config import create_hypercorn_config
from api.main import app

logger = logging.getLogger(__name__)


class ServerManager:
    """Manages the HTTP/3 server lifecycle"""
    
    def __init__(self, api_config: APIConfig, hypercorn_config: HypercornConfig):
        self.api_config = api_config
        self.hypercorn_config = hypercorn_config
        self.shutdown_event = asyncio.Event()
    
    async def start_server(self):
        """Start the Hypercorn server with HTTP/3 support"""
        logger.info("Starting TTV Pipeline API server with HTTP/3 support")
        
        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        try:
            # Start the server
            await serve(app, self.hypercorn_config, shutdown_trigger=self.shutdown_event.wait)
            
        except Exception as e:
            logger.error(f"Server startup failed: {e}")
            raise
        finally:
            logger.info("Server shutdown complete")
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown_event.set()
        
        # Handle SIGTERM and SIGINT for graceful shutdown
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Handle SIGUSR1 for log rotation (if needed)
        if hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, lambda s, f: logger.info("Log rotation signal received"))


@click.command()
@click.option(
    '--environment', '-e',
    type=click.Choice(['development', 'production']),
    default='development',
    help='Environment to run in'
)
@click.option(
    '--certfile', '-c',
    type=click.Path(exists=True),
    help='Path to SSL certificate file (required for production)'
)
@click.option(
    '--keyfile', '-k',
    type=click.Path(exists=True),
    help='Path to SSL private key file (required for production)'
)
@click.option(
    '--ca-certs',
    type=click.Path(exists=True),
    help='Path to CA certificates file (optional)'
)
@click.option(
    '--host',
    help='Host to bind to (overrides config)'
)
@click.option(
    '--port',
    type=int,
    help='HTTP port to bind to (overrides config)'
)
@click.option(
    '--quic-port',
    type=int,
    help='QUIC/HTTP3 port to bind to (overrides config)'
)
@click.option(
    '--workers',
    type=int,
    help='Number of worker processes (overrides config)'
)
@click.option(
    '--reload',
    is_flag=True,
    help='Enable auto-reload for development'
)
def main(
    environment: str,
    certfile: Optional[str],
    keyfile: Optional[str],
    ca_certs: Optional[str],
    host: Optional[str],
    port: Optional[int],
    quic_port: Optional[int],
    workers: Optional[int],
    reload: bool
):
    """
    Start the TTV Pipeline API server with HTTP/3 support.
    
    Examples:
        # Development server (HTTP/1.1 and HTTP/2 only)
        python -m api.server --environment development
        
        # Production server with HTTP/3
        python -m api.server --environment production \\
            --certfile /path/to/cert.pem \\
            --keyfile /path/to/key.pem
        
        # Development with custom port
        python -m api.server --port 9000 --quic-port 9443
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        # Load API configuration
        api_config = get_config_from_env()
        
        # Apply CLI overrides
        if host:
            api_config.server.host = host
        if port:
            api_config.server.port = port
        if quic_port:
            api_config.server.quic_port = quic_port
        if workers:
            api_config.server.workers = workers
        
        # Validate production requirements
        if environment == "production":
            if not certfile or not keyfile:
                click.echo("Error: Certificate and key files are required for production", err=True)
                sys.exit(1)
        
        # Create Hypercorn configuration
        hypercorn_config = create_hypercorn_config(
            api_config=api_config,
            environment=environment,
            certfile=certfile,
            keyfile=keyfile,
            ca_certs=ca_certs
        )
        
        # Override reload setting if specified
        if reload:
            hypercorn_config.reload = True
            hypercorn_config.workers = 1  # Reload only works with single worker
        
        # Display startup information
        click.echo(f"Starting TTV Pipeline API server in {environment} mode")
        click.echo(f"HTTP bind: {api_config.server.host}:{api_config.server.port}")
        
        if environment == "production" and certfile and keyfile:
            click.echo(f"QUIC bind: {api_config.server.host}:{api_config.server.quic_port}")
            click.echo("HTTP/3 over QUIC: Enabled")
        else:
            click.echo("HTTP/3 over QUIC: Disabled (development mode or no certificates)")
        
        click.echo(f"Workers: {hypercorn_config.workers}")
        click.echo(f"ALPN protocols: {hypercorn_config.alpn_protocols}")
        
        # Create and start server
        server_manager = ServerManager(api_config, hypercorn_config)
        
        # Run the server
        asyncio.run(server_manager.start_server())
        
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)


def run_development_server():
    """Convenience function to run development server"""
    os.environ.setdefault('API_CONFIG_PATH', '')
    main(['--environment', 'development'])


def run_production_server(certfile: str, keyfile: str, ca_certs: Optional[str] = None):
    """
    Convenience function to run production server
    
    Args:
        certfile: Path to SSL certificate file
        keyfile: Path to SSL private key file
        ca_certs: Path to CA certificates file (optional)
    """
    args = ['--environment', 'production', '--certfile', certfile, '--keyfile', keyfile]
    if ca_certs:
        args.extend(['--ca-certs', ca_certs])
    
    main(args)


if __name__ == "__main__":
    main()