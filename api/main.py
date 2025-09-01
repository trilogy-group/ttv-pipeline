"""
Main FastAPI application for the TTV Pipeline API server.

This module sets up the FastAPI application with proper middleware,
routing, and configuration management.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from api.config import get_config_from_env, APIConfig
from api.models import ErrorResponse
from api.logging_config import setup_structured_logging, get_logger, log_api_error
from api import __version__

# Configure structured logging
setup_structured_logging(
    level=logging.INFO,
    enable_console=True,
    enable_file=False  # Can be enabled via environment variable
)
logger = get_logger(__name__)

# Global configuration
config: APIConfig = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global config
    
    # Startup
    logger.info("Starting TTV Pipeline API server")
    try:
        config = get_config_from_env()
        logger.info(f"Configuration loaded successfully")
        logger.info(f"Server will bind to {config.server.host}:{config.server.port}")
        logger.info(f"QUIC/HTTP3 port: {config.server.quic_port}")
        logger.info(f"Redis: {config.redis.host}:{config.redis.port}")
        logger.info(f"GCS bucket: {config.gcs.bucket}")
        
        # Store config in app state for access by routes
        app.state.config = config
        
        # Initialize queue infrastructure
        from api.queue import initialize_queue_infrastructure
        redis_manager, job_queue = initialize_queue_infrastructure(config.redis)
        logger.info("Queue infrastructure initialized successfully")
        
        # Store queue infrastructure in app state for access by routes
        app.state.redis_manager = redis_manager
        app.state.job_queue = job_queue
        logger.info(f"Stored job_queue in app state: {job_queue is not None}")
        logger.info(f"App state keys: {list(app.state.__dict__.keys())}")
        
        # Update CORS middleware with actual origins from config
        cors_middleware = None
        for middleware in app.user_middleware:
            if middleware.cls.__name__ == "CORSMiddleware":
                cors_middleware = middleware
                break
        
        if cors_middleware:
            # Update CORS origins from configuration
            cors_middleware.kwargs["allow_origins"] = config.server.cors_origins
            logger.info(f"CORS origins configured: {config.server.cors_origins}")
        
        # Update rate limiting from configuration
        rate_limit_middleware = None
        for middleware in app.user_middleware:
            if hasattr(middleware.cls, '__name__') and middleware.cls.__name__ == "RateLimitMiddleware":
                rate_limit_middleware = middleware
                break
        
        if rate_limit_middleware:
            # Update rate limit from configuration
            rate_limit_middleware.kwargs["requests_per_minute"] = config.security.rate_limit_per_minute
            logger.info(f"Rate limit configured: {config.security.rate_limit_per_minute} requests/minute")
        
        # Update HTTP/3 middleware from configuration
        http3_middleware = None
        for middleware in app.user_middleware:
            if hasattr(middleware.cls, '__name__') and middleware.cls.__name__ == "HTTP3Middleware":
                http3_middleware = middleware
                break
        
        if http3_middleware:
            # Update QUIC port from configuration
            http3_middleware.kwargs["quic_port"] = config.server.quic_port
            logger.info(f"HTTP/3 QUIC port configured: {config.server.quic_port}")
        
        logger.info("Security middleware configured successfully")
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down TTV Pipeline API server")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    app = FastAPI(
        title="TTV Pipeline API",
        description="REST API for text-to-video generation pipeline",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Add middleware
    setup_middleware(app)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    # Add routes (will be implemented in subsequent tasks)
    setup_routes(app)
    
    return app


def setup_middleware(app: FastAPI):
    """Configure middleware for the application"""
    
    # Import middleware classes
    from api.middleware import (
        SecurityHeadersMiddleware,
        RequestLoggingMiddleware, 
        RateLimitMiddleware,
        RequestValidationMiddleware,
        HTTP3Middleware
    )
    
    # Request validation middleware (first in chain)
    app.add_middleware(RequestValidationMiddleware, max_content_length=10 * 1024 * 1024)
    
    # Rate limiting middleware
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=60,  # Will be updated from config in startup
        burst_limit=10,
        window_size=60
    )
    
    # HTTP/3 middleware for Alt-Svc headers and protocol negotiation
    app.add_middleware(
        HTTP3Middleware,
        quic_port=8443,  # Will be updated from config in startup
        enable_alt_svc=True
    )
    
    # Security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Request logging middleware (should be early but after security)
    app.add_middleware(
        RequestLoggingMiddleware,
        log_body=False,  # Set to True for debugging if needed
        redact_headers={'authorization', 'x-api-key', 'cookie', 'set-cookie'}
    )
    
    # CORS middleware (last in chain so it can add headers to all responses)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Will be updated from config in startup
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )


def setup_exception_handlers(app: FastAPI):
    """Configure global exception handlers with comprehensive error logging"""
    
    from api.exceptions import APIException
    from pydantic import ValidationError as PydanticValidationError
    
    @app.exception_handler(APIException)
    async def api_exception_handler(request: Request, exc: APIException):
        """Handle custom API exceptions with structured logging"""
        correlation_id = getattr(request.state, 'correlation_id', None)
        
        # Set correlation ID if not already set
        if correlation_id and not exc.correlation_id:
            exc.correlation_id = correlation_id
        
        # Create error response from exception
        error_dict = exc.to_dict()
        error_response = ErrorResponse(
            error=error_dict['error'],
            message=error_dict['message'],
            details=error_dict.get('details'),
            timestamp=datetime.fromisoformat(error_dict['timestamp'].replace('Z', '+00:00')),
            request_id=error_dict.get('request_id'),
            retryable=error_dict.get('retryable')
        )
        
        # Log the error with appropriate level and context
        log_level = logging.ERROR if exc.status_code >= 500 else logging.WARNING
        
        log_api_error(
            logger,
            exc,
            correlation_id or exc.correlation_id,
            method=request.method,
            path=request.url.path,
            status_code=exc.status_code,
            error_code=exc.error_code,
            retryable=exc.retryable,
            user_agent=request.headers.get("user-agent"),
            client_ip=_get_client_ip(request)
        )
        
        # Add retry headers for retryable errors
        headers = {}
        if exc.retryable and exc.status_code in [429, 503]:
            retry_after = exc.details.get('retry_after', 60)
            headers['Retry-After'] = str(retry_after)
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump(mode='json'),
            headers=headers
        )
    
    @app.exception_handler(PydanticValidationError)
    async def pydantic_validation_handler(request: Request, exc: PydanticValidationError):
        """Handle Pydantic validation errors with detailed field information"""
        correlation_id = getattr(request.state, 'correlation_id', None)
        
        # Extract validation error details
        validation_errors = []
        for error in exc.errors():
            validation_errors.append({
                'field': '.'.join(str(x) for x in error['loc']),
                'message': error['msg'],
                'type': error['type'],
                'input': error.get('input')
            })
        
        from api.exceptions import ValidationError
        api_exc = ValidationError(
            message="Request validation failed",
            validation_errors=validation_errors
        ).with_correlation_id(correlation_id)
        
        return await api_exception_handler(request, api_exc)
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle FastAPI HTTP exceptions with consistent error format"""
        correlation_id = getattr(request.state, 'correlation_id', None)
        
        error_response = ErrorResponse(
            error=exc.__class__.__name__,
            message=exc.detail,
            timestamp=datetime.now(timezone.utc),
            request_id=correlation_id
        )
        
        # Log with structured format
        logger.warning(
            f"HTTP error: {exc.detail}",
            extra={
                'extra': {
                    'correlation_id': correlation_id,
                    'event': 'http_error',
                    'status_code': exc.status_code,
                    'method': request.method,
                    'path': request.url.path,
                    'error_type': exc.__class__.__name__,
                    'client_ip': _get_client_ip(request),
                    'user_agent': request.headers.get("user-agent")
                }
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump(mode='json')
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions with comprehensive logging"""
        correlation_id = getattr(request.state, 'correlation_id', None)
        
        # Log the full exception with stack trace
        logger.error(
            f"Unhandled exception: {type(exc).__name__}: {str(exc)}",
            extra={
                'extra': {
                    'correlation_id': correlation_id,
                    'event': 'unhandled_exception',
                    'error_type': type(exc).__name__,
                    'error_message': str(exc),
                    'method': request.method,
                    'path': request.url.path,
                    'client_ip': _get_client_ip(request),
                    'user_agent': request.headers.get("user-agent"),
                    'query_params': dict(request.query_params)
                }
            },
            exc_info=True
        )
        
        error_response = ErrorResponse(
            error="InternalServerError",
            message="An internal server error occurred",
            timestamp=datetime.now(timezone.utc),
            request_id=correlation_id
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(mode='json')
        )


def _get_client_ip(request: Request) -> str:
    """Extract client IP address from request headers"""
    # Check for forwarded headers (common in proxy setups)
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip
    
    # Fall back to direct client IP
    if request.client:
        return request.client.host
    
    return "unknown"


def setup_routes(app: FastAPI):
    """Configure application routes"""
    
    # Import route modules
    from api.routes import jobs, health
    
    # Include job management routes
    app.include_router(jobs.router, prefix="/jobs")
    
    # Include health and monitoring routes  
    app.include_router(health.router)
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "name": "TTV Pipeline API",
            "version": __version__,
            "description": "REST API for text-to-video generation pipeline",
            "docs_url": "/docs"
        }


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    # Load config for development server
    config = get_config_from_env()
    
    uvicorn.run(
        "api.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=True,
        log_level="info"
    )