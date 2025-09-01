"""
Middleware components for the API server.

This module contains security, logging, and other middleware components
for request processing, validation, and tracing.
"""

import json
import logging
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Callable, Dict, Optional, Set
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api.exceptions import (
    RateLimitError, ValidationError, PayloadTooLargeError, 
    UnsupportedMediaTypeError
)
from api.models import ErrorResponse
from api.logging_config import get_logger, log_api_request, log_api_response, log_api_error

logger = get_logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers appropriate for internal services.
    
    Adds comprehensive security headers for internal network deployment
    while maintaining compatibility with API clients.
    """
    
    def __init__(self, app, trusted_hosts: Optional[Set[str]] = None):
        super().__init__(app)
        self.trusted_hosts = trusted_hosts or set()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Security headers appropriate for internal services
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        
        # Content Security Policy for internal services
        # Allow CDN resources for API documentation (Swagger UI and ReDoc)
        if request.url.path in ["/docs", "/redoc"]:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https://fastapi.tiangolo.com; "
                "connect-src 'self'"
            )
        else:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "connect-src 'self'"
            )
        
        # Cache control for API responses
        if request.url.path.startswith("/v1/"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured request logging with correlation IDs.
    
    Provides comprehensive request/response logging with correlation tracking
    and sensitive data redaction for security.
    """
    
    def __init__(self, app, log_body: bool = False, redact_headers: Optional[Set[str]] = None):
        super().__init__(app)
        self.log_body = log_body
        self.redact_headers = redact_headers or {
            'authorization', 'x-api-key', 'cookie', 'set-cookie'
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Generate correlation ID for request tracing
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Extract client information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log request using structured logging
        log_api_request(
            logger,
            method=request.method,
            path=request.url.path,
            correlation_id=correlation_id,
            client_ip=client_ip,
            user_agent=user_agent,
            query_params=dict(request.query_params),
            headers=self._redact_headers(dict(request.headers)),
            content_length=request.headers.get("content-length"),
            content_type=request.headers.get("content-type")
        )
        
        # Log request body for POST/PUT requests if enabled
        if self.log_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    # Try to parse as JSON for better logging
                    try:
                        body_data = json.loads(body.decode())
                        logger.debug(
                            "Request body content",
                            extra={
                                'extra': {
                                    'correlation_id': correlation_id,
                                    'event': 'request_body',
                                    'body': body_data
                                }
                            }
                        )
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        logger.debug(
                            f"Request body size: {len(body)} bytes",
                            extra={
                                'extra': {
                                    'correlation_id': correlation_id,
                                    'event': 'request_body',
                                    'body_size': len(body)
                                }
                            }
                        )
            except Exception as e:
                logger.warning(
                    f"Failed to read request body: {e}",
                    extra={
                        'extra': {
                            'correlation_id': correlation_id,
                            'event': 'request_body_error',
                            'error': str(e)
                        }
                    }
                )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Log successful response
            duration = time.time() - start_time
            log_api_response(
                logger,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2),
                correlation_id=correlation_id,
                response_headers=self._redact_headers(dict(response.headers)),
                content_length=response.headers.get("content-length")
            )
            
            # Record enhanced metrics for monitoring
            try:
                from api.routes.health import record_request_latency
                record_request_latency(duration, response.status_code)
            except ImportError:
                pass  # Health module not available yet
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
            
        except Exception as e:
            # Log error response
            duration = time.time() - start_time
            log_api_error(
                logger,
                e,
                correlation_id,
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration * 1000, 2),
                client_ip=client_ip,
                user_agent=user_agent
            )
            
            # Re-raise the exception to be handled by exception handlers
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request headers"""
        # Check for forwarded headers (common in proxy setups)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _redact_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Redact sensitive headers for logging"""
        redacted = {}
        for key, value in headers.items():
            if key.lower() in self.redact_headers:
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = value
        return redacted


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware for API protection.
    
    Implements sliding window rate limiting per client IP with configurable
    limits and time windows.
    """
    
    def __init__(
        self, 
        app, 
        requests_per_minute: int = 60,
        burst_limit: int = 10,
        window_size: int = 60
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.window_size = window_size
        
        # In-memory storage for rate limiting (Redis would be better for production)
        self.request_counts: Dict[str, deque] = defaultdict(deque)
        self.burst_counts: Dict[str, int] = defaultdict(int)
        self.last_reset: Dict[str, float] = defaultdict(float)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ["/healthz", "/readyz", "/metrics"]:
            return await call_next(request)
        
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Check rate limits
        if self._is_rate_limited(client_ip, current_time):
            correlation_id = getattr(request.state, 'correlation_id', None)
            
            # Create enhanced rate limit error
            rate_limit_error = RateLimitError(
                message=f"Rate limit exceeded: {self.requests_per_minute} requests per minute",
                limit=self.requests_per_minute,
                window_seconds=self.window_size,
                retry_after=60
            ).with_correlation_id(correlation_id)
            
            # Log rate limit violation
            logger.warning(
                f"Rate limit exceeded for client {client_ip}",
                extra={
                    'extra': {
                        'event': 'rate_limit_exceeded',
                        'client_ip': client_ip,
                        'correlation_id': correlation_id,
                        'rate_limit': self.requests_per_minute,
                        'window_seconds': self.window_size,
                        'current_requests': len(self.request_counts[client_ip])
                    }
                }
            )
            
            error_response = ErrorResponse(**rate_limit_error.to_dict())
            
            return JSONResponse(
                status_code=429,
                content=error_response.model_dump(mode='json'),
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(current_time + 60))
                }
            )
        
        # Record the request
        self._record_request(client_ip, current_time)
        
        response = await call_next(request)
        
        # Add rate limit headers to response
        remaining = self._get_remaining_requests(client_ip, current_time)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _is_rate_limited(self, client_ip: str, current_time: float) -> bool:
        """Check if client is rate limited"""
        # Clean old entries
        self._cleanup_old_entries(client_ip, current_time)
        
        # Check sliding window rate limit
        request_times = self.request_counts[client_ip]
        if len(request_times) >= self.requests_per_minute:
            return True
        
        # Check burst limit (requests in last 10 seconds)
        recent_requests = sum(1 for t in request_times if current_time - t <= 10)
        if recent_requests >= self.burst_limit:
            return True
        
        return False
    
    def _record_request(self, client_ip: str, current_time: float):
        """Record a request for rate limiting"""
        self.request_counts[client_ip].append(current_time)
        self._cleanup_old_entries(client_ip, current_time)
    
    def _cleanup_old_entries(self, client_ip: str, current_time: float):
        """Remove old entries outside the time window"""
        request_times = self.request_counts[client_ip]
        while request_times and current_time - request_times[0] > self.window_size:
            request_times.popleft()
    
    def _get_remaining_requests(self, client_ip: str, current_time: float) -> int:
        """Get remaining requests for client"""
        self._cleanup_old_entries(client_ip, current_time)
        return max(0, self.requests_per_minute - len(self.request_counts[client_ip]))


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for basic request validation and security checks.
    
    Performs early validation of requests before they reach route handlers.
    """
    
    def __init__(self, app, max_content_length: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_content_length = max_content_length
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_content_length:
            correlation_id = getattr(request.state, 'correlation_id', None)
            
            payload_error = PayloadTooLargeError(
                message=f"Request body too large. Maximum size: {self.max_content_length} bytes",
                max_size=self.max_content_length,
                actual_size=int(content_length)
            ).with_correlation_id(correlation_id)
            
            logger.warning(
                f"Request payload too large: {content_length} bytes",
                extra={
                    'extra': {
                        'event': 'payload_too_large',
                        'correlation_id': correlation_id,
                        'max_size': self.max_content_length,
                        'actual_size': int(content_length),
                        'method': request.method,
                        'path': request.url.path
                    }
                }
            )
            
            error_response = ErrorResponse(**payload_error.to_dict())
            
            return JSONResponse(
                status_code=413,
                content=error_response.model_dump(mode='json')
            )
        
        # Validate content type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("application/json"):
                correlation_id = getattr(request.state, 'correlation_id', None)
                
                media_type_error = UnsupportedMediaTypeError(
                    message="Content-Type must be application/json",
                    provided_type=content_type,
                    supported_types=["application/json"]
                ).with_correlation_id(correlation_id)
                
                logger.warning(
                    f"Unsupported media type: {content_type}",
                    extra={
                        'extra': {
                            'event': 'unsupported_media_type',
                            'correlation_id': correlation_id,
                            'provided_type': content_type,
                            'supported_types': ["application/json"],
                            'method': request.method,
                            'path': request.url.path
                        }
                    }
                )
                
                error_response = ErrorResponse(**media_type_error.to_dict())
                
                return JSONResponse(
                    status_code=415,
                    content=error_response.model_dump(mode='json')
                )
        
        return await call_next(request)


class HTTP3Middleware(BaseHTTPMiddleware):
    """
    Middleware for HTTP/3 protocol support and Alt-Svc header management.
    
    Adds Alt-Svc headers to advertise HTTP/3 availability and handles
    protocol negotiation for optimal client performance.
    """
    
    def __init__(self, app, quic_port: Optional[int] = None, enable_alt_svc: bool = True):
        super().__init__(app)
        self.quic_port = quic_port
        self.enable_alt_svc = enable_alt_svc
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Get configuration from app state if available
        config = getattr(request.app.state, 'config', None)
        quic_port = self.quic_port or (config.server.quic_port if config else 8443)
        
        # Add Alt-Svc header to advertise HTTP/3 availability
        if self.enable_alt_svc and self._should_add_alt_svc(request):
            # Get the host from the request
            host = request.headers.get("host", "").split(":")[0]
            
            # Add Alt-Svc header for HTTP/3
            alt_svc_value = f'h3=":{quic_port}"; ma=86400'
            
            # Also advertise HTTP/2 as fallback
            alt_svc_value += f', h2=":{request.url.port or 443}"; ma=86400'
            
            response.headers["Alt-Svc"] = alt_svc_value
        
        # Add HTTP/3 specific headers
        self._add_http3_headers(request, response)
        
        return response
    
    def _should_add_alt_svc(self, request: Request) -> bool:
        """Determine if Alt-Svc header should be added"""
        # Only add Alt-Svc for HTTPS requests or when explicitly enabled
        is_https = request.url.scheme == "https"
        is_forwarded_https = request.headers.get("x-forwarded-proto") == "https"
        
        # Add for API endpoints
        is_api_endpoint = request.url.path.startswith("/v1/")
        
        return (is_https or is_forwarded_https) and is_api_endpoint
    
    def _add_http3_headers(self, request: Request, response: Response):
        """Add HTTP/3 specific headers for optimization"""
        # Add protocol version header for debugging
        protocol_version = self._get_protocol_version(request)
        if protocol_version:
            response.headers["X-Protocol-Version"] = protocol_version
        
        # Add connection info for monitoring
        if hasattr(request, 'scope') and 'http_version' in request.scope:
            response.headers["X-HTTP-Version"] = request.scope['http_version']
        
        # Add QUIC-specific headers if using HTTP/3
        if protocol_version == "h3":
            # Add server push hints for related resources (if needed)
            if request.url.path == "/":
                response.headers["Link"] = '</docs>; rel=preload; as=document'
            
            # Add connection coalescing hints
            response.headers["Connection"] = "keep-alive"
    
    def _get_protocol_version(self, request: Request) -> Optional[str]:
        """Extract protocol version from request"""
        # Check ALPN protocol from request scope
        if hasattr(request, 'scope'):
            # HTTP/3 over QUIC
            if request.scope.get('scheme') == 'https' and 'quic' in str(request.scope.get('server', '')):
                return "h3"
            
            # HTTP/2
            http_version = request.scope.get('http_version', '')
            if http_version == '2.0':
                return "h2"
            elif http_version == '1.1':
                return "http/1.1"
            elif http_version == '1.0':
                return "http/1.0"
        
        # Check headers for protocol hints
        upgrade = request.headers.get("upgrade", "").lower()
        if "h2c" in upgrade:
            return "h2c"
        
        return None