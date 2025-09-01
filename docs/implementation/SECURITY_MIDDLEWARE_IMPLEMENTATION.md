# Security Middleware Implementation Summary

## Task 4: Implement Basic Security Middleware

This document summarizes the implementation of the security middleware for the TTV Pipeline API server, completed as part of task 4 from the API server specification.

## Requirements Addressed

Based on requirement 5.3 from the specification:
- ✅ Add security headers and CORS configuration for internal network use
- ✅ Implement request validation and basic error handling  
- ✅ Add request logging and correlation IDs for tracing

## Components Implemented

### 1. Security Headers Middleware (`SecurityHeadersMiddleware`)

**Location**: `api/middleware.py`

**Features**:
- Adds comprehensive security headers appropriate for internal services:
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `X-XSS-Protection: 1; mode=block`
  - `Referrer-Policy: strict-origin-when-cross-origin`
  - `X-Permitted-Cross-Domain-Policies: none`
  - Content Security Policy for internal services
- Adds cache control headers for API endpoints (`/v1/*` paths):
  - `Cache-Control: no-cache, no-store, must-revalidate`
  - `Pragma: no-cache`
  - `Expires: 0`

### 2. Request Logging Middleware (`RequestLoggingMiddleware`)

**Location**: `api/middleware.py`

**Features**:
- Generates UUID-based correlation IDs for request tracing
- Structured JSON logging with comprehensive request/response data
- Sensitive header redaction (authorization, cookies, etc.)
- Client IP extraction from forwarded headers
- Request/response timing and performance metrics
- Error logging with correlation tracking

**Log Structure**:
```json
{
  "event": "request_start",
  "correlation_id": "uuid-string",
  "method": "GET",
  "path": "/api/endpoint",
  "query_params": {},
  "client_ip": "192.168.1.1",
  "user_agent": "client/1.0",
  "headers": {"header": "value"},
  "timestamp": "2025-08-31T12:34:56Z"
}
```

### 3. Rate Limiting Middleware (`RateLimitMiddleware`)

**Location**: `api/middleware.py`

**Features**:
- Sliding window rate limiting per client IP
- Configurable requests per minute and burst limits
- Health endpoint exemption (`/healthz`, `/readyz`, `/metrics`)
- Rate limit headers in responses:
  - `X-RateLimit-Limit`
  - `X-RateLimit-Remaining`
  - `X-RateLimit-Reset`
- Proper 429 error responses with retry-after headers

### 4. Request Validation Middleware (`RequestValidationMiddleware`)

**Location**: `api/middleware.py`

**Features**:
- Content-length validation (configurable max size)
- Content-type validation for POST/PUT/PATCH requests
- Early request rejection before reaching route handlers
- Proper HTTP status codes (413, 415)

### 5. Enhanced Exception Handling

**Location**: `api/main.py`

**Features**:
- Custom API exception handler for structured error responses
- HTTP exception handler with correlation ID tracking
- General exception handler with proper logging
- Consistent error response format using Pydantic models

## Configuration Integration

### Middleware Configuration

The middleware integrates with the existing configuration system:

```python
# Rate limiting configuration
config.security.rate_limit_per_minute = 60

# CORS configuration  
config.server.cors_origins = ["http://localhost:3000", "https://example.com"]
```

### Runtime Configuration Updates

The middleware configuration is updated during application startup in the lifespan manager:

```python
# Update CORS origins from config
cors_middleware.kwargs["allow_origins"] = config.server.cors_origins

# Update rate limits from config
rate_limit_middleware.kwargs["requests_per_minute"] = config.security.rate_limit_per_minute
```

## Testing

### Unit Tests (`tests/test_middleware.py`)

- **SecurityHeadersMiddleware**: Tests security header addition and cache control
- **RequestLoggingMiddleware**: Tests correlation ID generation, structured logging, and header redaction
- **RateLimitMiddleware**: Tests rate limiting, health endpoint exemption, and proper error responses
- **RequestValidationMiddleware**: Tests content validation and proper error handling
- **Integration**: Tests middleware interaction and proper header propagation

### Integration Tests (`tests/test_security_integration.py`)

- Security header presence across all endpoints
- Correlation ID tracking across requests
- Rate limiting integration with application
- Request validation integration
- Error handling with correlation IDs
- Health endpoint rate limit bypass
- Structured logging integration
- Custom configuration application

### Test Coverage

- **Total Tests**: 23 middleware-specific tests
- **Coverage**: All middleware components and integration scenarios
- **Status**: All tests passing ✅

## Security Features

### Headers Applied

```http
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
X-Permitted-Cross-Domain-Policies: none
Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'
X-Correlation-ID: uuid-string
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1756681955
```

### Rate Limiting

- Default: 60 requests per minute per IP
- Burst limit: 10 requests per 10 seconds
- Health endpoints exempted
- Configurable via `config.security.rate_limit_per_minute`

### Request Validation

- Max content length: 10MB (configurable)
- Content-type validation for mutation requests
- Early rejection before route processing

### Logging Security

- Sensitive headers redacted: `authorization`, `x-api-key`, `cookie`, `set-cookie`
- Correlation IDs for request tracing
- Structured JSON format for log aggregation
- No credential leakage in logs

## Performance Considerations

### Middleware Order

The middleware is applied in optimal order for performance:

1. **RequestValidationMiddleware** - Early rejection of invalid requests
2. **RateLimitMiddleware** - Rate limiting before processing
3. **SecurityHeadersMiddleware** - Header addition
4. **RequestLoggingMiddleware** - Logging (after security)
5. **CORSMiddleware** - CORS headers (last to apply to all responses)

### Memory Usage

- Rate limiting uses in-memory storage with automatic cleanup
- Log data is not stored in memory (streamed to logging system)
- Correlation IDs are lightweight UUIDs

### Scalability

- Rate limiting is per-process (Redis-based solution recommended for multi-instance deployments)
- Middleware is stateless except for rate limiting counters
- Efficient header processing with minimal overhead

## Future Enhancements

### Recommended Improvements

1. **Redis-based Rate Limiting**: For multi-instance deployments
2. **Authentication Middleware**: Bearer token validation
3. **Request Size Streaming**: For large file uploads
4. **Metrics Collection**: Prometheus-compatible metrics
5. **IP Allowlisting**: Network-level security controls

### Configuration Extensions

```python
class SecurityConfig(BaseModel):
    auth_token: Optional[str] = None
    rate_limit_per_minute: int = 60
    max_request_size: int = 10 * 1024 * 1024
    allowed_ips: List[str] = []
    enable_request_logging: bool = True
    log_request_body: bool = False
```

## Compliance

### Internal Network Security

The implementation follows best practices for internal microservices:

- Appropriate security headers for internal deployment
- CORS configuration for known internal origins
- Rate limiting to prevent abuse
- Comprehensive logging for audit trails
- Request validation to prevent malformed requests

### Standards Compliance

- **OWASP**: Security headers follow OWASP recommendations
- **RFC 7234**: Cache control headers for API responses
- **RFC 6585**: Proper 429 status code usage for rate limiting
- **RFC 7231**: Correct HTTP status codes for validation errors

## Summary

The security middleware implementation successfully addresses all requirements from task 4:

✅ **Security Headers**: Comprehensive security headers for internal network deployment  
✅ **CORS Configuration**: Configurable CORS origins for internal services  
✅ **Request Validation**: Content-length and content-type validation  
✅ **Error Handling**: Structured error responses with proper HTTP status codes  
✅ **Request Logging**: Structured logging with correlation IDs for tracing  
✅ **Rate Limiting**: Configurable rate limiting with health endpoint exemption  
✅ **Integration**: Seamless integration with existing configuration system  
✅ **Testing**: Comprehensive unit and integration test coverage  

The middleware provides a solid security foundation for the TTV Pipeline API server while maintaining performance and configurability for internal network deployment.