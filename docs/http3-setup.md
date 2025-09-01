# HTTP/3 Setup Guide

This document describes the HTTP/3 setup for the TTV Pipeline API server using Hypercorn ASGI server with QUIC support.

## Overview

The API server supports HTTP/3 over QUIC with fallbacks to HTTP/2 and HTTP/1.1. The architecture includes:

- **Angie Edge Proxy**: HTTP/3 edge termination with QUIC parameter tuning
- **Hypercorn ASGI Server**: HTTP/3 over QUIC upstream support
- **FastAPI Application**: Enhanced with HTTP/3 middleware
- **Alt-Svc Headers**: Automatic protocol negotiation

### Architecture Flow

```
Client → Angie (HTTP/3 edge) → Hypercorn (HTTP/3 upstream) → FastAPI
```

The Angie edge proxy provides:
- HTTP/3 edge termination with optimized QUIC parameters
- HTTP/3 upstream proxy to Hypercorn application server
- TLS certificate management and OCSP stapling
- Connection migration support with QUIC host keys
- Rate limiting and security headers

## Quick Start

### Development (HTTP/1.1 and HTTP/2 only)

```bash
# Start development server
python -m api.server --environment development

# Or using the module
python -m api
```

### Production (with HTTP/3)

```bash
# Generate certificates (for testing)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Start production server with HTTP/3
python -m api.server \
    --environment production \
    --certfile cert.pem \
    --keyfile key.pem
```

## Configuration

### API Configuration

Create `api_config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  quic_port: 8443
  workers: 4
  cors_origins:
    - "https://example.com"

redis:
  host: "localhost"
  port: 6379

gcs:
  bucket: "ttv-api-artifacts"
  credentials_path: "./credentials/gcs-key.json"

security:
  rate_limit_per_minute: 60
```

### Environment Variables

```bash
# Server configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export API_QUIC_PORT=8443

# Redis configuration
export REDIS_HOST=localhost
export REDIS_PORT=6379

# GCS configuration
export GCS_BUCKET=ttv-api-artifacts
export GCS_CREDENTIALS_PATH=/path/to/credentials.json
```

## Docker Deployment

### Using Docker Compose with Angie Edge Proxy

```bash
# Setup certificates first
./scripts/setup_http3_certs.sh

# Validate configuration
./scripts/validate_angie_config.sh

# Start with HTTP/3 support
docker-compose -f docker-compose.http3.yml up -d

# Check logs
docker-compose -f docker-compose.http3.yml logs -f angie api
```

### Certificate Management

The setup script automatically creates the certificate structure:

```
certs/
├── cert.pem          # SSL certificate (self-signed for development)
├── key.pem           # Private key
└── quic_host.key     # QUIC host key for connection migration

credentials/
└── gcs-key.json      # GCS service account key (manual)
```

For production, replace the self-signed certificates with proper CA-signed certificates.

## Protocol Negotiation

The server supports automatic protocol negotiation:

1. **HTTP/3 over QUIC**: Primary protocol for HTTPS connections
2. **HTTP/2**: Fallback for clients that don't support HTTP/3
3. **HTTP/1.1**: Final fallback for legacy clients

### Alt-Svc Headers

The server automatically adds Alt-Svc headers to advertise HTTP/3 availability:

```
Alt-Svc: h3=":8443"; ma=86400, h2=":443"; ma=86400
```

## Testing HTTP/3

### Using the Test Script

```bash
# Test protocol negotiation through Angie proxy
python scripts/test_http3.py --test-protocols --https-url https://localhost:443

# Test all endpoints through proxy
python scripts/test_http3.py --all-tests --url https://localhost:443

# Test direct to API server (bypass proxy)
python scripts/test_http3.py --url https://localhost:8443 --test-endpoints
```

### Validate Angie Configuration

```bash
# Full validation
./scripts/validate_angie_config.sh

# Test connectivity only
./scripts/validate_angie_config.sh --connectivity

# Generate configuration report
./scripts/validate_angie_config.sh --report
```

### Using curl (HTTP/3)

```bash
# Test HTTP/3 (requires curl with HTTP/3 support)
curl --http3 https://localhost:8443/healthz

# Test with Alt-Svc discovery
curl -v https://localhost:8443/ | grep -i alt-svc
```

### Using httpx (Python)

```python
import httpx

async def test_http3():
    async with httpx.AsyncClient(http2=True) as client:
        response = await client.get("https://localhost:8443/healthz")
        print(f"Protocol: {response.http_version}")
        print(f"Alt-Svc: {response.headers.get('alt-svc')}")
```

## Performance Considerations

### HTTP/3 Benefits

- **Reduced Connection Setup**: QUIC's 0-RTT connection establishment
- **Multiplexing**: No head-of-line blocking
- **Connection Migration**: Survives network changes
- **Built-in Encryption**: TLS 1.3 by default

### Configuration Tuning

```python
# Hypercorn configuration for high performance
config.quic_max_concurrent_connections = 1000
config.h3_max_concurrent_streams = 100
config.keep_alive_timeout = 75
config.max_requests = 10000
```

## Monitoring

### Metrics

The server exposes HTTP/3 specific metrics:

- `http3_connections_total`: Total HTTP/3 connections
- `http3_requests_total`: Total HTTP/3 requests
- `protocol_negotiation_duration`: Time to negotiate protocol

### Logging

HTTP/3 requests are logged with protocol information:

```json
{
  "event": "request_complete",
  "protocol": "h3",
  "connection_id": "abc123",
  "stream_id": 4,
  "duration_ms": 150
}
```

## Troubleshooting

### Common Issues

1. **Certificate Errors**
   ```bash
   # Verify certificate
   openssl x509 -in cert.pem -text -noout
   ```

2. **Port Conflicts**
   ```bash
   # Check if QUIC port is available
   netstat -an | grep :8443
   ```

3. **Firewall Issues**
   ```bash
   # Allow UDP traffic for QUIC
   sudo ufw allow 8443/udp
   ```

### Debug Mode

Enable debug logging:

```bash
python -m api.server \
    --environment development \
    --reload \
    --log-level debug
```

### Client Compatibility

Not all clients support HTTP/3 yet. The server gracefully falls back to HTTP/2 or HTTP/1.1.

## Security Considerations

### TLS Configuration

- **Minimum TLS 1.2**: For compatibility
- **Prefer TLS 1.3**: For HTTP/3 performance
- **ALPN Protocols**: `["h3", "h2", "http/1.1"]`

### QUIC Security

- **Connection IDs**: Randomized for privacy
- **Stateless Reset**: Prevents connection tracking
- **Path Validation**: Prevents address spoofing

## References

- [HTTP/3 Specification (RFC 9114)](https://tools.ietf.org/html/rfc9114)
- [QUIC Specification (RFC 9000)](https://tools.ietf.org/html/rfc9000)
- [Hypercorn Documentation](https://hypercorn.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)