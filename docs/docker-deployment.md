# Docker Deployment Guide

This guide covers deploying the TTV Pipeline API using Docker and Docker Compose with HTTP/3 support.

## Overview

The Docker deployment includes:
- **API Server**: FastAPI application with HTTP/3 support via Hypercorn
- **Nginx Proxy**: Edge proxy with HTTP/2 termination and upstream forwarding
- **Workers**: RQ workers for video generation job processing
- **Redis**: Job queue and metadata storage
- **GPU Workers**: Optional GPU-accelerated workers

## Quick Start

### Development Environment

1. **Copy environment configuration:**
   ```bash
   cp .env.dev .env
   ```

2. **Start development stack:**
   ```bash
   ./scripts/deploy.sh development up
   ```

3. **Access the API:**
   - API: http://localhost:8000
   - Redis Commander: http://localhost:8081 (for debugging)

### Production Environment

1. **Copy and configure environment:**
   ```bash
   cp .env.prod .env
   # Edit .env with your production settings
   ```

2. **Set up certificates:**
   ```bash
   mkdir -p certs
   # Copy your TLS certificates to ./certs/
   ```

3. **Start production stack:**
   ```bash
   ./scripts/deploy.sh production up
   ```

4. **Access the API:**
   - HTTPS: https://your-domain.com
   - HTTP/3: Available automatically with proper client support

## Architecture

```
Client (HTTPS) → Nginx Proxy → API Server (Hypercorn)
                                     ↓
                               Redis Queue
                                     ↓
                               RQ Workers → GCS
```

## Environment Configuration

### Environment Files

- `.env.dev` - Development configuration
- `.env.prod` - Production configuration  
- `.env.example` - Template with all available options

### Key Configuration Options

```bash
# Environment type
ENVIRONMENT=production

# API Server
API_HOST=0.0.0.0
API_PORT=8000
API_QUIC_PORT=8443
API_WORKERS=4

# Workers
WORKER_REPLICAS=4
WORKER_CONCURRENCY=2
GPU_WORKER_REPLICAS=1

# Google Cloud Storage
GCS_BUCKET=ttv-api-artifacts
GCS_PREFIX=ttv-api

# File Paths
CREDENTIALS_PATH=./credentials
PIPELINE_CONFIG_PATH=./pipeline_config.yaml
API_CONFIG_PATH=./api_config.yaml
CERTS_PATH=./certs
```

## Docker Images

### Multi-stage Build Targets

- **development**: Development image with hot reload and dev dependencies
- **api**: Production API server image
- **worker**: Production worker image for CPU processing
- **gpu-worker**: GPU-enabled worker image

### Building Images

```bash
# Build all images
docker-compose build

# Build specific target
docker-compose build api
docker-compose build worker
```

## Deployment Commands

### Using the Deploy Script

```bash
# Development
./scripts/deploy.sh dev up
./scripts/deploy.sh dev logs
./scripts/deploy.sh dev down

# Production
./scripts/deploy.sh prod up
./scripts/deploy.sh prod status
./scripts/deploy.sh prod restart

# With GPU support
./scripts/deploy.sh prod gpu

# With debug services
./scripts/deploy.sh dev debug
```

### Manual Docker Compose

```bash
# Development
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# With profiles
docker-compose --profile gpu up -d
docker-compose --profile debug up -d
```

## Service Configuration

### API Server

The API server runs on Hypercorn with HTTP/3 support:

```yaml
api:
  build:
    target: api
  ports:
    - "8000:8000/tcp"   # HTTP
    - "8443:8443/udp"   # QUIC/HTTP3
  environment:
    - API_WORKERS=4
    - ENVIRONMENT=production
```

### Workers

RQ workers process video generation jobs:

```yaml
worker:
  build:
    target: worker
  deploy:
    replicas: 4
  environment:
    - WORKER_CONCURRENCY=2
```

### GPU Workers

Optional GPU-accelerated workers:

```yaml
gpu-worker:
  build:
    target: gpu-worker
  runtime: nvidia
  environment:
    - NVIDIA_VISIBLE_DEVICES=all
  profiles:
    - gpu
```

### Redis

Job queue and metadata storage:

```yaml
redis:
  image: redis:7-alpine
  volumes:
    - redis-data:/data
    - ./config/redis.conf:/usr/local/etc/redis/redis.conf:ro
  command: redis-server /usr/local/etc/redis/redis.conf
```

## Health Monitoring

### Health Check Script

```bash
# Check all services
./scripts/health-check.sh production

# Check development environment
./scripts/health-check.sh development
```

### Health Endpoints

- `/healthz` - Basic liveness check
- `/readyz` - Readiness check (includes Redis connectivity)
- `/metrics` - Prometheus-compatible metrics

### Container Health Checks

All services include Docker health checks:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1
```

## Security Considerations

### TLS Certificates

For production deployment with HTTP/3:

1. **Obtain certificates:**
   ```bash
   # Using Let's Encrypt
   certbot certonly --standalone -d your-domain.com
   
   # Copy to certs directory
   cp /etc/letsencrypt/live/your-domain.com/fullchain.pem certs/cert.pem
   cp /etc/letsencrypt/live/your-domain.com/privkey.pem certs/key.pem
   ```

2. **Update Nginx configuration:**
   ```nginx
   ssl_certificate /etc/ssl/certs/cert.pem;
   ssl_certificate_key /etc/ssl/certs/key.pem;
   ```

### Secrets Management

- Store GCS credentials in `./credentials/gcs-key.json`
- Set Redis password in production: `REDIS_PASSWORD=your-secure-password`
- Use Docker secrets for sensitive data in production

### Network Security

- Services communicate via internal Docker network
- Only necessary ports are exposed
- CORS configured for internal network access

## Scaling and Performance

### Horizontal Scaling

```bash
# Scale workers
docker-compose up -d --scale worker=8

# Scale with environment variables
WORKER_REPLICAS=8 docker-compose up -d
```

### Resource Limits

Production configuration includes resource limits:

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
    reservations:
      cpus: '2.0'
      memory: 4G
```

### Performance Tuning

- **Redis**: Configured for job queue workloads
- **Workers**: Adjustable concurrency per worker
- **API**: Multiple Hypercorn workers with HTTP/3 optimization

## Troubleshooting

### Common Issues

1. **Port conflicts:**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8000
   
   # Use different ports
   API_PORT=8001 docker-compose up -d
   ```

2. **Certificate issues:**
   ```bash
   # Verify certificate files
   ls -la certs/
   openssl x509 -in certs/cert.pem -text -noout
   ```

3. **Redis connection issues:**
   ```bash
   # Test Redis connectivity
   redis-cli -h localhost -p 6379 ping
   
   # Check Redis logs
   docker-compose logs redis
   ```

### Debugging

1. **Enable debug services:**
   ```bash
   ./scripts/deploy.sh dev debug
   ```

2. **View logs:**
   ```bash
   # All services
   docker-compose logs -f
   
   # Specific service
   docker-compose logs -f api
   docker-compose logs -f worker
   ```

3. **Access containers:**
   ```bash
   # API server
   docker-compose exec api bash
   
   # Worker
   docker-compose exec worker bash
   ```

## Monitoring and Observability

### Metrics Collection

- Prometheus-compatible metrics at `/metrics`
- Request latency and throughput tracking
- Job queue depth and processing time
- HTTP/3 connection statistics

### Log Aggregation

Production logging configuration:

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### External Monitoring

Integrate with monitoring systems:

- **Prometheus**: Scrape `/metrics` endpoint
- **Grafana**: Visualize metrics and logs
- **AlertManager**: Configure alerts for failures

## Backup and Recovery

### Redis Data

```bash
# Backup Redis data
docker-compose exec redis redis-cli BGSAVE
docker cp $(docker-compose ps -q redis):/data/dump.rdb ./backup/

# Restore Redis data
docker cp ./backup/dump.rdb $(docker-compose ps -q redis):/data/
docker-compose restart redis
```

### Configuration Backup

```bash
# Backup configuration
tar -czf backup/config-$(date +%Y%m%d).tar.gz \
    .env* docker-compose*.yml config/ credentials/
```

## Maintenance

### Updates

```bash
# Update images
docker-compose pull
docker-compose build --no-cache

# Rolling update
docker-compose up -d --force-recreate
```

### Cleanup

```bash
# Clean up unused resources
./scripts/deploy.sh prod clean

# Remove all data (destructive)
docker-compose down -v --remove-orphans
docker system prune -a
```