# TTV Pipeline API Deployment Guide

This comprehensive guide covers deploying the TTV Pipeline API across different environments with proper configuration, monitoring, and operational procedures.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Local Development](#local-development)
5. [Production Deployment](#production-deployment)
6. [Configuration Management](#configuration-management)
7. [Monitoring and Health Checks](#monitoring-and-health-checks)
8. [Scaling and Performance](#scaling-and-performance)
9. [Security Considerations](#security-considerations)
10. [Troubleshooting](#troubleshooting)
11. [Maintenance and Updates](#maintenance-and-updates)

## Overview

The TTV Pipeline API is deployed using Docker Compose with the following architecture:

```
Client → Nginx Proxy (HTTPS) → FastAPI Server → Redis Queue → Workers → GCS
```

### Key Components

- **Nginx Proxy**: HTTP/2 edge proxy with TLS termination
- **FastAPI Server**: API application with Hypercorn ASGI server
- **Redis**: Job queue and metadata storage
- **Workers**: Video generation job processors
- **GPU Workers**: Optional GPU-accelerated processors

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB available space
- Network: Stable internet connection

**Recommended for Production:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 100GB+ SSD
- GPU: NVIDIA GPU with 8GB+ VRAM (optional)

### Software Dependencies

```bash
# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose V2
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

### Additional Tools

```bash
# Install useful tools
sudo apt-get install curl jq redis-tools

# For certificate management
sudo apt-get install certbot
```

## Environment Setup

### 1. Clone and Setup Repository

```bash
git clone <repository-url>
cd ttv-pipeline-api
```

### 2. Create Required Directories

```bash
mkdir -p credentials certs backup logs
chmod 700 credentials
```

### 3. Configure Environment

Choose your deployment environment and copy the appropriate configuration:

```bash
# For development
cp .env.dev .env

# For production
cp .env.prod .env
```

Edit the `.env` file with your specific configuration.

## Local Development

### Quick Start

```bash
# Setup development environment
make setup-dev

# Start development stack
make dev

# Or using the deploy script
./scripts/deploy.sh dev up
```

### Development Services

The development environment includes:

- **API Server**: http://localhost:8000 (with hot reload)
- **Redis Commander**: http://localhost:8081 (queue debugging)
- **File Server**: http://localhost:8082 (local artifacts)
- **MinIO**: http://localhost:9001 (S3-compatible storage)

### Development Workflow

1. **Start services:**
   ```bash
   make dev
   ```

2. **View logs:**
   ```bash
   make dev-logs
   ```

3. **Run tests:**
   ```bash
   make dev-test
   ```

4. **Access container shell:**
   ```bash
   make dev-shell
   ```

5. **Stop services:**
   ```bash
   make down ENV=dev
   ```

### Hot Reload

The development environment supports hot reload for:
- API server code changes
- Worker code changes
- Configuration updates

### Debug Services

Enable debug services for additional development tools:

```bash
# Start with debug services
make debug ENV=dev

# Or using profiles
docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile debug up -d
```

## Production Deployment

### 1. Server Preparation

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Configure firewall
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 443/udp  # For QUIC/HTTP3
sudo ufw enable

# Optimize system for production
echo 'net.core.somaxconn = 65535' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_max_syn_backlog = 65535' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### 2. SSL Certificate Setup

#### Option A: Let's Encrypt (Recommended)

```bash
# Install certbot
sudo apt-get install certbot

# Obtain certificate
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem certs/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem certs/key.pem
sudo chown $USER:$USER certs/*.pem
```

#### Option B: Custom Certificates

```bash
# Copy your certificates
cp your-cert.pem certs/cert.pem
cp your-key.pem certs/key.pem
chmod 600 certs/key.pem
```

### 3. Google Cloud Storage Setup

```bash
# Create service account key
# Download from Google Cloud Console and save as:
cp your-gcs-key.json credentials/gcs-key.json
chmod 600 credentials/gcs-key.json
```

### 4. Production Configuration

Edit `.env` with production settings:

```bash
# Environment
ENVIRONMENT=production

# API Configuration
API_WORKERS=4
LOG_LEVEL=INFO

# Worker Configuration
WORKER_REPLICAS=4
WORKER_CONCURRENCY=2

# Security
REDIS_PASSWORD=your-secure-redis-password

# GCS Configuration
GCS_BUCKET=your-production-bucket
```

### 5. Deploy Production Stack

```bash
# Setup production environment
make setup-prod

# Build and deploy
make prod-deploy

# Or using the deploy script
./scripts/deploy.sh prod up
```

### 6. Verify Deployment

```bash
# Check service status
make status ENV=prod

# Run health checks
make health ENV=prod

# Test API endpoints
curl -f https://your-domain.com/healthz
curl -f https://your-domain.com/readyz
```

## Configuration Management

### Environment Variables

Key configuration options:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Deployment environment | `production` | Yes |
| `API_WORKERS` | Number of API workers | `2` | No |
| `WORKER_REPLICAS` | Number of job workers | `2` | No |
| `GCS_BUCKET` | GCS bucket name | - | Yes |
| `REDIS_PASSWORD` | Redis password | - | Recommended |

### Configuration Files

- **Pipeline Config**: `pipeline_config.yaml` - Core pipeline settings
- **API Config**: `api_config.yaml` - API-specific settings
- **Redis Config**: `config/redis.conf` - Redis optimization
- **Nginx Config**: `config/nginx.conf` - Proxy configuration

### Secrets Management

For production deployments, consider using:

- **Docker Secrets**: For sensitive configuration
- **External Secret Stores**: HashiCorp Vault, AWS Secrets Manager
- **Environment Files**: Properly secured `.env` files

Example with Docker secrets:

```yaml
services:
  api:
    secrets:
      - gcs_credentials
      - redis_password
    environment:
      - GCS_CREDENTIALS_FILE=/run/secrets/gcs_credentials
      - REDIS_PASSWORD_FILE=/run/secrets/redis_password

secrets:
  gcs_credentials:
    file: ./credentials/gcs-key.json
  redis_password:
    file: ./credentials/redis-password.txt
```

## Monitoring and Health Checks

### Health Endpoints

The API provides several health check endpoints:

- **`/healthz`**: Basic liveness check
- **`/readyz`**: Readiness check (includes dependencies)
- **`/metrics`**: Prometheus-compatible metrics

### Automated Health Checks

```bash
# Run comprehensive health checks
./scripts/health-check.sh production

# Check specific components
curl -f http://localhost:8000/healthz
redis-cli -h localhost -p 6379 ping
```

### Monitoring Setup

#### Prometheus Configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'ttv-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

#### Grafana Dashboard

Key metrics to monitor:

- **Request Rate**: `rate(http_requests_total[5m])`
- **Response Time**: `histogram_quantile(0.95, http_request_duration_seconds_bucket)`
- **Error Rate**: `rate(http_requests_total{status=~"5.."}[5m])`
- **Queue Depth**: `redis_queue_depth`
- **Worker Status**: `worker_active_jobs`

### Log Management

#### Structured Logging

All services use structured JSON logging:

```json
{
  "timestamp": "2025-08-31T12:34:56Z",
  "level": "INFO",
  "component": "api",
  "job_id": "job_123",
  "message": "Job processing started",
  "duration_ms": 150
}
```

#### Log Aggregation

For production, consider:

- **ELK Stack**: Elasticsearch, Logstash, Kibana
- **Loki**: Grafana Loki with Promtail
- **Cloud Logging**: Google Cloud Logging, AWS CloudWatch

Example Loki configuration:

```yaml
# docker-compose.logging.yml
services:
  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    volumes:
      - ./config/loki.yml:/etc/loki/local-config.yaml

  promtail:
    image: grafana/promtail:latest
    volumes:
      - /var/log:/var/log:ro
      - ./config/promtail.yml:/etc/promtail/config.yml
```

## Scaling and Performance

### Horizontal Scaling

#### Scale Workers

```bash
# Scale to 8 workers
make scale-workers REPLICAS=8

# Or using environment variable
WORKER_REPLICAS=8 docker compose up -d
```

#### Auto-scaling with Docker Swarm

```yaml
# docker-stack.yml
services:
  worker:
    deploy:
      replicas: 4
      update_config:
        parallelism: 2
        delay: 10s
      restart_policy:
        condition: on-failure
      placement:
        constraints:
          - node.role == worker
```

### Performance Optimization

#### API Server Tuning

```yaml
# Hypercorn configuration
bind: "0.0.0.0:8000"
workers: 4
worker_class: "trio"
max_requests: 1000
max_requests_jitter: 100
preload_app: true
```

#### Redis Optimization

```conf
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

#### HTTP/3 Optimization

```nginx
# nginx.conf
# HTTP/2 configuration (HTTP/3 requires nginx-quic build)
listen 443 ssl http2;
ssl_protocols TLSv1.2 TLSv1.3;
```

### Load Testing

```bash
# Install k6
sudo apt-get install k6

# Run load test
k6 run --vus 10 --duration 30s scripts/load-test.js
```

Example load test script:

```javascript
// scripts/load-test.js
import http from 'k6/http';
import { check } from 'k6';

export default function () {
  const payload = JSON.stringify({
    prompt: 'A test video generation prompt'
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  const response = http.post('http://localhost:8000/v1/jobs', payload, params);
  
  check(response, {
    'status is 202': (r) => r.status === 202,
    'has job id': (r) => r.json('id') !== undefined,
  });
}
```

## Security Considerations

### Network Security

```bash
# Configure firewall
sudo ufw deny 6379  # Block Redis port
sudo ufw deny 8000  # Block direct API access
sudo ufw allow 443   # Allow HTTPS
sudo ufw allow 443/udp  # Allow QUIC
```

### Container Security

```yaml
# Security-hardened container configuration
services:
  api:
    user: "1000:1000"
    read_only: true
    tmpfs:
      - /tmp
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
```

### Secrets Management

```bash
# Secure credential files
chmod 600 credentials/*
chown root:root credentials/
```

### TLS Configuration

```nginx
# nginx.conf - Security headers
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
add_header X-Content-Type-Options nosniff always;
add_header X-Frame-Options DENY always;
add_header X-XSS-Protection "1; mode=block" always;
```

## Troubleshooting

### Common Issues

#### 1. Port Conflicts

```bash
# Check port usage
sudo netstat -tulpn | grep :8000

# Use different ports
API_PORT=8001 docker compose up -d
```

#### 2. Certificate Issues

```bash
# Verify certificates
openssl x509 -in certs/cert.pem -text -noout
openssl rsa -in certs/key.pem -check

# Test TLS connection
openssl s_client -connect localhost:443 -servername your-domain.com
```

#### 3. Redis Connection Issues

```bash
# Test Redis connectivity
redis-cli -h localhost -p 6379 ping

# Check Redis logs
docker compose logs redis
```

#### 4. Worker Issues

```bash
# Check worker logs
docker compose logs worker

# Monitor job queue
redis-cli -h localhost -p 6379 llen rq:queue:default
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Start with debug services
./scripts/deploy.sh dev debug

# Access Redis Commander
open http://localhost:8081

# View detailed logs
docker compose logs -f --tail=100
```

### Performance Issues

#### Monitor Resource Usage

```bash
# Container resource usage
docker stats

# System resource usage
htop
iotop
```

#### Profile API Performance

```bash
# Enable profiling
PROFILING_ENABLED=true docker compose up -d

# Access profiling data
curl http://localhost:8000/debug/pprof/
```

## Maintenance and Updates

### Regular Maintenance

#### Daily Tasks

```bash
# Check service health
./scripts/health-check.sh production

# Monitor logs for errors
docker compose logs --since 24h | grep ERROR

# Check disk space
df -h
```

#### Weekly Tasks

```bash
# Update images
docker compose pull

# Clean up unused resources
docker system prune -f

# Backup configuration and data
make backup ENV=prod
```

#### Monthly Tasks

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Rotate logs
docker compose exec api logrotate /etc/logrotate.conf

# Review and update certificates
certbot renew --dry-run
```

### Updates and Rollbacks

#### Rolling Updates

```bash
# Update with zero downtime
docker compose pull
docker compose up -d --force-recreate --no-deps api
docker compose up -d --force-recreate --no-deps worker
```

#### Rollback Procedure

```bash
# Tag current version
docker tag ttv-api:latest ttv-api:backup-$(date +%Y%m%d)

# Rollback to previous version
docker compose down
git checkout previous-stable-tag
docker compose up -d
```

### Backup and Recovery

#### Automated Backups

```bash
# Create backup script
cat > scripts/backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="backup/$DATE"

mkdir -p "$BACKUP_DIR"

# Backup Redis data
docker compose exec redis redis-cli BGSAVE
docker cp $(docker compose ps -q redis):/data/dump.rdb "$BACKUP_DIR/"

# Backup configuration
tar -czf "$BACKUP_DIR/config.tar.gz" .env* docker-compose*.yml config/ credentials/

# Upload to cloud storage (optional)
gsutil cp -r "$BACKUP_DIR" gs://your-backup-bucket/
EOF

chmod +x scripts/backup.sh
```

#### Recovery Procedure

```bash
# Stop services
docker compose down

# Restore Redis data
docker cp backup/latest/dump.rdb $(docker compose ps -q redis):/data/

# Restore configuration
tar -xzf backup/latest/config.tar.gz

# Restart services
docker compose up -d
```

### Monitoring Alerts

#### Alertmanager Configuration

```yaml
# alertmanager.yml
groups:
  - name: ttv-api
    rules:
      - alert: APIDown
        expr: up{job="ttv-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "TTV API is down"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"

      - alert: QueueBacklog
        expr: redis_queue_depth > 100
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Job queue backlog detected"
```

This comprehensive deployment guide provides everything needed to successfully deploy and operate the TTV Pipeline API in various environments with proper monitoring, security, and maintenance procedures.