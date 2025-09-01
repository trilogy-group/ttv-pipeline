# TTV Pipeline API Documentation

Welcome to the TTV Pipeline API documentation. This directory contains comprehensive guides for deploying, operating, and monitoring the TTV Pipeline API service.

## Documentation Overview

### 📚 Core Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| [Deployment Guide](deployment-guide.md) | Complete deployment instructions for all environments | DevOps, Developers |
| [Operations & Monitoring](operations-monitoring.md) | Operational procedures and monitoring setup | SRE, Operations |
| [Docker Deployment](docker-deployment.md) | Docker-specific deployment details | DevOps |
| [HTTP/3 Setup](http3-setup.md) | HTTP/3 configuration and optimization | Network Engineers |

### 🚀 Quick Start Guides

#### Local Development
```bash
# Setup development environment
./scripts/setup-environment.sh development

# Start development stack
make dev

# Access API
curl http://localhost:8000/healthz
```

#### Production Deployment
```bash
# Setup production environment
./scripts/setup-environment.sh production

# Configure certificates and credentials
# (See deployment guide for details)

# Deploy production stack
make prod-deploy

# Verify deployment
make health ENV=prod
```

### 🛠️ Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/setup-environment.sh` | Environment setup and validation | `./scripts/setup-environment.sh [dev\|prod]` |
| `scripts/deploy.sh` | Deployment automation | `./scripts/deploy.sh [dev\|prod] [up\|down\|restart]` |
| `scripts/health-check.sh` | Health monitoring | `./scripts/health-check.sh [environment]` |
| `scripts/setup-monitoring.sh` | Monitoring stack setup | `./scripts/setup-monitoring.sh [basic\|prometheus\|full]` |

### 📊 Monitoring & Observability

The API provides comprehensive monitoring capabilities:

- **Health Endpoints**: `/healthz`, `/readyz`, `/metrics`
- **Structured Logging**: JSON logs with correlation IDs
- **Metrics Collection**: Prometheus-compatible metrics
- **Alerting**: Configurable alerts for critical conditions
- **Dashboards**: Pre-built Grafana dashboards

#### Quick Monitoring Setup
```bash
# Setup basic monitoring
./scripts/setup-monitoring.sh basic

# Setup Prometheus + Grafana
./scripts/setup-monitoring.sh prometheus

# Setup full monitoring stack
./scripts/setup-monitoring.sh full
```

### 🏗️ Architecture Overview

```
Client Applications
        ↓
   Nginx Proxy (HTTPS)
        ↓
   FastAPI Server
        ↓
    Redis Queue
        ↓
   RQ Workers → Google Cloud Storage
```

#### Key Components

- **Nginx Proxy**: HTTP/2 edge proxy with TLS termination
- **FastAPI Server**: API application with Hypercorn ASGI server
- **Redis**: Job queue and metadata storage
- **Workers**: Video generation job processors
- **GCS**: Artifact storage and delivery

### 🔧 Configuration Management

#### Environment Files

| File | Purpose | Environment |
|------|---------|-------------|
| `.env.dev` | Development configuration | Local development |
| `.env.prod` | Production configuration | Production deployment |
| `.env.example` | Configuration template | Reference |

#### Key Configuration Areas

- **API Server**: Workers, ports, logging
- **Workers**: Scaling, concurrency, GPU support
- **Redis**: Connection, persistence, security
- **GCS**: Bucket, credentials, paths
- **Security**: TLS certificates, passwords, CORS

### 🚨 Troubleshooting

#### Common Issues

1. **Port Conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8000
   
   # Use different ports
   API_PORT=8001 make dev
   ```

2. **Certificate Issues**
   ```bash
   # Verify certificates
   openssl x509 -in certs/cert.pem -text -noout
   
   # Generate self-signed for testing
   openssl req -x509 -newkey rsa:4096 -keyout certs/key.pem -out certs/cert.pem -days 365 -nodes
   ```

3. **Redis Connection Issues**
   ```bash
   # Test Redis connectivity
   redis-cli -h localhost -p 6379 ping
   
   # Check Redis logs
   docker compose logs redis
   ```

4. **Worker Issues**
   ```bash
   # Check worker status
   docker compose ps worker
   
   # Monitor job queue
   redis-cli llen rq:queue:default
   ```

#### Debug Mode

Enable debug services for detailed troubleshooting:

```bash
# Start with debug services
make debug ENV=dev

# Access Redis Commander
open http://localhost:8081

# View detailed logs
docker compose logs -f --tail=100
```

### 📈 Performance & Scaling

#### Scaling Guidelines

| Component | Scaling Trigger | Action |
|-----------|----------------|--------|
| API Server | CPU > 80% for 15min | Increase `API_WORKERS` |
| Workers | Queue depth > 50 | Increase `WORKER_REPLICAS` |
| Redis | Memory > 80% | Increase Redis memory limit |
| Storage | Disk > 85% | Add storage or cleanup |

#### Performance Optimization

```bash
# Scale workers
make scale-workers REPLICAS=8

# Monitor performance
docker stats

# Check queue depth
redis-cli llen rq:queue:default
```

### 🔒 Security Considerations

#### Production Security Checklist

- [ ] TLS certificates configured and valid
- [ ] Redis password set
- [ ] GCS credentials secured (600 permissions)
- [ ] Firewall configured (block Redis port)
- [ ] Security headers enabled in Nginx
- [ ] Log sanitization enabled
- [ ] Regular security updates scheduled

#### Security Best Practices

1. **Credential Management**
   ```bash
   # Secure credential files
   chmod 600 credentials/*
   chown root:root credentials/
   ```

2. **Network Security**
   ```bash
   # Configure firewall
   sudo ufw deny 6379  # Block Redis
   sudo ufw allow 443  # Allow HTTPS
   sudo ufw allow 443/udp  # Allow QUIC
   ```

3. **Container Security**
   - Run containers as non-root user
   - Use read-only filesystems where possible
   - Apply security patches regularly

### 🔄 Maintenance & Updates

#### Regular Maintenance Tasks

**Daily**:
- Health check monitoring
- Log review for errors
- Resource usage monitoring

**Weekly**:
- Docker image updates
- System cleanup
- Configuration backup

**Monthly**:
- Security updates
- Certificate renewal
- Performance review

#### Update Procedures

```bash
# Rolling update (zero downtime)
docker compose pull
docker compose up -d --force-recreate --no-deps api
docker compose up -d --force-recreate --no-deps worker

# Full restart (brief downtime)
make restart ENV=prod
```

### 📞 Support & Resources

#### Getting Help

1. **Documentation**: Check relevant guides in this directory
2. **Health Checks**: Run `./scripts/health-check.sh production`
3. **Logs**: Check `docker compose logs` for error details
4. **Monitoring**: Use Grafana dashboards for system insights

#### Useful Commands

```bash
# Quick status check
make status ENV=prod

# View recent logs
make logs ENV=prod

# Run health checks
make health ENV=prod

# Access container shell
docker compose exec api bash

# Monitor resources
docker stats

# Check Redis queue
redis-cli -h localhost -p 6379 llen rq:queue:default
```

#### External Resources

- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Redis Documentation](https://redis.io/documentation)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

---

## Document Index

### Setup & Deployment
- [Environment Setup Script](../scripts/setup-environment.sh)
- [Deployment Script](../scripts/deploy.sh)
- [Docker Compose Files](../docker-compose.yml)
- [Makefile](../Makefile)

### Configuration
- [Environment Examples](../.env.example)
- [API Configuration](../api_config.yaml)
- [Pipeline Configuration](../pipeline_config.yaml)
- [Nginx Configuration](../config/nginx.conf)
- [Redis Configuration](../config/redis.conf)

### Monitoring & Operations
- [Health Check Script](../scripts/health-check.sh)
- [Monitoring Setup Script](../scripts/setup-monitoring.sh)
- [Operations Guide](operations-monitoring.md)

### Development
- [Development Docker Compose](../docker-compose.dev.yml)
- [Test Configuration](../tests/)
- [Example Scripts](../examples/)

---

*For the most up-to-date information, always refer to the individual documentation files and inline code comments.*