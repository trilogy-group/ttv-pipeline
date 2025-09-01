# TTV Pipeline API Testing Guide

This guide explains how to test the TTV Pipeline API after deployment with `docker compose up`.

## Quick Start

1. **Start the API stack:**
   ```bash
   docker compose up -d
   ```

2. **Verify all services are healthy:**
   ```bash
   docker compose ps
   ```
   All services should show "healthy" status.

## API Endpoints Testing

### Health Check Endpoints

**Basic Health Check:**
```bash
curl http://localhost:8000/healthz
```
Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-09-01T05:30:49.724994Z",
  "version": "1.0.0",
  "components": {
    "api": "healthy"
  }
}
```

**Readiness Check:**
```bash
curl http://localhost:8000/readyz
```
Expected response (may show GCS as unhealthy without credentials):
```json
{
  "status": "ready",
  "timestamp": "2025-09-01T05:31:04.969660Z",
  "version": "1.0.0",
  "components": {
    "redis": "healthy",
    "gcs": "unhealthy",
    "workers": "healthy",
    "api": "healthy"
  }
}
```

**Via Nginx Proxy:**
```bash
curl http://localhost/healthz
```

### Job Management Endpoints

**List Jobs:**
```bash
curl http://localhost:8000/jobs
```

**Create a Job:**
```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing with a ball"
  }'
```

**Get Job Status:**
```bash
curl http://localhost:8000/jobs/{job_id}
```

**Cancel a Job:**
```bash
curl -X DELETE http://localhost:8000/jobs/{job_id}
```

**Get Video URL (for completed jobs):**
```bash
curl http://localhost:8000/jobs/{job_id}/video-url
```

**Get Video URL with custom expiration:**
```bash
curl "http://localhost:8000/jobs/{job_id}/video-url?expiration_seconds=7200"
```

### Generator Endpoints

**List Available Generators:**
```bash
curl http://localhost:8000/generators
```

**Get Generator Details:**
```bash
curl http://localhost:8000/generators/minimax
```

### Artifact Endpoints

**List Artifacts:**
```bash
curl http://localhost:8000/artifacts
```

**Download Artifact:**
```bash
curl http://localhost:8000/artifacts/{artifact_id}/download
```

## Testing with Different Environments

### Development Mode
```bash
# Uses .env file with ENVIRONMENT=development
docker compose up -d
```

### Production Mode
```bash
# Copy and modify .env.prod
cp .env.example .env.prod
# Edit .env.prod to set ENVIRONMENT=production and add SSL certificates
docker compose --env-file .env.prod up -d
```

## Monitoring and Debugging

### View Logs
```bash
# All services
docker compose logs

# Specific service
docker compose logs api
docker compose logs worker
docker compose logs redis
docker compose logs nginx

# Follow logs in real-time
docker compose logs -f api
```

### Check Service Health
```bash
# Container status
docker compose ps

# Detailed health information
docker compose exec api curl localhost:8000/healthz
docker compose exec redis redis-cli ping
```

### Redis Debugging (Optional)
```bash
# Start Redis Commander for GUI debugging
docker compose --profile debug up -d redis-commander

# Access at http://localhost:8081
```

## Performance Testing

### Load Testing with curl
```bash
# Simple load test
for i in {1..10}; do
  curl -s http://localhost:8000/healthz &
done
wait
```

### Using Apache Bench (if installed)
```bash
ab -n 100 -c 10 http://localhost:8000/healthz
```

## Troubleshooting

### Common Issues

1. **API fails to start with SSL errors:**
   - Ensure `ENVIRONMENT=development` in `.env` file
   - Or provide SSL certificates for production mode

2. **Redis connection errors:**
   - Check if Redis container is healthy: `docker compose ps`
   - Verify Redis logs: `docker compose logs redis`

3. **Worker not processing jobs:**
   - Check worker logs: `docker compose logs worker`
   - Verify Redis connectivity from workers

4. **Nginx proxy errors:**
   - Check nginx configuration: `docker compose logs nginx`
   - Verify API is healthy before nginx starts

### Reset Everything
```bash
# Stop and remove all containers, networks, and volumes
docker compose down -v

# Rebuild and restart
docker compose build --no-cache
docker compose up -d
```

## API Documentation

- **OpenAPI/Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc  
- **OpenAPI JSON:** http://localhost:8000/openapi.json

### Troubleshooting Documentation UI

If the Swagger UI shows a blank page:

1. **Check browser console** for JavaScript errors
2. **Disable ad blockers** that might block CDN resources
3. **Try ReDoc instead:** http://localhost:8000/redoc
4. **Use the raw OpenAPI JSON:** http://localhost:8000/openapi.json
5. **Import into Postman/Insomnia:** Use the OpenAPI JSON URL

### Alternative API Testing Tools

**Using curl with OpenAPI spec:**
```bash
# Get the OpenAPI spec
curl http://localhost:8000/openapi.json > api-spec.json

# Use tools like httpie, postman, or insomnia to import the spec
```

**Using httpie (if installed):**
```bash
# Install httpie: pip install httpie
http GET localhost:8000/healthz
http POST localhost:8000/jobs prompt="A cat playing with a ball"
```

## Environment Variables

Key environment variables for testing:

```bash
# Core settings
ENVIRONMENT=development          # or production
LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING, ERROR

# API settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1                   # Number of API workers

# Redis settings
REDIS_HOST=redis                # Container name in docker-compose
REDIS_PORT=6379

# Worker settings
WORKER_REPLICAS=2               # Number of worker containers
WORKER_CONCURRENCY=1            # Jobs per worker

# GCS settings (optional for basic testing)
GCS_BUCKET=ttv-api-artifacts
GCS_CREDENTIALS_PATH=/app/credentials/gcs-key.json
```

## Next Steps

After successful testing:

1. **Add GCS credentials** for full functionality
2. **Configure SSL certificates** for production deployment
3. **Set up monitoring** with the provided monitoring scripts
4. **Scale workers** based on load requirements
5. **Configure backup** for Redis data persistence