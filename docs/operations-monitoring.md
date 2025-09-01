# Operations and Monitoring Guide

This guide provides comprehensive operational procedures and monitoring strategies for the TTV Pipeline API in production environments.

## Table of Contents

1. [Monitoring Overview](#monitoring-overview)
2. [Health Check Procedures](#health-check-procedures)
3. [Metrics and Alerting](#metrics-and-alerting)
4. [Log Management](#log-management)
5. [Performance Monitoring](#performance-monitoring)
6. [Incident Response](#incident-response)
7. [Capacity Planning](#capacity-planning)
8. [Operational Runbooks](#operational-runbooks)

## Monitoring Overview

### Monitoring Stack Architecture

```
Application → Metrics → Prometheus → Grafana → Alerts
     ↓           ↓
   Logs → Loki → Grafana → Notifications
```

### Key Monitoring Components

- **Health Endpoints**: Built-in API health checks
- **Metrics Collection**: Prometheus-compatible metrics
- **Log Aggregation**: Structured JSON logging
- **Alerting**: Automated incident detection
- **Dashboards**: Real-time operational visibility

## Health Check Procedures

### Automated Health Checks

The system provides multiple health check endpoints for different monitoring needs:

#### 1. Liveness Check (`/healthz`)

**Purpose**: Verify the API server is running and responsive
**Endpoint**: `GET /healthz`
**Expected Response**: `200 OK`

```bash
# Manual check
curl -f http://localhost:8000/healthz

# Automated monitoring
*/1 * * * * curl -f http://localhost:8000/healthz || echo "API health check failed" | mail -s "Alert" ops@company.com
```

#### 2. Readiness Check (`/readyz`)

**Purpose**: Verify the API server and all dependencies are ready
**Endpoint**: `GET /readyz`
**Checks**: Redis connectivity, GCS access, worker availability

```bash
# Manual check
curl -f http://localhost:8000/readyz

# Kubernetes readiness probe
readinessProbe:
  httpGet:
    path: /readyz
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
```

#### 3. Comprehensive Health Script

Use the provided health check script for complete system validation:

```bash
# Run full health check
./scripts/health-check.sh production

# Automated daily health check
0 6 * * * /path/to/ttv-api/scripts/health-check.sh production >> /var/log/ttv-health.log 2>&1
```

### Component-Specific Health Checks

#### Redis Health Check

```bash
# Basic connectivity
redis-cli -h localhost -p 6379 ping

# Memory usage check
redis-cli -h localhost -p 6379 info memory | grep used_memory_human

# Queue depth monitoring
redis-cli -h localhost -p 6379 llen rq:queue:default
```

#### Worker Health Check

```bash
# Check active workers
docker compose ps worker

# Monitor worker logs for errors
docker compose logs worker --since 1h | grep ERROR

# Check job processing rate
redis-cli -h localhost -p 6379 info stats | grep instantaneous_ops_per_sec
```

#### GCS Connectivity Check

```bash
# Test GCS access (requires gsutil)
gsutil ls gs://your-bucket-name/

# Check recent uploads
gsutil ls -l gs://your-bucket-name/ttv-api/$(date +%Y-%m)/ | tail -10
```

## Metrics and Alerting

### Key Performance Indicators (KPIs)

#### API Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `http_requests_total` | Total HTTP requests | - |
| `http_request_duration_seconds` | Request latency | P95 > 2s |
| `http_requests_errors_total` | HTTP error count | Rate > 5% |
| `api_active_connections` | Active connections | > 1000 |

#### Job Processing Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `jobs_queued_total` | Jobs in queue | > 100 |
| `jobs_processing_total` | Jobs being processed | - |
| `jobs_completed_total` | Completed jobs | - |
| `jobs_failed_total` | Failed jobs | Rate > 10% |
| `job_duration_seconds` | Job processing time | P95 > 300s |

#### System Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `redis_connected_clients` | Redis connections | > 100 |
| `redis_used_memory_bytes` | Redis memory usage | > 80% |
| `worker_active_jobs` | Active worker jobs | - |
| `gcs_upload_duration_seconds` | GCS upload time | P95 > 30s |

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "ttv-api-rules.yml"

scrape_configs:
  - job_name: 'ttv-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    scrape_timeout: 10s

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']  # Redis exporter

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Alert Rules

```yaml
# ttv-api-rules.yml
groups:
  - name: ttv-api-alerts
    rules:
      # API Health Alerts
      - alert: APIDown
        expr: up{job="ttv-api"} == 0
        for: 1m
        labels:
          severity: critical
          component: api
        annotations:
          summary: "TTV API is down"
          description: "The TTV API has been down for more than 1 minute"
          runbook_url: "https://docs.company.com/runbooks/api-down"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
          component: api
        annotations:
          summary: "High API error rate"
          description: "API error rate is {{ $value | humanizePercentage }} over the last 5 minutes"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 10m
        labels:
          severity: warning
          component: api
        annotations:
          summary: "High API latency"
          description: "95th percentile latency is {{ $value }}s"

      # Job Processing Alerts
      - alert: JobQueueBacklog
        expr: redis_queue_depth{queue="default"} > 100
        for: 15m
        labels:
          severity: warning
          component: workers
        annotations:
          summary: "Job queue backlog"
          description: "{{ $value }} jobs are queued for processing"

      - alert: HighJobFailureRate
        expr: rate(jobs_failed_total[10m]) / rate(jobs_total[10m]) > 0.1
        for: 10m
        labels:
          severity: warning
          component: workers
        annotations:
          summary: "High job failure rate"
          description: "Job failure rate is {{ $value | humanizePercentage }}"

      - alert: NoActiveWorkers
        expr: worker_active_jobs == 0 and redis_queue_depth > 0
        for: 5m
        labels:
          severity: critical
          component: workers
        annotations:
          summary: "No active workers"
          description: "No workers are processing jobs but queue has {{ $value }} jobs"

      # System Resource Alerts
      - alert: RedisHighMemoryUsage
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.8
        for: 10m
        labels:
          severity: warning
          component: redis
        annotations:
          summary: "Redis high memory usage"
          description: "Redis memory usage is {{ $value | humanizePercentage }}"

      - alert: RedisDown
        expr: up{job="redis"} == 0
        for: 1m
        labels:
          severity: critical
          component: redis
        annotations:
          summary: "Redis is down"
          description: "Redis has been down for more than 1 minute"
```

### Grafana Dashboards

#### API Performance Dashboard

```json
{
  "dashboard": {
    "title": "TTV API Performance",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])",
            "format": "percentunit"
          }
        ]
      }
    ]
  }
}
```

#### Job Processing Dashboard

```json
{
  "dashboard": {
    "title": "TTV Job Processing",
    "panels": [
      {
        "title": "Queue Depth",
        "type": "graph",
        "targets": [
          {
            "expr": "redis_queue_depth",
            "legendFormat": "{{queue}}"
          }
        ]
      },
      {
        "title": "Job Processing Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(jobs_completed_total[5m])",
            "legendFormat": "Completed"
          },
          {
            "expr": "rate(jobs_failed_total[5m])",
            "legendFormat": "Failed"
          }
        ]
      },
      {
        "title": "Active Workers",
        "type": "singlestat",
        "targets": [
          {
            "expr": "sum(worker_active_jobs)"
          }
        ]
      }
    ]
  }
}
```

## Log Management

### Log Structure

All services emit structured JSON logs with consistent fields:

```json
{
  "timestamp": "2025-08-31T12:34:56.789Z",
  "level": "INFO",
  "component": "api",
  "job_id": "job_abc123",
  "user_id": "user_456",
  "request_id": "req_789",
  "message": "Job processing started",
  "duration_ms": 150,
  "metadata": {
    "prompt_length": 45,
    "worker_id": "worker_001"
  }
}
```

### Log Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General operational messages
- **WARNING**: Warning conditions that don't affect operation
- **ERROR**: Error conditions that may affect operation
- **CRITICAL**: Critical conditions that require immediate attention

### Log Aggregation with Loki

#### Loki Configuration

```yaml
# loki-config.yml
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
    final_sleep: 0s

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 168h

storage_config:
  boltdb:
    directory: /loki/index
  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h
```

#### Promtail Configuration

```yaml
# promtail-config.yml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: ttv-api
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: ['__meta_docker_container_label_com_docker_compose_service']
        target_label: 'service'
      - source_labels: ['__meta_docker_container_name']
        target_label: 'container'
    pipeline_stages:
      - json:
          expressions:
            level: level
            component: component
            job_id: job_id
            message: message
      - labels:
          level:
          component:
          job_id:
```

### Log Analysis Queries

#### Common LogQL Queries

```logql
# All error logs in the last hour
{service="ttv-api"} |= "ERROR" | json | level="ERROR"

# Job processing logs for specific job
{service="ttv-api"} | json | job_id="job_abc123"

# API request logs with high latency
{service="ttv-api"} | json | component="api" | duration_ms > 2000

# Worker error rate
rate({service="ttv-api"} | json | component="worker" | level="ERROR"[5m])

# Failed job analysis
{service="ttv-api"} | json | message=~".*failed.*" | line_format "{{.timestamp}} {{.job_id}} {{.message}}"
```

## Performance Monitoring

### System Resource Monitoring

#### CPU and Memory Monitoring

```bash
# Monitor container resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

# System-wide monitoring
htop
iotop -o
```

#### Disk Usage Monitoring

```bash
# Check disk space
df -h

# Monitor Docker volume usage
docker system df

# Check Redis data size
du -sh /var/lib/docker/volumes/ttv-api_redis-data/_data/
```

### Application Performance Monitoring (APM)

#### Request Tracing

Enable distributed tracing for request flow analysis:

```python
# Example: OpenTelemetry integration
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
```

#### Performance Profiling

```bash
# Enable profiling in development
PROFILING_ENABLED=true docker compose up -d

# Access profiling data
curl http://localhost:8000/debug/pprof/profile > cpu.prof
go tool pprof cpu.prof
```

### Database Performance (Redis)

#### Redis Monitoring Commands

```bash
# Monitor Redis performance
redis-cli --latency-history -h localhost -p 6379

# Check slow queries
redis-cli slowlog get 10

# Monitor memory usage
redis-cli info memory

# Check key statistics
redis-cli --scan --pattern "rq:*" | wc -l
```

## Incident Response

### Incident Classification

#### Severity Levels

- **P0 (Critical)**: Complete service outage
- **P1 (High)**: Major functionality impaired
- **P2 (Medium)**: Minor functionality impaired
- **P3 (Low)**: Cosmetic issues or minor bugs

### Incident Response Procedures

#### P0 - Critical Incidents

**Examples**: API completely down, all workers failed, data loss

**Response Time**: 15 minutes
**Resolution Time**: 2 hours

**Immediate Actions**:
1. Acknowledge the incident
2. Assess impact and scope
3. Implement immediate mitigation
4. Communicate to stakeholders
5. Begin root cause analysis

```bash
# Emergency response commands
# Check service status
./scripts/health-check.sh production

# Restart all services
docker compose restart

# Scale up workers if needed
docker compose up -d --scale worker=8

# Check recent logs for errors
docker compose logs --since 30m | grep ERROR
```

#### P1 - High Priority Incidents

**Examples**: High error rate, significant performance degradation

**Response Time**: 30 minutes
**Resolution Time**: 4 hours

**Actions**:
1. Investigate root cause
2. Implement temporary workaround
3. Monitor system stability
4. Plan permanent fix

### Escalation Procedures

```
Level 1: On-call Engineer (0-30 minutes)
    ↓
Level 2: Senior Engineer + Team Lead (30-60 minutes)
    ↓
Level 3: Engineering Manager + Product Owner (1-2 hours)
    ↓
Level 4: CTO + Executive Team (2+ hours)
```

### Communication Templates

#### Incident Notification

```
INCIDENT: [P0] TTV API Service Outage

Status: INVESTIGATING
Started: 2025-08-31 14:30 UTC
Impact: Complete service unavailability
Affected: All API endpoints

We are investigating reports of complete service unavailability.
Updates will be provided every 15 minutes.

Next Update: 14:45 UTC
Incident Commander: [Name]
```

#### Resolution Notification

```
RESOLVED: [P0] TTV API Service Outage

Status: RESOLVED
Started: 2025-08-31 14:30 UTC
Resolved: 2025-08-31 15:15 UTC
Duration: 45 minutes

Root Cause: Redis connection pool exhaustion
Resolution: Restarted Redis service and increased connection limits

Post-mortem will be published within 48 hours.
```

## Capacity Planning

### Resource Utilization Targets

| Resource | Target Utilization | Alert Threshold |
|----------|-------------------|-----------------|
| CPU | 70% | 85% |
| Memory | 80% | 90% |
| Disk | 70% | 85% |
| Network | 60% | 80% |

### Scaling Triggers

#### Horizontal Scaling

**Scale Up Triggers**:
- Queue depth > 50 for 10+ minutes
- CPU utilization > 80% for 15+ minutes
- Response time P95 > 2 seconds for 10+ minutes

**Scale Down Triggers**:
- Queue depth < 10 for 30+ minutes
- CPU utilization < 40% for 30+ minutes
- All workers idle for 20+ minutes

#### Vertical Scaling

**Memory Scaling**:
- Redis memory usage > 80%
- Worker memory usage > 85%
- Frequent OOM kills

**CPU Scaling**:
- Sustained high CPU usage
- Increased request latency
- Queue processing slowdown

### Growth Planning

#### Traffic Growth Projections

```python
# Example capacity calculation
current_rps = 100  # requests per second
growth_rate = 0.2  # 20% monthly growth
months = 12

projected_rps = current_rps * (1 + growth_rate) ** months
required_workers = projected_rps / 10  # 10 RPS per worker
required_memory = required_workers * 2  # 2GB per worker

print(f"Projected RPS in {months} months: {projected_rps:.0f}")
print(f"Required workers: {required_workers:.0f}")
print(f"Required memory: {required_memory:.0f}GB")
```

## Operational Runbooks

### Daily Operations Checklist

```bash
#!/bin/bash
# Daily operations checklist

echo "=== Daily TTV API Health Check ==="
date

# 1. Check service health
echo "1. Checking service health..."
./scripts/health-check.sh production

# 2. Check resource usage
echo "2. Checking resource usage..."
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# 3. Check queue depth
echo "3. Checking queue depth..."
redis-cli -h localhost -p 6379 llen rq:queue:default

# 4. Check error logs
echo "4. Checking recent errors..."
docker compose logs --since 24h | grep ERROR | wc -l

# 5. Check disk space
echo "5. Checking disk space..."
df -h | grep -E "(/$|/var)"

# 6. Check certificate expiry
echo "6. Checking certificate expiry..."
openssl x509 -in certs/cert.pem -noout -dates

echo "=== Daily check complete ==="
```

### Weekly Maintenance Tasks

```bash
#!/bin/bash
# Weekly maintenance tasks

echo "=== Weekly TTV API Maintenance ==="
date

# 1. Update Docker images
echo "1. Updating Docker images..."
docker compose pull

# 2. Clean up unused resources
echo "2. Cleaning up unused resources..."
docker system prune -f

# 3. Backup Redis data
echo "3. Backing up Redis data..."
./scripts/backup.sh

# 4. Check log rotation
echo "4. Checking log sizes..."
du -sh /var/lib/docker/containers/*/

# 5. Update system packages
echo "5. Checking for system updates..."
apt list --upgradable

echo "=== Weekly maintenance complete ==="
```

### Emergency Procedures

#### Complete Service Recovery

```bash
#!/bin/bash
# Emergency service recovery procedure

echo "=== EMERGENCY RECOVERY PROCEDURE ==="
echo "This will restart all services and may cause brief downtime"
read -p "Continue? (y/N): " confirm

if [[ $confirm == [yY] ]]; then
    # 1. Stop all services
    echo "Stopping all services..."
    docker compose down

    # 2. Clean up any stuck containers
    echo "Cleaning up containers..."
    docker container prune -f

    # 3. Check disk space
    echo "Checking disk space..."
    df -h

    # 4. Start services with fresh state
    echo "Starting services..."
    docker compose up -d

    # 5. Wait for services to be ready
    echo "Waiting for services to be ready..."
    sleep 30

    # 6. Run health checks
    echo "Running health checks..."
    ./scripts/health-check.sh production

    echo "=== RECOVERY COMPLETE ==="
else
    echo "Recovery cancelled"
fi
```

This comprehensive operations and monitoring guide provides the foundation for reliable production operation of the TTV Pipeline API with proper observability, incident response, and maintenance procedures.