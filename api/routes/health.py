"""
Health and monitoring routes for the API server.

This module provides comprehensive health checks, readiness validation,
and Prometheus-compatible metrics collection for monitoring and observability.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from fastapi import APIRouter, Request, HTTPException, Response
from fastapi.responses import PlainTextResponse

from api import __version__
from api.models import HealthResponse, MetricsResponse
from api.queue import get_job_queue, get_redis_manager
from api.gcs_client import GCSClient

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# Application startup time for uptime calculation
_startup_time = time.time()

# Enhanced metrics storage (in production, use proper metrics backend like Prometheus)
_metrics_storage = {
    "total_jobs_processed": 0,
    "total_request_count": 0,
    "total_processing_time": 0.0,
    "total_request_duration": 0.0,
    "request_latency_buckets": {
        "0.1": 0,    # < 100ms
        "0.5": 0,    # < 500ms  
        "1.0": 0,    # < 1s
        "2.5": 0,    # < 2.5s
        "5.0": 0,    # < 5s
        "10.0": 0,   # < 10s
        "+Inf": 0    # >= 10s
    },
    "job_duration_buckets": {
        "30": 0,     # < 30s
        "60": 0,     # < 1min
        "300": 0,    # < 5min
        "600": 0,    # < 10min
        "1800": 0,   # < 30min
        "3600": 0,   # < 1hr
        "+Inf": 0    # >= 1hr
    },
    "status_code_counts": {},
    "last_reset_time": _startup_time
}


@router.get("/healthz", response_model=HealthResponse)
async def health_check():
    """
    Basic liveness health check endpoint.
    
    Returns HTTP 200 if the application is running and can handle requests.
    This endpoint should always return success unless the application is
    completely broken.
    
    Returns:
        HealthResponse: Basic health status with timestamp and version
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version=__version__,
        components={"api": "healthy"}
    )


@router.get("/readyz", response_model=HealthResponse)
async def readiness_check(request: Request):
    """
    Readiness check with dependency validation.
    
    Checks that all required dependencies (Redis, GCS) are available
    and the service is ready to handle requests. Returns HTTP 503 if
    any critical dependency is unavailable.
    
    Args:
        request: FastAPI request object with app state
        
    Returns:
        HealthResponse: Readiness status with component health details
        
    Raises:
        HTTPException: 503 if service is not ready
    """
    config = request.app.state.config
    components = {}
    overall_status = "ready"
    
    # Check Redis connectivity
    try:
        redis_manager = getattr(request.app.state, 'redis_manager', None)
        if redis_manager and redis_manager.test_connection():
            components["redis"] = "healthy"
            logger.debug("Redis connectivity check passed")
        else:
            components["redis"] = "unhealthy"
            overall_status = "not_ready"
            logger.warning("Redis connectivity check failed")
    except Exception as e:
        components["redis"] = "unhealthy"
        overall_status = "not_ready"
        logger.error(f"Redis connectivity check error: {e}")
    
    # Check GCS connectivity
    try:
        gcs_client = GCSClient(config.gcs)
        if await gcs_client.test_connection():
            components["gcs"] = "healthy"
            logger.debug("GCS connectivity check passed")
        else:
            components["gcs"] = "unhealthy"
            overall_status = "not_ready"
            logger.warning("GCS connectivity check failed")
    except Exception as e:
        components["gcs"] = "unhealthy"
        overall_status = "not_ready"
        logger.error(f"GCS connectivity check error: {e}")
    
    # Check worker availability
    try:
        job_queue = getattr(request.app.state, 'job_queue', None)
        if job_queue:
            queue_stats = job_queue.get_queue_stats()
            
            # Consider workers available if we can get queue stats
            # In a more sophisticated setup, you might check for active workers
            if queue_stats is not None:
                components["workers"] = "healthy"
                logger.debug("Worker availability check passed")
            else:
                components["workers"] = "unhealthy"
                overall_status = "not_ready"
                logger.warning("Worker availability check failed")
        else:
            components["workers"] = "unhealthy"
            overall_status = "not_ready"
            logger.warning("Job queue not initialized")
    except Exception as e:
        components["workers"] = "unhealthy"
        overall_status = "not_ready"
        logger.error(f"Worker availability check error: {e}")
    
    # Add API component status
    components["api"] = "healthy"
    
    response = HealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        version=__version__,
        components=components
    )
    
    # Return 503 if not ready
    if overall_status != "ready":
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content=response.model_dump(mode='json')
        )
    
    return response


@router.get("/metrics")
async def metrics_json(request: Request) -> Dict[str, Any]:
    """
    JSON metrics endpoint for programmatic access.
    
    Returns metrics in JSON format for easy consumption by monitoring
    systems that prefer structured data over Prometheus format.
    
    Args:
        request: FastAPI request object with app state
        
    Returns:
        MetricsResponse: Current system metrics
    """
    try:
        job_queue = getattr(request.app.state, 'job_queue', None)
        queue_stats = job_queue.get_queue_stats() if job_queue else {}
        
        # Calculate uptime
        uptime_seconds = int(time.time() - _startup_time)
        
        # Calculate average processing time
        total_jobs = _metrics_storage["total_jobs_processed"]
        total_time = _metrics_storage["total_processing_time"]
        avg_processing_time = total_time / total_jobs if total_jobs > 0 else 0.0
        
        # Calculate average request latency
        total_requests = _metrics_storage["total_request_count"]
        total_request_duration = _metrics_storage["total_request_duration"]
        avg_request_latency = total_request_duration / total_requests if total_requests > 0 else 0.0
        
        # Active jobs = started + progress status jobs
        active_jobs = queue_stats.get("started_jobs", 0)
        queued_jobs = queue_stats.get("queued_jobs", 0)
        
        # Enhanced metrics response
        enhanced_metrics = MetricsResponse(
            active_jobs=active_jobs,
            queued_jobs=queued_jobs,
            total_jobs_processed=total_jobs,
            average_processing_time=avg_processing_time,
            uptime_seconds=uptime_seconds
        )
        
        # Add enhanced metrics as additional fields
        enhanced_metrics_dict = enhanced_metrics.model_dump()
        enhanced_metrics_dict.update({
            "total_requests": total_requests,
            "average_request_latency": round(avg_request_latency, 3),
            "requests_per_second": round(total_requests / uptime_seconds, 2) if uptime_seconds > 0 else 0.0,
            "jobs_per_hour": round(total_jobs * 3600 / uptime_seconds, 2) if uptime_seconds > 0 else 0.0,
            "queue_depth": queued_jobs + active_jobs,
            "status_code_distribution": dict(_metrics_storage["status_code_counts"])
        })
        
        return enhanced_metrics_dict
        
    except Exception as e:
        logger.error(f"Failed to collect metrics: {e}")
        # Return basic metrics even if queue is unavailable
        uptime_seconds = int(time.time() - _startup_time)
        total_jobs = _metrics_storage["total_jobs_processed"]
        total_requests = _metrics_storage["total_request_count"]
        
        return {
            "active_jobs": 0,
            "queued_jobs": 0,
            "total_jobs_processed": total_jobs,
            "average_processing_time": 0.0,
            "uptime_seconds": uptime_seconds,
            "total_requests": total_requests,
            "average_request_latency": 0.0,
            "requests_per_second": 0.0,
            "jobs_per_hour": 0.0,
            "queue_depth": 0,
            "status_code_distribution": dict(_metrics_storage["status_code_counts"])
        }


@router.get("/metrics/prometheus", response_class=PlainTextResponse)
async def metrics_prometheus(request: Request) -> str:
    """
    Prometheus-compatible metrics endpoint.
    
    Returns metrics in Prometheus exposition format for scraping by
    Prometheus monitoring systems. Includes standard metrics for
    job processing, queue depth, and system health.
    
    Args:
        request: FastAPI request object with app state
        
    Returns:
        str: Prometheus-formatted metrics
    """
    try:
        job_queue = getattr(request.app.state, 'job_queue', None)
        queue_stats = job_queue.get_queue_stats() if job_queue else {}
        
        # Calculate uptime
        uptime_seconds = int(time.time() - _startup_time)
        
        # Get metrics from storage
        total_jobs = _metrics_storage["total_jobs_processed"]
        total_requests = _metrics_storage["total_request_count"]
        total_time = _metrics_storage["total_processing_time"]
        total_request_duration = _metrics_storage["total_request_duration"]
        avg_processing_time = total_time / total_jobs if total_jobs > 0 else 0.0
        avg_request_latency = total_request_duration / total_requests if total_requests > 0 else 0.0
        
        # Calculate throughput metrics
        requests_per_second = total_requests / uptime_seconds if uptime_seconds > 0 else 0.0
        jobs_per_hour = total_jobs * 3600 / uptime_seconds if uptime_seconds > 0 else 0.0
        
        # Build Prometheus metrics
        metrics_lines = [
            "# HELP ttv_api_info API server information",
            "# TYPE ttv_api_info gauge",
            f'ttv_api_info{{version="{__version__}"}} 1',
            "",
            "# HELP ttv_api_uptime_seconds Server uptime in seconds",
            "# TYPE ttv_api_uptime_seconds counter",
            f"ttv_api_uptime_seconds {uptime_seconds}",
            "",
            "# HELP ttv_api_jobs_total Total number of jobs processed",
            "# TYPE ttv_api_jobs_total counter",
            f"ttv_api_jobs_total {total_jobs}",
            "",
            "# HELP ttv_api_requests_total Total number of API requests",
            "# TYPE ttv_api_requests_total counter",
            f"ttv_api_requests_total {total_requests}",
            "",
            "# HELP ttv_api_job_processing_time_seconds Average job processing time",
            "# TYPE ttv_api_job_processing_time_seconds gauge",
            f"ttv_api_job_processing_time_seconds {avg_processing_time:.2f}",
            "",
            "# HELP ttv_api_request_latency_seconds Average request latency",
            "# TYPE ttv_api_request_latency_seconds gauge", 
            f"ttv_api_request_latency_seconds {avg_request_latency:.3f}",
            "",
            "# HELP ttv_api_requests_per_second Request throughput rate",
            "# TYPE ttv_api_requests_per_second gauge",
            f"ttv_api_requests_per_second {requests_per_second:.2f}",
            "",
            "# HELP ttv_api_jobs_per_hour Job processing throughput rate",
            "# TYPE ttv_api_jobs_per_hour gauge", 
            f"ttv_api_jobs_per_hour {jobs_per_hour:.2f}",
            "",
            "# HELP ttv_api_request_duration_seconds_bucket Request latency histogram buckets",
            "# TYPE ttv_api_request_duration_seconds_bucket histogram",
        ]
        
        # Add request latency histogram buckets
        for bucket, count in _metrics_storage["request_latency_buckets"].items():
            if bucket == "+Inf":
                metrics_lines.append(f'ttv_api_request_duration_seconds_bucket{{le="+Inf"}} {count}')
            else:
                metrics_lines.append(f'ttv_api_request_duration_seconds_bucket{{le="{bucket}"}} {count}')
        
        metrics_lines.extend([
            "",
            "# HELP ttv_api_job_duration_seconds_bucket Job processing duration histogram buckets", 
            "# TYPE ttv_api_job_duration_seconds_bucket histogram",
        ])
        
        # Add job duration histogram buckets
        for bucket, count in _metrics_storage["job_duration_buckets"].items():
            if bucket == "+Inf":
                metrics_lines.append(f'ttv_api_job_duration_seconds_bucket{{le="+Inf"}} {count}')
            else:
                metrics_lines.append(f'ttv_api_job_duration_seconds_bucket{{le="{bucket}"}} {count}')
        
        metrics_lines.extend([
            "",
            "# HELP ttv_api_queue_jobs Current number of jobs in queue by status",
            "# TYPE ttv_api_queue_jobs gauge",
        ])
        
        # Add queue metrics by status
        for status, count in queue_stats.items():
            status_name = status.replace("_jobs", "")
            metrics_lines.append(f'ttv_api_queue_jobs{{status="{status_name}"}} {count}')
        
        # Add total queue depth metric
        total_queue_depth = sum(count for status, count in queue_stats.items() 
                               if status in ["queued_jobs", "started_jobs"])
        metrics_lines.extend([
            "",
            "# HELP ttv_api_queue_depth_total Total jobs in processing pipeline",
            "# TYPE ttv_api_queue_depth_total gauge",
            f"ttv_api_queue_depth_total {total_queue_depth}",
        ])
        
        # Add HTTP status code metrics
        if _metrics_storage["status_code_counts"]:
            metrics_lines.extend([
                "",
                "# HELP ttv_api_http_requests_total Total HTTP requests by status code",
                "# TYPE ttv_api_http_requests_total counter",
            ])
            for status_code, count in _metrics_storage["status_code_counts"].items():
                metrics_lines.append(f'ttv_api_http_requests_total{{code="{status_code}"}} {count}')
        
        metrics_lines.extend([
            "",
            "# HELP ttv_api_health_status Component health status (1=healthy, 0=unhealthy)",
            "# TYPE ttv_api_health_status gauge",
        ])
        
        # Add component health metrics
        try:
            redis_manager = getattr(request.app.state, 'redis_manager', None)
            redis_healthy = 1 if redis_manager and redis_manager.test_connection() else 0
            metrics_lines.append(f'ttv_api_health_status{{component="redis"}} {redis_healthy}')
        except Exception:
            metrics_lines.append('ttv_api_health_status{component="redis"} 0')
        
        try:
            config = request.app.state.config
            gcs_client = GCSClient(config.gcs)
            gcs_healthy = 1 if await gcs_client.test_connection() else 0
            metrics_lines.append(f'ttv_api_health_status{{component="gcs"}} {gcs_healthy}')
        except Exception:
            metrics_lines.append('ttv_api_health_status{component="gcs"} 0')
        
        # Always report API as healthy if we can generate metrics
        metrics_lines.append('ttv_api_health_status{component="api"} 1')
        
        return "\n".join(metrics_lines) + "\n"
        
    except Exception as e:
        logger.error(f"Failed to generate Prometheus metrics: {e}")
        # Return minimal metrics even if there's an error
        uptime_seconds = int(time.time() - _startup_time)
        return f"""# HELP ttv_api_uptime_seconds Server uptime in seconds
# TYPE ttv_api_uptime_seconds counter
ttv_api_uptime_seconds {uptime_seconds}

# HELP ttv_api_health_status Component health status (1=healthy, 0=unhealthy)
# TYPE ttv_api_health_status gauge
ttv_api_health_status{{component="api"}} 1
"""


@router.get("/metrics/system")
async def system_metrics(request: Request) -> Dict[str, Any]:
    """
    Comprehensive system metrics endpoint for advanced monitoring.
    
    Returns detailed system health, performance, and capacity metrics
    for comprehensive observability and alerting.
    
    Args:
        request: FastAPI request object with app state
        
    Returns:
        Dict: Comprehensive system metrics
    """
    try:
        # Get app state components
        redis_manager = getattr(request.app.state, 'redis_manager', None)
        job_queue = getattr(request.app.state, 'job_queue', None)
        
        # Get comprehensive system health metrics
        system_health = get_system_health_metrics(redis_manager, job_queue)
        
        # Get queue depth metrics
        queue_metrics = get_queue_depth_metrics(job_queue)
        
        # Get current timestamp
        timestamp = datetime.now(timezone.utc)
        
        # Compile comprehensive metrics
        comprehensive_metrics = {
            "timestamp": timestamp.isoformat(),
            "version": __version__,
            "system_health": system_health,
            "queue_metrics": queue_metrics,
            "histogram_data": {
                "request_latency_distribution": dict(_metrics_storage["request_latency_buckets"]),
                "job_duration_distribution": dict(_metrics_storage["job_duration_buckets"])
            },
            "throughput_metrics": {
                "requests_per_second": system_health["performance"]["requests_per_second"],
                "jobs_per_hour": system_health["performance"]["jobs_per_hour"],
                "total_requests": _metrics_storage["total_request_count"],
                "total_jobs": _metrics_storage["total_jobs_processed"]
            },
            "latency_metrics": {
                "average_request_latency": system_health["performance"]["average_request_latency"],
                "average_job_duration": system_health["performance"]["average_job_duration"],
                "p95_request_latency": _calculate_percentile(_metrics_storage["request_latency_buckets"], 0.95),
                "p99_request_latency": _calculate_percentile(_metrics_storage["request_latency_buckets"], 0.99)
            },
            "error_metrics": {
                "status_code_distribution": dict(_metrics_storage["status_code_counts"]),
                "error_rate": _calculate_error_rate(),
                "failed_jobs": queue_metrics["failed"]
            }
        }
        
        return comprehensive_metrics
        
    except Exception as e:
        logger.error(f"Failed to generate comprehensive system metrics: {e}")
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": __version__,
            "error": "Failed to collect system metrics",
            "message": str(e)
        }


def _calculate_percentile(buckets: Dict[str, int], percentile: float) -> float:
    """
    Calculate approximate percentile from histogram buckets.
    
    Args:
        buckets: Histogram bucket data (cumulative counts)
        percentile: Percentile to calculate (0.0 to 1.0)
        
    Returns:
        Approximate percentile value
    """
    # Find the highest bucket count (total observations)
    total_count = max(buckets.values()) if buckets.values() else 0
    if total_count == 0:
        return 0.0
    
    target_count = total_count * percentile
    
    # Sort buckets by threshold
    sorted_buckets = []
    for k, v in buckets.items():
        if k == "+Inf":
            sorted_buckets.append((float('inf'), v))
        else:
            sorted_buckets.append((float(k), v))
    
    sorted_buckets.sort(key=lambda x: x[0])
    
    # Find the bucket where cumulative count >= target
    for threshold, cumulative_count in sorted_buckets:
        if cumulative_count >= target_count:
            return threshold if threshold != float('inf') else 10.0  # Cap at 10s for display
    
    return 0.0


def _calculate_error_rate() -> float:
    """
    Calculate current error rate from status code distribution.
    
    Returns:
        Error rate as percentage (0.0 to 100.0)
    """
    status_counts = _metrics_storage["status_code_counts"]
    if not status_counts:
        return 0.0
    
    total_requests = sum(status_counts.values())
    error_requests = sum(count for status, count in status_counts.items() 
                        if int(status) >= 400)
    
    return round((error_requests / total_requests) * 100, 2) if total_requests > 0 else 0.0


def record_job_processed(processing_time_seconds: float):
    """
    Record that a job has been processed for metrics collection.
    
    Args:
        processing_time_seconds: Time taken to process the job
    """
    _metrics_storage["total_jobs_processed"] += 1
    _metrics_storage["total_processing_time"] += processing_time_seconds
    
    # Update job duration histogram buckets
    _update_histogram_bucket(_metrics_storage["job_duration_buckets"], processing_time_seconds)
    
    logger.debug(f"Recorded job processing time: {processing_time_seconds:.2f}s")


def record_request(duration_seconds: Optional[float] = None, status_code: Optional[int] = None):
    """
    Record that an API request has been made for metrics collection.
    
    Args:
        duration_seconds: Request duration in seconds
        status_code: HTTP status code
    """
    _metrics_storage["total_request_count"] += 1
    
    if duration_seconds is not None:
        _metrics_storage["total_request_duration"] += duration_seconds
        # Update request latency histogram buckets
        _update_histogram_bucket(_metrics_storage["request_latency_buckets"], duration_seconds)
    
    if status_code is not None:
        status_str = str(status_code)
        _metrics_storage["status_code_counts"][status_str] = _metrics_storage["status_code_counts"].get(status_str, 0) + 1


def record_request_latency(duration_seconds: float, status_code: int):
    """
    Record request latency and status code for enhanced metrics.
    
    Args:
        duration_seconds: Request duration in seconds
        status_code: HTTP status code
    """
    record_request(duration_seconds, status_code)


def _update_histogram_bucket(buckets: Dict[str, int], value: float):
    """
    Update histogram bucket counts for a given value.
    Histogram buckets are cumulative (le = "less than or equal to").
    
    Args:
        buckets: Dictionary of bucket thresholds to counts
        value: Value to categorize into buckets
    """
    # Update all buckets that the value falls into (cumulative)
    for threshold_str in buckets.keys():
        if threshold_str == "+Inf":
            buckets[threshold_str] += 1
        else:
            threshold = float(threshold_str)
            if value <= threshold:
                buckets[threshold_str] += 1


def get_queue_depth_metrics(job_queue=None) -> Dict[str, int]:
    """
    Get current queue depth metrics for monitoring.
    
    Args:
        job_queue: Optional JobQueue instance to use
    
    Returns:
        Dictionary with queue depth information
    """
    try:
        if not job_queue:
            job_queue = get_job_queue()
        queue_stats = job_queue.get_queue_stats()
        
        return {
            "total_depth": queue_stats.get("queued_jobs", 0) + queue_stats.get("started_jobs", 0),
            "queued": queue_stats.get("queued_jobs", 0),
            "active": queue_stats.get("started_jobs", 0),
            "finished": queue_stats.get("finished_jobs", 0),
            "failed": queue_stats.get("failed_jobs", 0),
            "deferred": queue_stats.get("deferred_jobs", 0),
            "scheduled": queue_stats.get("scheduled_jobs", 0)
        }
    except Exception as e:
        logger.error(f"Failed to get queue depth metrics: {e}")
        return {
            "total_depth": 0,
            "queued": 0,
            "active": 0,
            "finished": 0,
            "failed": 0,
            "deferred": 0,
            "scheduled": 0
        }


def get_system_health_metrics(redis_manager=None, job_queue=None) -> Dict[str, Dict[str, Any]]:
    """
    Get comprehensive system health metrics for monitoring.
    
    Args:
        redis_manager: Optional RedisConnectionManager instance
        job_queue: Optional JobQueue instance
    
    Returns:
        Dictionary with system component health and performance metrics
    """
    health_metrics = {
        "components": {},
        "performance": {},
        "capacity": {}
    }
    
    try:
        # Component health checks
        if not redis_manager:
            redis_manager = get_redis_manager()
        health_metrics["components"]["redis"] = {
            "healthy": redis_manager.test_connection(),
            "last_check": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        health_metrics["components"]["redis"] = {
            "healthy": False,
            "error": str(e),
            "last_check": datetime.now(timezone.utc).isoformat()
        }
    
    try:
        # GCS health check would need config access
        health_metrics["components"]["gcs"] = {
            "healthy": True,  # Placeholder - would need async context
            "last_check": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        health_metrics["components"]["gcs"] = {
            "healthy": False,
            "error": str(e),
            "last_check": datetime.now(timezone.utc).isoformat()
        }
    
    # Performance metrics
    uptime = time.time() - _startup_time
    total_requests = _metrics_storage["total_request_count"]
    total_jobs = _metrics_storage["total_jobs_processed"]
    
    health_metrics["performance"] = {
        "uptime_seconds": int(uptime),
        "requests_per_second": round(total_requests / uptime, 2) if uptime > 0 else 0.0,
        "jobs_per_hour": round(total_jobs * 3600 / uptime, 2) if uptime > 0 else 0.0,
        "average_request_latency": round(_metrics_storage["total_request_duration"] / total_requests, 3) if total_requests > 0 else 0.0,
        "average_job_duration": round(_metrics_storage["total_processing_time"] / total_jobs, 2) if total_jobs > 0 else 0.0
    }
    
    # Capacity metrics
    queue_metrics = get_queue_depth_metrics(job_queue)
    health_metrics["capacity"] = {
        "queue_depth": queue_metrics["total_depth"],
        "queue_utilization": min(queue_metrics["total_depth"] / 100, 1.0),  # Assume 100 is max capacity
        "active_jobs": queue_metrics["active"],
        "queue_backlog": queue_metrics["queued"]
    }
    
    return health_metrics


def reset_metrics():
    """Reset metrics storage (useful for testing)."""
    global _startup_time
    _startup_time = time.time()
    _metrics_storage.update({
        "total_jobs_processed": 0,
        "total_request_count": 0,
        "total_processing_time": 0.0,
        "total_request_duration": 0.0,
        "request_latency_buckets": {
            "0.1": 0, "0.5": 0, "1.0": 0, "2.5": 0, "5.0": 0, "10.0": 0, "+Inf": 0
        },
        "job_duration_buckets": {
            "30": 0, "60": 0, "300": 0, "600": 0, "1800": 0, "3600": 0, "+Inf": 0
        },
        "status_code_counts": {},
        "last_reset_time": _startup_time
    })
    logger.info("Enhanced metrics storage reset")