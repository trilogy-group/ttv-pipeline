"""
Pydantic models for API request/response validation and job management.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class JobStatus(str, Enum):
    """Job status enumeration"""
    QUEUED = "queued"
    STARTED = "started"
    PROGRESS = "progress"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELED = "canceled"


class JobCreateRequest(BaseModel):
    """Request model for job creation"""
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Text prompt for video generation"
    )
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        # Remove excessive whitespace
        v = ' '.join(v.split())
        if not v.strip():
            raise ValueError('Prompt cannot be empty or only whitespace')
        return v


class JobCreateResponse(BaseModel):
    """Response model for job creation"""
    id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(default=JobStatus.QUEUED, description="Initial job status")
    created_at: datetime = Field(..., description="Job creation timestamp")


class JobStatusResponse(BaseModel):
    """Response model for job status"""
    id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: int = Field(ge=0, le=100, description="Progress percentage (0-100)")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    finished_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    gcs_uri: Optional[str] = Field(None, description="GCS URI of generated video (when finished)")
    error: Optional[str] = Field(None, description="Error message (when failed)")


class ArtifactResponse(BaseModel):
    """Response model for artifact URL"""
    gcs_uri: str = Field(..., description="GCS URI of the artifact")
    url: str = Field(..., description="Signed HTTPS URL for download")
    expires_in: int = Field(..., description="URL expiration in seconds")


class LogsResponse(BaseModel):
    """Response model for job logs"""
    lines: List[str] = Field(..., description="Recent log lines")
    
    @field_validator('lines')
    @classmethod
    def validate_lines(cls, v):
        # Limit number of log lines returned
        if len(v) > 1000:
            return v[-1000:]  # Return last 1000 lines
        return v


class JobCancelResponse(BaseModel):
    """Response model for job cancellation"""
    id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Job status after cancellation request")
    message: str = Field(..., description="Cancellation status message")


class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request correlation ID")
    retryable: Optional[bool] = Field(None, description="Whether the error is retryable")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Health status (healthy/unhealthy/ready/not_ready)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Check timestamp")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Component health status")


class MetricsResponse(BaseModel):
    """Metrics response model"""
    active_jobs: int = Field(..., description="Number of active jobs")
    queued_jobs: int = Field(..., description="Number of queued jobs")
    total_jobs_processed: int = Field(..., description="Total jobs processed since startup")
    average_processing_time: float = Field(..., description="Average job processing time in seconds")
    uptime_seconds: int = Field(..., description="Server uptime in seconds")


class EnhancedMetricsResponse(BaseModel):
    """Enhanced metrics response model with detailed observability data"""
    # Basic metrics (backward compatibility)
    active_jobs: int = Field(..., description="Number of active jobs")
    queued_jobs: int = Field(..., description="Number of queued jobs") 
    total_jobs_processed: int = Field(..., description="Total jobs processed since startup")
    average_processing_time: float = Field(..., description="Average job processing time in seconds")
    uptime_seconds: int = Field(..., description="Server uptime in seconds")
    
    # Enhanced metrics
    total_requests: int = Field(..., description="Total API requests processed")
    average_request_latency: float = Field(..., description="Average request latency in seconds")
    requests_per_second: float = Field(..., description="Request throughput rate")
    jobs_per_hour: float = Field(..., description="Job processing throughput rate")
    queue_depth: int = Field(..., description="Total jobs in processing pipeline")
    status_code_distribution: Dict[str, int] = Field(..., description="HTTP status code distribution")


class SystemHealthMetrics(BaseModel):
    """System health metrics model"""
    timestamp: str = Field(..., description="Metrics collection timestamp")
    version: str = Field(..., description="API version")
    system_health: Dict[str, Any] = Field(..., description="Component health status")
    queue_metrics: Dict[str, int] = Field(..., description="Queue depth and status metrics")
    histogram_data: Dict[str, Dict[str, int]] = Field(..., description="Latency and duration histograms")
    throughput_metrics: Dict[str, float] = Field(..., description="Request and job throughput metrics")
    latency_metrics: Dict[str, float] = Field(..., description="Latency percentile metrics")
    error_metrics: Dict[str, Any] = Field(..., description="Error rate and failure metrics")


# Internal models for job management

class JobData(BaseModel):
    """Internal job data model for Redis storage"""
    id: str
    status: JobStatus
    progress: int = 0
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    prompt: str
    config: Dict[str, Any] = Field(default_factory=dict)
    gcs_uri: Optional[str] = None
    error: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    
    def to_status_response(self) -> JobStatusResponse:
        """Convert to API status response"""
        return JobStatusResponse(
            id=self.id,
            status=self.status,
            progress=self.progress,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            gcs_uri=self.gcs_uri,
            error=self.error
        )
    
    def add_log(self, message: str, max_logs: int = 1000):
        """Add a log message, keeping only the most recent entries"""
        timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        
        # Keep only the most recent logs
        if len(self.logs) > max_logs:
            self.logs = self.logs[-max_logs:]


class ConfigMergeRequest(BaseModel):
    """Request for configuration merging (internal use)"""
    base_config: Dict[str, Any]
    cli_args: Optional[Dict[str, Any]] = None
    http_overrides: Optional[Dict[str, Any]] = None
    
    model_config = {"arbitrary_types_allowed": True}