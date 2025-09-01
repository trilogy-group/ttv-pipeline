"""
Custom exceptions for the API server.

This module defines the exception hierarchy used throughout the API server
for consistent error handling and HTTP status code mapping with comprehensive
error context and proper logging integration.
"""

import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone


class APIException(Exception):
    """Base exception for all API-related errors"""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
        correlation_id: Optional[str] = None,
        retryable: bool = False
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        self.error_code = error_code or self.__class__.__name__
        self.correlation_id = correlation_id
        self.retryable = retryable
        self.timestamp = datetime.now(timezone.utc)
        self.stack_trace = traceback.format_exc() if traceback.format_exc() != 'NoneType: None\n' else None
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization"""
        result = {
            'error': self.error_code,
            'message': self.message,
            'status_code': self.status_code,
            'timestamp': self.timestamp.isoformat(),
            'retryable': self.retryable
        }
        
        if self.details:
            result['details'] = self.details
        
        if self.correlation_id:
            result['request_id'] = self.correlation_id
        
        return result
    
    def add_context(self, key: str, value: Any) -> 'APIException':
        """Add additional context to the exception"""
        self.details[key] = value
        return self
    
    def with_correlation_id(self, correlation_id: str) -> 'APIException':
        """Set correlation ID for the exception"""
        self.correlation_id = correlation_id
        return self


class ValidationError(APIException):
    """Input validation error"""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        field: Optional[str] = None,
        validation_errors: Optional[List[Dict[str, Any]]] = None
    ):
        enhanced_details = details or {}
        if field:
            enhanced_details['field'] = field
        if validation_errors:
            enhanced_details['validation_errors'] = validation_errors
        
        super().__init__(
            message, 
            400, 
            enhanced_details,
            error_code="ValidationError",
            retryable=False
        )


class AuthenticationError(APIException):
    """Authentication error"""
    
    def __init__(
        self, 
        message: str = "Authentication required",
        auth_scheme: Optional[str] = None
    ):
        details = {}
        if auth_scheme:
            details['auth_scheme'] = auth_scheme
        
        super().__init__(
            message, 
            401, 
            details,
            error_code="AuthenticationError",
            retryable=False
        )


class AuthorizationError(APIException):
    """Authorization error"""
    
    def __init__(
        self, 
        message: str = "Insufficient permissions",
        required_permission: Optional[str] = None
    ):
        details = {}
        if required_permission:
            details['required_permission'] = required_permission
        
        super().__init__(
            message, 
            403, 
            details,
            error_code="AuthorizationError",
            retryable=False
        )


class JobNotFoundError(APIException):
    """Job not found error"""
    
    def __init__(self, job_id: str):
        super().__init__(
            f"Job {job_id} not found", 
            404,
            {"job_id": job_id},
            error_code="JobNotFound",
            retryable=False
        )


class ArtifactNotReadyError(APIException):
    """Artifact not ready error"""
    
    def __init__(self, job_id: str, current_status: str):
        super().__init__(
            f"Artifact for job {job_id} not ready (status: {current_status})",
            404,
            {"job_id": job_id, "current_status": current_status},
            error_code="ArtifactNotReady",
            retryable=True
        )


class RateLimitError(APIException):
    """Rate limit exceeded error"""
    
    def __init__(
        self, 
        message: str = "Rate limit exceeded",
        limit: Optional[int] = None,
        window_seconds: Optional[int] = None,
        retry_after: Optional[int] = None
    ):
        details = {}
        if limit:
            details['limit'] = limit
        if window_seconds:
            details['window_seconds'] = window_seconds
        if retry_after:
            details['retry_after'] = retry_after
        
        super().__init__(
            message, 
            429, 
            details,
            error_code="RateLimitExceeded",
            retryable=True
        )


class ConfigurationError(APIException):
    """Configuration error"""
    
    def __init__(
        self, 
        message: str,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None
    ):
        details = {}
        if config_key:
            details['config_key'] = config_key
        if config_file:
            details['config_file'] = config_file
        
        super().__init__(
            f"Configuration error: {message}", 
            500, 
            details,
            error_code="ConfigurationError",
            retryable=False
        )


class RedisConnectionError(APIException):
    """Redis connection error"""
    
    def __init__(
        self, 
        message: str = "Redis connection failed",
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None
    ):
        details = {}
        if redis_host:
            details['redis_host'] = redis_host
        if redis_port:
            details['redis_port'] = redis_port
        
        super().__init__(
            f"Redis error: {message}", 
            503, 
            details,
            error_code="RedisConnectionError",
            retryable=True
        )


class GCSError(APIException):
    """Google Cloud Storage error"""
    
    def __init__(
        self, 
        message: str,
        bucket: Optional[str] = None,
        object_path: Optional[str] = None,
        operation: Optional[str] = None
    ):
        details = {}
        if bucket:
            details['bucket'] = bucket
        if object_path:
            details['object_path'] = object_path
        if operation:
            details['operation'] = operation
        
        super().__init__(
            f"GCS error: {message}", 
            503, 
            details,
            error_code="GCSError",
            retryable=True
        )


class WorkerUnavailableError(APIException):
    """No workers available error"""
    
    def __init__(
        self, 
        message: str = "No workers available to process jobs",
        queue_depth: Optional[int] = None
    ):
        details = {}
        if queue_depth is not None:
            details['queue_depth'] = queue_depth
        
        super().__init__(
            message, 
            503, 
            details,
            error_code="WorkerUnavailable",
            retryable=True
        )


class JobCancellationError(APIException):
    """Job cancellation error"""
    
    def __init__(self, job_id: str, reason: str, current_status: Optional[str] = None):
        details = {"job_id": job_id, "reason": reason}
        if current_status:
            details['current_status'] = current_status
        
        super().__init__(
            f"Cannot cancel job {job_id}: {reason}",
            409,
            details,
            error_code="JobCancellationError",
            retryable=False
        )


class PipelineExecutionError(APIException):
    """Pipeline execution error"""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
        exit_code: Optional[int] = None
    ):
        enhanced_details = details or {}
        if job_id:
            enhanced_details['job_id'] = job_id
        if exit_code is not None:
            enhanced_details['exit_code'] = exit_code
        
        super().__init__(
            f"Pipeline execution failed: {message}", 
            500, 
            enhanced_details,
            error_code="PipelineExecutionError",
            retryable=False
        )


# Additional exception types for comprehensive error handling

class TimeoutError(APIException):
    """Request timeout error"""
    
    def __init__(
        self, 
        message: str = "Request timeout",
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None
    ):
        details = {}
        if timeout_seconds:
            details['timeout_seconds'] = timeout_seconds
        if operation:
            details['operation'] = operation
        
        super().__init__(
            message, 
            408, 
            details,
            error_code="RequestTimeout",
            retryable=True
        )


class ConflictError(APIException):
    """Resource conflict error"""
    
    def __init__(
        self, 
        message: str,
        resource_id: Optional[str] = None,
        conflict_type: Optional[str] = None
    ):
        details = {}
        if resource_id:
            details['resource_id'] = resource_id
        if conflict_type:
            details['conflict_type'] = conflict_type
        
        super().__init__(
            message, 
            409, 
            details,
            error_code="ConflictError",
            retryable=False
        )


class PayloadTooLargeError(APIException):
    """Payload too large error"""
    
    def __init__(
        self, 
        message: str = "Request payload too large",
        max_size: Optional[int] = None,
        actual_size: Optional[int] = None
    ):
        details = {}
        if max_size:
            details['max_size'] = max_size
        if actual_size:
            details['actual_size'] = actual_size
        
        super().__init__(
            message, 
            413, 
            details,
            error_code="PayloadTooLarge",
            retryable=False
        )


class UnsupportedMediaTypeError(APIException):
    """Unsupported media type error"""
    
    def __init__(
        self, 
        message: str = "Unsupported media type",
        provided_type: Optional[str] = None,
        supported_types: Optional[List[str]] = None
    ):
        details = {}
        if provided_type:
            details['provided_type'] = provided_type
        if supported_types:
            details['supported_types'] = supported_types
        
        super().__init__(
            message, 
            415, 
            details,
            error_code="UnsupportedMediaType",
            retryable=False
        )


class ServiceUnavailableError(APIException):
    """Service unavailable error"""
    
    def __init__(
        self, 
        message: str = "Service temporarily unavailable",
        service_name: Optional[str] = None,
        retry_after: Optional[int] = None
    ):
        details = {}
        if service_name:
            details['service_name'] = service_name
        if retry_after:
            details['retry_after'] = retry_after
        
        super().__init__(
            message, 
            503, 
            details,
            error_code="ServiceUnavailable",
            retryable=True
        )