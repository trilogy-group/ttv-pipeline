"""
Structured logging configuration for the API server.

This module provides JSON-structured logging with credential redaction,
correlation ID tracking, and appropriate log levels for different components.
"""

import json
import logging
import logging.config
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set, Union


class CredentialRedactingFormatter(logging.Formatter):
    """
    Custom formatter that redacts sensitive information from log messages.
    
    Automatically detects and redacts common credential patterns including
    API keys, tokens, passwords, and signed URL parameters.
    """
    
    # Patterns for sensitive data detection
    SENSITIVE_PATTERNS = [
        # API keys and tokens
        (re.compile(r'(api[_-]?key["\s]*[:=]["\s]*)[^"\s&]+', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'(api\s+key[:\s]+)[^\s&]+', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'(token["\s]*[:=]["\s]*)[^"\s&]+', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'(bearer\s+)[^\s&]+', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'(authorization["\s]*[:=]["\s]*)[^"\s&]+', re.IGNORECASE), r'\1[REDACTED]'),
        
        # Passwords
        (re.compile(r'(password["\s]*[:=]["\s]*)[^"\s&]+', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'(passwd["\s]*[:=]["\s]*)[^"\s&]+', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'(pwd["\s]*[:=]["\s]*)[^"\s&]+', re.IGNORECASE), r'\1[REDACTED]'),
        
        # GCS signed URL parameters
        (re.compile(r'(X-Goog-Signature=)[^&\s]+', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'(X-Goog-Credential=)[^&\s]+', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'(Signature=)[^&\s]+', re.IGNORECASE), r'\1[REDACTED]'),
        
        # Generic secrets
        (re.compile(r'(secret["\s]*[:=]["\s]*)[^"\s&]+', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'(key["\s]*[:=]["\s]*)[^"\s&]{20,}', re.IGNORECASE), r'\1[REDACTED]'),
        
        # Credit card numbers (basic pattern)
        (re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'), '[REDACTED-CC]'),
        
        # Email addresses in sensitive contexts
        (re.compile(r'(email["\s]*[:=]["\s]*)[^"\s&]+@[^"\s&]+', re.IGNORECASE), r'\1[REDACTED]'),
    ]
    
    # Sensitive header names (case-insensitive)
    SENSITIVE_HEADERS = {
        'authorization', 'x-api-key', 'cookie', 'set-cookie', 
        'x-auth-token', 'x-access-token', 'x-csrf-token',
        'x-goog-iam-authorization-token', 'x-goog-iam-authority-selector'
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with credential redaction"""
        # Create a copy of the record to avoid modifying the original
        record_dict = record.__dict__.copy()
        
        # Redact message
        if hasattr(record, 'getMessage'):
            message = record.getMessage()
            record_dict['message'] = self._redact_sensitive_data(message)
        
        # Redact extra fields
        if hasattr(record, 'extra') and record.extra:
            record_dict['extra'] = self._redact_dict(record.extra)
        
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record_dict.get('message', ''),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add correlation ID if available
        correlation_id = getattr(record, 'correlation_id', None)
        if correlation_id:
            log_entry['correlation_id'] = correlation_id
        
        # Add extra fields
        extra = record_dict.get('extra', {})
        if extra:
            log_entry.update(extra)
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add stack info if present
        if record.stack_info:
            log_entry['stack_info'] = self.formatStack(record.stack_info)
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)
    
    def _redact_sensitive_data(self, text: str) -> str:
        """Redact sensitive data from text using patterns"""
        if not isinstance(text, str):
            return str(text)
        
        redacted = text
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        
        return redacted
    
    def _redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively redact sensitive data from dictionary"""
        if not isinstance(data, dict):
            return data
        
        redacted = {}
        for key, value in data.items():
            # Check if key is sensitive
            if key.lower() in self.SENSITIVE_HEADERS:
                redacted[key] = '[REDACTED]'
            elif isinstance(value, dict):
                redacted[key] = self._redact_dict(value)
            elif isinstance(value, list):
                redacted[key] = [self._redact_dict(item) if isinstance(item, dict) 
                               else self._redact_sensitive_data(str(item)) for item in value]
            elif isinstance(value, str):
                redacted[key] = self._redact_sensitive_data(value)
            else:
                redacted[key] = value
        
        return redacted


class CorrelationFilter(logging.Filter):
    """
    Filter to add correlation ID to log records.
    
    Attempts to extract correlation ID from various sources including
    request context, thread local storage, and explicit parameters.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to log record if available"""
        # Try to get correlation ID from various sources
        correlation_id = getattr(record, 'correlation_id', None)
        
        if not correlation_id:
            # Try to get from thread local storage (if implemented)
            try:
                import threading
                local_data = getattr(threading.current_thread(), 'local_data', None)
                if local_data and hasattr(local_data, 'correlation_id'):
                    correlation_id = local_data.correlation_id
            except:
                pass
        
        if correlation_id:
            record.correlation_id = correlation_id
        
        return True


def setup_structured_logging(
    level: Union[str, int] = logging.INFO,
    enable_console: bool = True,
    enable_file: bool = False,
    file_path: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Configure structured JSON logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_console: Enable console logging
        enable_file: Enable file logging
        file_path: Path to log file (required if enable_file=True)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    json_formatter = CredentialRedactingFormatter()
    
    # Create correlation filter
    correlation_filter = CorrelationFilter()
    
    # Configure handlers
    handlers = []
    
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(json_formatter)
        console_handler.addFilter(correlation_filter)
        handlers.append(console_handler)
    
    if enable_file and file_path:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            file_path, 
            maxBytes=max_bytes, 
            backupCount=backup_count
        )
        file_handler.setFormatter(json_formatter)
        file_handler.addFilter(correlation_filter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True  # Override existing configuration
    )
    
    # Set specific logger levels
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('hypercorn').setLevel(logging.WARNING)
    logging.getLogger('fastapi').setLevel(logging.INFO)
    logging.getLogger('api').setLevel(logging.INFO)
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)
    logging.getLogger('redis').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with proper configuration.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_with_correlation(
    logger: logging.Logger,
    level: int,
    message: str,
    correlation_id: Optional[str] = None,
    **kwargs
) -> None:
    """
    Log a message with correlation ID and additional context.
    
    Args:
        logger: Logger instance
        level: Log level (logging.DEBUG, INFO, etc.)
        message: Log message
        correlation_id: Request correlation ID
        **kwargs: Additional context to include in log
    """
    extra = kwargs.copy()
    if correlation_id:
        extra['correlation_id'] = correlation_id
    
    logger.log(level, message, extra={'extra': extra})


# Convenience functions for common logging patterns
def log_api_request(
    logger: logging.Logger,
    method: str,
    path: str,
    correlation_id: str,
    client_ip: str,
    user_agent: str,
    **kwargs
) -> None:
    """Log API request with standard fields"""
    log_with_correlation(
        logger, logging.INFO,
        f"API Request: {method} {path}",
        correlation_id=correlation_id,
        event="api_request",
        method=method,
        path=path,
        client_ip=client_ip,
        user_agent=user_agent,
        **kwargs
    )


def log_api_response(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    correlation_id: str,
    **kwargs
) -> None:
    """Log API response with standard fields"""
    log_with_correlation(
        logger, logging.INFO,
        f"API Response: {method} {path} -> {status_code}",
        correlation_id=correlation_id,
        event="api_response",
        method=method,
        path=path,
        status_code=status_code,
        duration_ms=duration_ms,
        **kwargs
    )


def log_api_error(
    logger: logging.Logger,
    error: Exception,
    correlation_id: str,
    method: str = None,
    path: str = None,
    **kwargs
) -> None:
    """Log API error with standard fields"""
    log_with_correlation(
        logger, logging.ERROR,
        f"API Error: {type(error).__name__}: {str(error)}",
        correlation_id=correlation_id,
        event="api_error",
        error_type=type(error).__name__,
        error_message=str(error),
        method=method,
        path=path,
        **kwargs
    )


def log_job_event(
    logger: logging.Logger,
    job_id: str,
    event: str,
    message: str,
    correlation_id: str = None,
    **kwargs
) -> None:
    """Log job-related event with standard fields"""
    log_with_correlation(
        logger, logging.INFO,
        message,
        correlation_id=correlation_id,
        event="job_event",
        job_id=job_id,
        job_event=event,
        **kwargs
    )