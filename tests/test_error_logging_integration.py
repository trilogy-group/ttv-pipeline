"""
Integration tests for error handling and logging functionality.

This module tests the complete error handling and logging pipeline
including structured logging, credential redaction, and correlation IDs.
"""

import json
import logging
from io import StringIO
from unittest.mock import patch
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from api.exceptions import JobNotFoundError, ValidationError
from api.logging_config import setup_structured_logging, CredentialRedactingFormatter
from api.main import setup_exception_handlers
from api.middleware import RequestLoggingMiddleware


class TestErrorHandlingIntegration:
    """Test complete error handling pipeline"""
    
    def setup_method(self):
        """Set up test FastAPI app with full error handling"""
        self.app = FastAPI()
        
        # Add middleware
        self.app.add_middleware(RequestLoggingMiddleware)
        
        # Add exception handlers
        setup_exception_handlers(self.app)
        
        # Add test endpoints
        @self.app.get("/test-job-not-found")
        async def test_job_not_found():
            raise JobNotFoundError("job-123")
        
        @self.app.get("/test-validation-error")
        async def test_validation_error():
            raise ValidationError("Invalid input", field="prompt")
        
        @self.app.get("/test-unhandled-error")
        async def test_unhandled_error():
            raise ValueError("Unexpected error")
        
        @self.app.get("/test-success")
        async def test_success():
            return {"message": "success"}
        
        self.client = TestClient(self.app)
    
    def test_api_exception_with_correlation_id(self):
        """Test that API exceptions include correlation IDs"""
        response = self.client.get("/test-job-not-found")
        
        assert response.status_code == 404
        
        error_data = response.json()
        assert error_data['error'] == 'JobNotFound'
        assert error_data['message'] == 'Job job-123 not found'
        assert 'request_id' in error_data
        assert 'timestamp' in error_data
        
        # Check correlation ID in headers
        assert 'X-Correlation-ID' in response.headers
        assert error_data['request_id'] == response.headers['X-Correlation-ID']
    
    def test_validation_error_with_details(self):
        """Test that validation errors include field details"""
        response = self.client.get("/test-validation-error")
        
        assert response.status_code == 400
        
        error_data = response.json()
        assert error_data['error'] == 'ValidationError'
        assert error_data['details']['field'] == 'prompt'
        assert error_data['retryable'] is False
    
    def test_unhandled_exception_handling(self):
        """Test that unhandled exceptions are properly caught and logged"""
        response = self.client.get("/test-unhandled-error")
        
        assert response.status_code == 500
        
        error_data = response.json()
        assert error_data['error'] == 'InternalServerError'
        assert error_data['message'] == 'An internal server error occurred'
        assert 'request_id' in error_data
    
    def test_successful_request_with_correlation_id(self):
        """Test that successful requests also get correlation IDs"""
        response = self.client.get("/test-success")
        
        assert response.status_code == 200
        assert 'X-Correlation-ID' in response.headers
        
        data = response.json()
        assert data['message'] == 'success'


class TestStructuredLoggingIntegration:
    """Test structured logging integration"""
    
    def test_credential_redaction_in_logs(self):
        """Test that credentials are redacted in actual log output"""
        # Capture log output
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(CredentialRedactingFormatter())
        
        logger = logging.getLogger('test_logger')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Log messages with sensitive data
        logger.info("API key: abc123xyz")
        logger.info("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
        logger.info("GCS URL: https://storage.googleapis.com/bucket/file?X-Goog-Signature=secret123")
        
        # Get log output
        log_output = log_stream.getvalue()
        log_lines = log_output.strip().split('\n')
        
        # Parse JSON logs and check redaction
        for line in log_lines:
            if line.strip():
                log_data = json.loads(line)
                message = log_data['message']
                
                # Check that sensitive data is redacted
                assert '[REDACTED]' in message
                assert 'abc123xyz' not in message
                assert 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9' not in message
                assert 'secret123' not in message
    
    def test_structured_json_format(self):
        """Test that logs are properly formatted as JSON"""
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(CredentialRedactingFormatter())
        
        logger = logging.getLogger('test_logger')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Log a message with extra data
        logger.info(
            "Test message",
            extra={
                'extra': {
                    'correlation_id': 'test-123',
                    'event': 'test_event',
                    'user_id': 'user-456'
                }
            }
        )
        
        # Parse log output
        log_output = log_stream.getvalue().strip()
        log_data = json.loads(log_output)
        
        # Check required fields
        assert log_data['level'] == 'INFO'
        assert log_data['message'] == 'Test message'
        assert log_data['correlation_id'] == 'test-123'
        assert log_data['event'] == 'test_event'
        assert log_data['user_id'] == 'user-456'
        assert 'timestamp' in log_data
        assert 'logger' in log_data


class TestErrorContextAndTracing:
    """Test error context and request tracing"""
    
    def test_exception_context_preservation(self):
        """Test that exception context is preserved through the pipeline"""
        from api.exceptions import APIException
        
        # Create exception with context
        exc = APIException(
            message="Test error",
            status_code=500,
            correlation_id="test-123"
        )
        exc.add_context("job_id", "job-456")
        exc.add_context("operation", "video_generation")
        
        # Convert to dict (simulating serialization)
        error_dict = exc.to_dict()
        
        # Check that all context is preserved
        assert error_dict['request_id'] == 'test-123'
        assert error_dict['details']['job_id'] == 'job-456'
        assert error_dict['details']['operation'] == 'video_generation'
        assert error_dict['retryable'] is False  # Default value
    
    def test_correlation_id_propagation(self):
        """Test that correlation IDs are properly propagated"""
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)
        setup_exception_handlers(app)
        
        @app.get("/test")
        async def test_endpoint(request: Request):
            # Access correlation ID from request state
            correlation_id = getattr(request.state, 'correlation_id', None)
            if not correlation_id:
                raise ValueError("No correlation ID found")
            
            # Raise exception with correlation ID
            exc = JobNotFoundError("job-123")
            exc.correlation_id = correlation_id
            raise exc
        
        client = TestClient(app)
        response = client.get("/test")
        
        # Check that correlation ID is consistent
        error_data = response.json()
        header_correlation_id = response.headers.get('X-Correlation-ID')
        
        assert error_data['request_id'] == header_correlation_id


if __name__ == "__main__":
    pytest.main([__file__])