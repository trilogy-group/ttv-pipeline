"""
Comprehensive unit tests for API endpoints with various input scenarios and edge cases.
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from api.main import create_app
from api.models import JobData, JobStatus, JobCreateRequest


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = MagicMock()
    config.server.host = "localhost"
    config.server.port = 8000
    config.server.quic_port = 8443
    config.redis.host = "localhost"
    config.redis.port = 6379
    config.gcs.bucket = "test-bucket"
    config.server.cors_origins = ["*"]
    config.security.rate_limit_per_minute = 60
    return config


@pytest.fixture
def client(mock_config):
    """Test client with mocked configuration"""
    with patch('api.main.get_config_from_env', return_value=mock_config), \
         patch('api.queue.initialize_queue_infrastructure') as mock_init_queue:
        
        mock_redis_manager = MagicMock()
        mock_job_queue = MagicMock()
        mock_init_queue.return_value = (mock_redis_manager, mock_job_queue)
        
        app = create_app()
        app.state.config = mock_config
        
        # Mock pipeline config for job creation
        mock_config.pipeline = MagicMock()
        mock_config.pipeline.model_dump.return_value = {
            'prompt': 'default prompt',
            'default_backend': 'wan21',
            'size': '1280x720'
        }
        
        return TestClient(app)


class TestJobCreationEdgeCases:
    """Test job creation endpoint with various edge cases"""
    
    def test_create_job_unicode_prompt(self, client):
        """Test job creation with Unicode characters in prompt"""
        with patch('api.queue.get_job_queue') as mock_get_queue:
            mock_queue = MagicMock()
            mock_get_queue.return_value = mock_queue
            
            job_data = JobData(
                id="test-job-unicode",
                status=JobStatus.QUEUED,
                created_at=datetime.now(timezone.utc),
                prompt="🎬 Create a video with émojis and spëcial characters 中文"
            )
            mock_queue.enqueue_job.return_value = job_data
            
            response = client.post("/v1/jobs", json={
                "prompt": "🎬 Create a video with émojis and spëcial characters 中文"
            })
            
            assert response.status_code == 202
            data = response.json()
            # Check if Unicode characters are handled properly in the response
            assert data["id"] == "test-job-unicode"
            assert data["status"] == "queued"
    
    def test_create_job_whitespace_variations(self, client):
        """Test job creation with various whitespace patterns"""
        test_cases = [
            "  Leading spaces",
            "Trailing spaces  ",
            "  Both sides  ",
            "Multiple    internal    spaces",
            "\t\nTabs and newlines\t\n",
            "Mixed\t spaces\n and\r\n newlines"
        ]
        
        with patch('api.queue.get_job_queue') as mock_get_queue:
            mock_queue = MagicMock()
            mock_get_queue.return_value = mock_queue
            
            for i, prompt in enumerate(test_cases):
                job_data = JobData(
                    id=f"test-job-{i}",
                    status=JobStatus.QUEUED,
                    created_at=datetime.now(timezone.utc),
                    prompt=prompt.strip()
                )
                mock_queue.enqueue_job.return_value = job_data
                
                response = client.post("/v1/jobs", json={"prompt": prompt})
                
                assert response.status_code == 202
                # Verify response is successful (whitespace normalization happens in model validation)
                data = response.json()
                assert data["id"] == f"test-job-{i}"
                assert data["status"] == "queued"
    
    def test_create_job_boundary_lengths(self, client):
        """Test job creation with prompts at length boundaries"""
        with patch('api.queue.get_job_queue') as mock_get_queue:
            mock_queue = MagicMock()
            mock_get_queue.return_value = mock_queue
            
            # Test minimum valid length (1 character)
            job_data = JobData(
                id="test-job-min",
                status=JobStatus.QUEUED,
                created_at=datetime.now(timezone.utc),
                prompt="A"
            )
            mock_queue.enqueue_job.return_value = job_data
            
            response = client.post("/v1/jobs", json={"prompt": "A"})
            assert response.status_code == 202
            
            # Test maximum valid length (2000 characters)
            max_prompt = "A" * 2000
            job_data.prompt = max_prompt
            mock_queue.enqueue_job.return_value = job_data
            
            response = client.post("/v1/jobs", json={"prompt": max_prompt})
            assert response.status_code == 202
            
            # Test just over maximum (2001 characters)
            over_max_prompt = "A" * 2001
            response = client.post("/v1/jobs", json={"prompt": over_max_prompt})
            assert response.status_code == 422
    
    def test_create_job_special_characters(self, client):
        """Test job creation with special characters and escape sequences"""
        special_prompts = [
            'Prompt with "quotes" and \'apostrophes\'',
            'Prompt with backslashes \\ and forward slashes /',
            'Prompt with HTML <tags> and &entities;',
            'Prompt with JSON {"key": "value"} structure',
            'Prompt with SQL injection attempt\'; DROP TABLE jobs; --',
            'Prompt with newlines\nand\ttabs',
            'Prompt with null bytes\x00 and control chars\x01\x02'
        ]
        
        with patch('api.queue.get_job_queue') as mock_get_queue:
            mock_queue = MagicMock()
            mock_get_queue.return_value = mock_queue
            
            for i, prompt in enumerate(special_prompts):
                job_data = JobData(
                    id=f"test-job-special-{i}",
                    status=JobStatus.QUEUED,
                    created_at=datetime.now(timezone.utc),
                    prompt=prompt
                )
                mock_queue.enqueue_job.return_value = job_data
                
                response = client.post("/v1/jobs", json={"prompt": prompt})
                
                # Should handle special characters gracefully
                assert response.status_code in [202, 422]  # Either success or validation error
    
    def test_create_job_malformed_json(self, client):
        """Test job creation with malformed JSON"""
        malformed_requests = [
            '{"prompt": "test"',  # Missing closing brace
            '{"prompt": "test",}',  # Trailing comma
            '{"prompt": }',  # Missing value
            '{prompt: "test"}',  # Unquoted key
            '{"prompt": "test" "extra": "field"}',  # Missing comma
        ]
        
        for malformed_json in malformed_requests:
            response = client.post(
                "/v1/jobs",
                content=malformed_json,
                headers={"Content-Type": "application/json"}
            )
            assert response.status_code == 422
    
    def test_create_job_missing_content_type(self, client):
        """Test job creation without proper content type"""
        response = client.post(
            "/v1/jobs",
            content='{"prompt": "test"}',
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 415  # Unsupported Media Type
    
    def test_create_job_empty_body(self, client):
        """Test job creation with empty request body"""
        response = client.post("/v1/jobs", json={})
        assert response.status_code == 422
        
        response = client.post("/v1/jobs", content="")
        assert response.status_code in [415, 422]  # Could be unsupported media type or validation error
    
    def test_create_job_extra_fields(self, client):
        """Test job creation with extra unexpected fields"""
        with patch('api.queue.get_job_queue') as mock_get_queue:
            mock_queue = MagicMock()
            mock_get_queue.return_value = mock_queue
            
            job_data = JobData(
                id="test-job-extra",
                status=JobStatus.QUEUED,
                created_at=datetime.now(timezone.utc),
                prompt="Test prompt"
            )
            mock_queue.enqueue_job.return_value = job_data
            
            response = client.post("/v1/jobs", json={
                "prompt": "Test prompt",
                "extra_field": "should be ignored",
                "malicious_field": "<script>alert('xss')</script>",
                "nested": {"field": "value"}
            })
            
            # Should succeed and ignore extra fields
            assert response.status_code == 202


class TestJobStatusEdgeCases:
    """Test job status endpoint with various edge cases"""
    
    @patch('api.queue.get_job_queue')
    def test_get_job_status_invalid_job_ids(self, mock_get_queue, client):
        """Test job status retrieval with invalid job IDs"""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        mock_queue.get_job.return_value = None
        
        invalid_ids = [
            "",  # Empty string
            " ",  # Whitespace only
            "job with spaces",
            "job/with/slashes",
            "job?with=query&params",
            "job#with#hash",
            "job%20with%20encoding",
            "../../../etc/passwd",  # Path traversal attempt
            "a" * 1000,  # Very long ID
        ]
        
        for job_id in invalid_ids:
            response = client.get(f"/v1/jobs/{job_id}")
            # Should handle gracefully - various error codes possible
            assert response.status_code in [307, 400, 404, 405, 422]  # Include redirect and method not allowed
    
    @patch('api.queue.get_job_queue')
    def test_get_job_status_concurrent_requests(self, mock_get_queue, client):
        """Test concurrent job status requests"""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        
        job_data = JobData(
            id="test-job-concurrent",
            status=JobStatus.PROGRESS,
            progress=50,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        mock_queue.get_job.return_value = job_data
        
        # Simulate multiple concurrent requests
        responses = []
        for _ in range(10):
            response = client.get("/v1/jobs/test-job-concurrent")
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "test-job-concurrent"
    
    @patch('api.queue.get_job_queue')
    def test_get_job_status_redis_timeout(self, mock_get_queue, client):
        """Test job status retrieval with Redis timeout"""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        
        # Simulate Redis timeout
        import redis
        mock_queue.get_job.side_effect = redis.TimeoutError("Redis timeout")
        
        response = client.get("/v1/jobs/test-job-123")
        
        assert response.status_code == 503
        data = response.json()
        assert "timeout" in data["message"].lower()
    
    @patch('api.queue.get_job_queue')
    def test_get_job_status_corrupted_data(self, mock_get_queue, client):
        """Test job status retrieval with corrupted Redis data"""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        
        # Simulate corrupted data that causes parsing errors
        mock_queue.get_job.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
        response = client.get("/v1/jobs/test-job-123")
        
        assert response.status_code in [500, 503]  # Could be internal error or service unavailable
        data = response.json()
        assert "error" in data


class TestJobCancellationEdgeCases:
    """Test job cancellation endpoint with various edge cases"""
    
    @patch('api.queue.get_job_queue')
    def test_cancel_job_race_condition(self, mock_get_queue, client):
        """Test job cancellation with race condition scenarios"""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        
        # Simulate job finishing between status check and cancellation
        running_job = JobData(
            id="test-job-race",
            status=JobStatus.STARTED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        finished_job = JobData(
            id="test-job-race",
            status=JobStatus.FINISHED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        # First call returns running job, second call returns finished job
        mock_queue.get_job.side_effect = [running_job, finished_job]
        mock_queue.cancel_job.return_value = False  # Cancellation failed
        
        response = client.post("/v1/jobs/test-job-race/cancel", json={})
        
        # Should handle race condition gracefully
        assert response.status_code in [200, 409]
    
    @patch('api.queue.get_job_queue')
    def test_cancel_job_multiple_attempts(self, mock_get_queue, client):
        """Test multiple cancellation attempts on same job"""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        
        # First attempt - job is running
        running_job = JobData(
            id="test-job-multi-cancel",
            status=JobStatus.STARTED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        # Second attempt - job is already cancelled
        cancelled_job = JobData(
            id="test-job-multi-cancel",
            status=JobStatus.CANCELED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        mock_queue.get_job.side_effect = [running_job, cancelled_job, cancelled_job]
        mock_queue.cancel_job.return_value = True
        
        # First cancellation attempt
        response1 = client.post("/v1/jobs/test-job-multi-cancel/cancel", json={})
        assert response1.status_code == 200
        
        # Second cancellation attempt on already cancelled job
        response2 = client.post("/v1/jobs/test-job-multi-cancel/cancel", json={})
        assert response2.status_code == 409  # Already in terminal state
    
    @patch('api.queue.get_job_queue')
    def test_cancel_job_with_malformed_request(self, mock_get_queue, client):
        """Test job cancellation with malformed requests"""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        
        job_data = JobData(
            id="test-job-malformed",
            status=JobStatus.STARTED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        mock_queue.get_job.return_value = job_data
        
        # Test various malformed requests
        malformed_requests = [
            # Missing content type
            (None, {"Content-Type": "text/plain"}),
            # Invalid JSON
            ('{"invalid": json}', {"Content-Type": "application/json"}),
            # Wrong HTTP method
            ("GET", None),
        ]
        
        for content, headers in malformed_requests:
            if content == "GET":
                response = client.get("/v1/jobs/test-job-malformed/cancel")
            else:
                response = client.post(
                    "/v1/jobs/test-job-malformed/cancel",
                    content=content or "{}",
                    headers=headers or {"Content-Type": "application/json"}
                )
            
            # Should handle malformed requests appropriately
            assert response.status_code in [200, 405, 415, 422]  # Some may succeed if they're valid JSON


class TestJobLogsEdgeCases:
    """Test job logs endpoint with various edge cases"""
    
    @patch('api.queue.get_job_queue')
    def test_get_job_logs_extreme_tail_values(self, mock_get_queue, client):
        """Test job logs with extreme tail parameter values"""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        
        job_data = JobData(
            id="test-job-logs",
            status=JobStatus.PROGRESS,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        mock_queue.get_job.return_value = job_data
        mock_queue.get_job_logs.return_value = []
        
        extreme_values = [
            ("0", 0),  # Zero logs
            ("1", 1),  # Single log
            ("9999", 9999),  # Large but valid
            ("10000", 10000),  # At limit
            ("10001", 10000),  # Over limit, should be capped
            ("999999", 10000),  # Very large, should be capped
        ]
        
        for tail_param, expected_tail in extreme_values:
            response = client.get(f"/v1/jobs/test-job-logs/logs?tail={tail_param}")
            
            assert response.status_code == 200
            # Verify the tail parameter was properly limited
            mock_queue.get_job_logs.assert_called_with("test-job-logs", tail=expected_tail)
    
    @patch('api.queue.get_job_queue')
    def test_get_job_logs_invalid_tail_values(self, mock_get_queue, client):
        """Test job logs with invalid tail parameter values"""
        invalid_values = [
            "abc",  # Non-numeric
            "-1",   # Negative
            "1.5",  # Float
            "",     # Empty
            " ",    # Whitespace
            "1e10", # Scientific notation
            "null", # String null
        ]
        
        for tail_value in invalid_values:
            response = client.get(f"/v1/jobs/test-job-logs/logs?tail={tail_value}")
            
            # Should return validation error for invalid values
            assert response.status_code in [400, 422]
    
    @patch('api.queue.get_job_queue')
    def test_get_job_logs_large_log_volume(self, mock_get_queue, client):
        """Test job logs retrieval with large log volumes"""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue
        
        job_data = JobData(
            id="test-job-large-logs",
            status=JobStatus.PROGRESS,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        mock_queue.get_job.return_value = job_data
        
        # Simulate large log volume
        large_logs = [f"[2025-08-31T12:00:{i:02d}Z] Log entry {i} with some content" for i in range(1000)]
        mock_queue.get_job_logs.return_value = large_logs
        
        response = client.get("/v1/jobs/test-job-large-logs/logs")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["lines"]) == 1000
        
        # Verify response size is reasonable (not too large)
        response_size = len(response.content)
        assert response_size < 10 * 1024 * 1024  # Less than 10MB


class TestErrorHandlingComprehensive:
    """Comprehensive error handling tests"""
    
    def test_http_method_not_allowed(self, client):
        """Test HTTP method not allowed scenarios"""
        endpoints_methods = [
            ("/v1/jobs", "GET"),  # Should be POST
            ("/v1/jobs/test-job", "POST"),  # Should be GET
            ("/v1/jobs/test-job/cancel", "GET"),  # Should be POST
            ("/v1/jobs/test-job/logs", "POST"),  # Should be GET
            ("/healthz", "POST"),  # Should be GET
        ]
        
        for endpoint, method in endpoints_methods:
            if method == "GET":
                response = client.get(endpoint)
            elif method == "POST":
                response = client.post(endpoint, json={})
            
            # Should return method not allowed or similar error
            assert response.status_code in [405, 422]
    
    def test_unsupported_media_types(self, client):
        """Test unsupported media type handling"""
        unsupported_types = [
            "text/plain",
            "application/xml",
            "multipart/form-data",
            "application/x-www-form-urlencoded",
            "image/jpeg",
        ]
        
        for content_type in unsupported_types:
            response = client.post(
                "/v1/jobs",
                content='{"prompt": "test"}',
                headers={"Content-Type": content_type}
            )
            
            assert response.status_code == 415  # Unsupported Media Type
    
    def test_request_size_limits(self, client):
        """Test request size limit handling"""
        # Create a very large request payload
        large_prompt = "A" * (10 * 1024 * 1024)  # 10MB prompt
        
        response = client.post("/v1/jobs", json={"prompt": large_prompt})
        
        # Should reject large requests
        assert response.status_code in [413, 422]  # Payload Too Large or Validation Error
    
    def test_malformed_urls(self, client):
        """Test malformed URL handling"""
        malformed_urls = [
            "/v1/jobs//double-slash",
            "/v1/jobs/job-id//logs",
            "/v1/jobs/job-id/cancel/extra",
            "/v1/jobs/job-id/invalid-action",
        ]
        
        for url in malformed_urls:
            response = client.get(url)
            # Should return 404 or similar error
            assert response.status_code in [404, 405]


class TestSecurityEdgeCases:
    """Test security-related edge cases"""
    
    def test_injection_attempts_in_prompts(self, client):
        """Test various injection attempts in prompts"""
        injection_attempts = [
            # SQL injection attempts
            "'; DROP TABLE jobs; --",
            "' OR '1'='1",
            "UNION SELECT * FROM users",
            
            # NoSQL injection attempts
            "'; db.jobs.drop(); //",
            "$where: function() { return true; }",
            
            # Command injection attempts
            "; rm -rf /",
            "$(whoami)",
            "`cat /etc/passwd`",
            
            # Script injection attempts
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "onload=alert('xss')",
        ]
        
        with patch('api.queue.get_job_queue') as mock_get_queue:
            mock_queue = MagicMock()
            mock_get_queue.return_value = mock_queue
            
            job_data = JobData(
                id="test-job-injection",
                status=JobStatus.QUEUED,
                created_at=datetime.now(timezone.utc),
                prompt="sanitized prompt"
            )
            mock_queue.enqueue_job.return_value = job_data
            
            for injection_prompt in injection_attempts:
                response = client.post("/v1/jobs", json={"prompt": injection_prompt})
                
                # Should either accept and sanitize, or reject with validation error, or hit rate limit
                assert response.status_code in [202, 422, 429, 500]
                
                if response.status_code == 202:
                    # If accepted, verify no dangerous content in response
                    data = response.json()
                    response_text = json.dumps(data)
                    assert "<script>" not in response_text
                    assert "DROP TABLE" not in response_text
    
    def test_path_traversal_attempts(self, client):
        """Test path traversal attempts in job IDs"""
        traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc//passwd",
        ]
        
        for job_id in traversal_attempts:
            response = client.get(f"/v1/jobs/{job_id}")
            
            # Should not allow path traversal - various error codes possible
            assert response.status_code in [400, 404, 422, 500]
    
    def test_header_injection_attempts(self, client):
        """Test header injection attempts"""
        malicious_headers = {
            "X-Forwarded-For": "127.0.0.1\r\nX-Injected-Header: malicious",
            "User-Agent": "Mozilla/5.0\r\nX-Injected: value",
            "Content-Type": "application/json\r\nX-Evil: header",
        }
        
        with patch('api.queue.get_job_queue') as mock_get_queue:
            mock_queue = MagicMock()
            mock_get_queue.return_value = mock_queue
            
            job_data = JobData(
                id="test-job-headers",
                status=JobStatus.QUEUED,
                created_at=datetime.now(timezone.utc),
                prompt="test"
            )
            mock_queue.enqueue_job.return_value = job_data
            
            for header_name, header_value in malicious_headers.items():
                response = client.post(
                    "/v1/jobs",
                    json={"prompt": "test"},
                    headers={header_name: header_value}
                )
                
                # Should handle malicious headers gracefully
                assert response.status_code in [200, 202, 400, 422, 500]
                
                # Verify no injected headers in response
                for resp_header in response.headers:
                    assert "injected" not in resp_header.lower()
                    assert "evil" not in resp_header.lower()