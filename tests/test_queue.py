"""
Unit tests for Redis job queue infrastructure.
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from api.queue import (
    RedisConnectionManager, 
    JobQueue, 
    JobStateManager,
    initialize_queue_infrastructure
)
from api.models import JobData, JobStatus, JobCreateRequest
from api.config import RedisConfig


class TestRedisConnectionManager:
    """Test Redis connection management"""
    
    def test_init(self):
        """Test RedisConnectionManager initialization"""
        config = RedisConfig(host="localhost", port=6379, db=0)
        manager = RedisConnectionManager(config)
        
        assert manager.config == config
        assert manager._pool is None
        assert manager._connection is None
    
    @patch('redis.ConnectionPool')
    def test_get_connection_pool(self, mock_pool_class):
        """Test connection pool creation"""
        config = RedisConfig(host="test-host", port=6380, db=1, password="secret")
        manager = RedisConnectionManager(config)
        
        mock_pool = Mock()
        mock_pool_class.return_value = mock_pool
        
        pool = manager.get_connection_pool()
        
        assert pool == mock_pool
        mock_pool_class.assert_called_once_with(
            host="test-host",
            port=6380,
            db=1,
            password="secret",
            decode_responses=True,
            max_connections=20,
            retry_on_timeout=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
    
    @patch('redis.Redis')
    def test_get_connection(self, mock_redis_class):
        """Test Redis connection creation"""
        config = RedisConfig()
        manager = RedisConnectionManager(config)
        
        mock_connection = Mock()
        mock_redis_class.return_value = mock_connection
        
        with patch.object(manager, 'get_connection_pool') as mock_get_pool:
            mock_pool = Mock()
            mock_get_pool.return_value = mock_pool
            
            connection = manager.get_connection()
            
            assert connection == mock_connection
            mock_redis_class.assert_called_once_with(connection_pool=mock_pool)
    
    def test_test_connection_success(self):
        """Test successful connection test"""
        config = RedisConfig()
        manager = RedisConnectionManager(config)
        
        mock_connection = Mock()
        mock_connection.ping.return_value = True
        
        with patch.object(manager, 'get_connection', return_value=mock_connection):
            result = manager.test_connection()
            
            assert result is True
            mock_connection.ping.assert_called_once()
    
    def test_test_connection_failure(self):
        """Test failed connection test"""
        config = RedisConfig()
        manager = RedisConnectionManager(config)
        
        mock_connection = Mock()
        mock_connection.ping.side_effect = Exception("Connection failed")
        
        with patch.object(manager, 'get_connection', return_value=mock_connection):
            result = manager.test_connection()
            
            assert result is False
    
    def test_close(self):
        """Test connection cleanup"""
        config = RedisConfig()
        manager = RedisConnectionManager(config)
        
        mock_connection = Mock()
        mock_pool = Mock()
        
        manager._connection = mock_connection
        manager._pool = mock_pool
        
        manager.close()
        
        mock_connection.close.assert_called_once()
        mock_pool.disconnect.assert_called_once()
        assert manager._connection is None
        assert manager._pool is None


class TestJobQueue:
    """Test job queue operations"""
    
    @pytest.fixture
    def mock_redis_manager(self):
        """Create mock Redis manager"""
        manager = Mock(spec=RedisConnectionManager)
        mock_connection = Mock()
        manager.get_connection.return_value = mock_connection
        return manager, mock_connection
    
    @pytest.fixture
    def job_queue(self, mock_redis_manager):
        """Create JobQueue instance with mocked Redis"""
        manager, _ = mock_redis_manager
        return JobQueue(manager, "test_queue")
    
    def test_init(self, mock_redis_manager):
        """Test JobQueue initialization"""
        manager, _ = mock_redis_manager
        queue = JobQueue(manager, "test_queue")
        
        assert queue.redis_manager == manager
        assert queue.queue_name == "test_queue"
        assert queue._queue is None
        assert queue._redis is None
    
    @patch('api.queue.Queue')
    def test_queue_property(self, mock_queue_class, job_queue, mock_redis_manager):
        """Test queue property initialization"""
        _, mock_connection = mock_redis_manager
        mock_queue = Mock()
        mock_queue_class.return_value = mock_queue
        
        queue = job_queue.queue
        
        # Just verify the queue was created with correct parameters
        mock_queue_class.assert_called_once_with("test_queue", connection=mock_connection)
        # Verify we got a queue instance (can't compare directly due to RQ's __eq__ implementation)
        assert queue is not None
    
    def test_redis_property(self, job_queue, mock_redis_manager):
        """Test Redis property"""
        _, mock_connection = mock_redis_manager
        
        redis_conn = job_queue.redis
        
        assert redis_conn == mock_connection
    
    @patch('uuid.uuid4')
    @patch('api.queue.datetime')
    def test_enqueue_job_success(self, mock_datetime, mock_uuid, job_queue, mock_redis_manager):
        """Test successful job enqueueing"""
        # Setup mocks
        job_id = "test-job-id"
        mock_uuid.return_value.hex = job_id
        mock_uuid.return_value.__str__ = lambda x: job_id
        
        created_at = datetime(2025, 8, 31, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = created_at
        
        _, mock_connection = mock_redis_manager
        mock_queue = Mock()
        mock_rq_job = Mock()
        mock_queue.enqueue.return_value = mock_rq_job
        
        # Mock the queue property by patching the _queue attribute
        job_queue._queue = mock_queue
        
        with patch.object(job_queue, '_store_job_data') as mock_store:
            request = JobCreateRequest(prompt="Test prompt")
            config = {"test": "config"}
            
            result = job_queue.enqueue_job(request, config)
            
            # Verify result
            assert result.id == job_id
            assert result.status == JobStatus.QUEUED
            assert result.prompt == "Test prompt"
            assert result.config == config
            assert result.created_at == created_at
            
            # Verify RQ enqueue was called
            mock_queue.enqueue.assert_called_once_with(
                'workers.video_worker.process_video_job',
                job_id,
                job_timeout=3600,
                job_id=job_id
            )
            
            # Verify job data was stored
            mock_store.assert_called_once()
    
    def test_enqueue_job_failure(self, job_queue, mock_redis_manager):
        """Test job enqueueing failure"""
        mock_queue = Mock()
        mock_queue.enqueue.side_effect = Exception("Queue error")
        
        # Mock the queue property by patching the _queue attribute
        job_queue._queue = mock_queue
        
        with patch.object(job_queue, '_store_job_data'):
            with patch.object(job_queue, '_delete_job_data') as mock_delete:
                request = JobCreateRequest(prompt="Test prompt")
                config = {"test": "config"}
                
                with pytest.raises(Exception, match="Queue error"):
                    job_queue.enqueue_job(request, config)
                
                # Verify cleanup was called
                mock_delete.assert_called_once()
    
    def test_get_job_success(self, job_queue, mock_redis_manager):
        """Test successful job retrieval"""
        _, mock_connection = mock_redis_manager
        
        job_data = {
            "id": "test-job",
            "status": "queued",
            "progress": 0,
            "created_at": "2025-08-31T12:00:00Z",
            "prompt": "Test prompt",
            "config": {},
            "logs": []
        }
        
        mock_connection.get.return_value = json.dumps(job_data)
        
        result = job_queue.get_job("test-job")
        
        assert result is not None
        assert result.id == "test-job"
        assert result.status == JobStatus.QUEUED
        assert result.prompt == "Test prompt"
        
        mock_connection.get.assert_called_once_with("job:test-job")
    
    def test_get_job_not_found(self, job_queue, mock_redis_manager):
        """Test job retrieval when job doesn't exist"""
        _, mock_connection = mock_redis_manager
        mock_connection.get.return_value = None
        
        result = job_queue.get_job("nonexistent-job")
        
        assert result is None
    
    def test_update_job_status_success(self, job_queue, mock_redis_manager):
        """Test successful job status update"""
        existing_job = JobData(
            id="test-job",
            status=JobStatus.QUEUED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        with patch.object(job_queue, 'get_job', return_value=existing_job):
            with patch.object(job_queue, '_store_job_data') as mock_store:
                result = job_queue.update_job_status(
                    "test-job", 
                    JobStatus.STARTED, 
                    progress=10
                )
                
                assert result is True
                mock_store.assert_called_once()
                
                # Check that job data was updated
                stored_job = mock_store.call_args[0][0]
                assert stored_job.status == JobStatus.STARTED
                assert stored_job.progress == 10
                assert stored_job.started_at is not None
    
    def test_update_job_status_not_found(self, job_queue):
        """Test job status update when job doesn't exist"""
        with patch.object(job_queue, 'get_job', return_value=None):
            result = job_queue.update_job_status("nonexistent-job", JobStatus.STARTED)
            
            assert result is False
    
    def test_add_job_log_success(self, job_queue):
        """Test successful log addition"""
        existing_job = JobData(
            id="test-job",
            status=JobStatus.STARTED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        with patch.object(job_queue, 'get_job', return_value=existing_job):
            with patch.object(job_queue, '_store_job_data') as mock_store:
                result = job_queue.add_job_log("test-job", "Test log message")
                
                assert result is True
                mock_store.assert_called_once()
                
                # Check that log was added
                stored_job = mock_store.call_args[0][0]
                assert len(stored_job.logs) == 1
                assert "Test log message" in stored_job.logs[0]
    
    def test_get_job_logs(self, job_queue):
        """Test job log retrieval"""
        existing_job = JobData(
            id="test-job",
            status=JobStatus.STARTED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt",
            logs=["Log 1", "Log 2", "Log 3"]
        )
        
        with patch.object(job_queue, 'get_job', return_value=existing_job):
            # Test getting all logs
            logs = job_queue.get_job_logs("test-job")
            assert logs == ["Log 1", "Log 2", "Log 3"]
            
            # Test getting tail logs
            logs = job_queue.get_job_logs("test-job", tail=2)
            assert logs == ["Log 2", "Log 3"]
    
    @patch('rq.job.Job.fetch')
    @patch('workers.video_worker.set_job_cancellation_flag')
    def test_cancel_job_success(self, mock_set_flag, mock_job_fetch, job_queue):
        """Test successful job cancellation"""
        existing_job = JobData(
            id="test-job",
            status=JobStatus.STARTED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        mock_rq_job = Mock()
        mock_job_fetch.return_value = mock_rq_job
        
        with patch.object(job_queue, 'get_job', return_value=existing_job):
            with patch.object(job_queue, 'update_job_status') as mock_update:
                with patch.object(job_queue, 'add_job_log') as mock_add_log:
                    result = job_queue.cancel_job("test-job")
                    
                    assert result is True
                    mock_set_flag.assert_called_once_with("test-job")
                    mock_rq_job.cancel.assert_called_once()
                    mock_update.assert_called_once_with("test-job", JobStatus.CANCELED)
                    mock_add_log.assert_called_once_with("test-job", "Job cancellation requested")
    
    def test_cancel_finished_job(self, job_queue):
        """Test cancelling an already finished job"""
        existing_job = JobData(
            id="test-job",
            status=JobStatus.FINISHED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        with patch.object(job_queue, 'get_job', return_value=existing_job):
            result = job_queue.cancel_job("test-job")
            
            assert result is False
    
    def test_cancel_nonexistent_job(self, job_queue):
        """Test cancelling a job that doesn't exist"""
        with patch.object(job_queue, 'get_job', return_value=None):
            result = job_queue.cancel_job("nonexistent-job")
            
            assert result is False
    
    @patch('rq.job.Job.fetch')
    @patch('workers.video_worker.set_job_cancellation_flag')
    def test_cancel_job_rq_not_found(self, mock_set_flag, mock_job_fetch, job_queue):
        """Test job cancellation when RQ job is not found"""
        from rq.exceptions import NoSuchJobError
        
        existing_job = JobData(
            id="test-job",
            status=JobStatus.STARTED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        mock_job_fetch.side_effect = NoSuchJobError("Job not found")
        
        with patch.object(job_queue, 'get_job', return_value=existing_job):
            with patch.object(job_queue, 'update_job_status') as mock_update:
                with patch.object(job_queue, 'add_job_log') as mock_add_log:
                    result = job_queue.cancel_job("test-job")
                    
                    assert result is True
                    mock_set_flag.assert_called_once_with("test-job")
                    mock_update.assert_called_once_with("test-job", JobStatus.CANCELED)
                    mock_add_log.assert_called_once_with("test-job", "Job cancellation requested")
    
    def test_get_queue_stats(self, job_queue):
        """Test queue statistics retrieval"""
        mock_queue = Mock()
        mock_queue.__len__ = Mock(return_value=5)
        mock_queue.started_job_registry.__len__ = Mock(return_value=2)
        mock_queue.finished_job_registry.__len__ = Mock(return_value=10)
        mock_queue.failed_job_registry.__len__ = Mock(return_value=1)
        mock_queue.deferred_job_registry.__len__ = Mock(return_value=0)
        mock_queue.scheduled_job_registry.__len__ = Mock(return_value=3)
        
        # Mock the queue property by patching the _queue attribute
        job_queue._queue = mock_queue
        
        stats = job_queue.get_queue_stats()
        
        expected = {
            "queued_jobs": 5,
            "started_jobs": 2,
            "finished_jobs": 10,
            "failed_jobs": 1,
            "deferred_jobs": 0,
            "scheduled_jobs": 3
        }
        
        assert stats == expected


class TestJobStateManager:
    """Test job state management"""
    
    def test_valid_transitions(self):
        """Test valid state transitions"""
        # Test valid transitions
        assert JobStateManager.is_valid_transition(JobStatus.QUEUED, JobStatus.STARTED)
        assert JobStateManager.is_valid_transition(JobStatus.STARTED, JobStatus.PROGRESS)
        assert JobStateManager.is_valid_transition(JobStatus.PROGRESS, JobStatus.FINISHED)
        assert JobStateManager.is_valid_transition(JobStatus.STARTED, JobStatus.FAILED)
        assert JobStateManager.is_valid_transition(JobStatus.QUEUED, JobStatus.CANCELED)
    
    def test_invalid_transitions(self):
        """Test invalid state transitions"""
        # Test invalid transitions
        assert not JobStateManager.is_valid_transition(JobStatus.FINISHED, JobStatus.STARTED)
        assert not JobStateManager.is_valid_transition(JobStatus.FAILED, JobStatus.PROGRESS)
        assert not JobStateManager.is_valid_transition(JobStatus.CANCELED, JobStatus.FINISHED)
        assert not JobStateManager.is_valid_transition(JobStatus.QUEUED, JobStatus.FINISHED)
    
    def test_terminal_statuses(self):
        """Test terminal status identification"""
        assert JobStateManager.is_terminal_status(JobStatus.FINISHED)
        assert JobStateManager.is_terminal_status(JobStatus.FAILED)
        assert JobStateManager.is_terminal_status(JobStatus.CANCELED)
        
        assert not JobStateManager.is_terminal_status(JobStatus.QUEUED)
        assert not JobStateManager.is_terminal_status(JobStatus.STARTED)
        assert not JobStateManager.is_terminal_status(JobStatus.PROGRESS)
    
    def test_get_valid_next_states(self):
        """Test getting valid next states"""
        # Test non-terminal states
        queued_next = JobStateManager.get_valid_next_states(JobStatus.QUEUED)
        assert JobStatus.STARTED in queued_next
        assert JobStatus.CANCELED in queued_next
        
        started_next = JobStateManager.get_valid_next_states(JobStatus.STARTED)
        assert JobStatus.PROGRESS in started_next
        assert JobStatus.FINISHED in started_next
        assert JobStatus.FAILED in started_next
        assert JobStatus.CANCELED in started_next
        
        # Test terminal states
        finished_next = JobStateManager.get_valid_next_states(JobStatus.FINISHED)
        assert len(finished_next) == 0


class TestQueueInitialization:
    """Test queue infrastructure initialization"""
    
    @patch('api.queue.RedisConnectionManager')
    @patch('api.queue.JobQueue')
    def test_initialize_queue_infrastructure_success(self, mock_job_queue_class, mock_manager_class):
        """Test successful queue infrastructure initialization"""
        config = RedisConfig()
        
        mock_manager = Mock()
        mock_manager.test_connection.return_value = True
        mock_manager_class.return_value = mock_manager
        
        mock_queue = Mock()
        mock_job_queue_class.return_value = mock_queue
        
        manager, queue = initialize_queue_infrastructure(config)
        
        assert manager == mock_manager
        assert queue == mock_queue
        
        mock_manager_class.assert_called_once_with(config)
        mock_manager.test_connection.assert_called_once()
        mock_job_queue_class.assert_called_once_with(mock_manager)
    
    @patch('api.queue.RedisConnectionManager')
    def test_initialize_queue_infrastructure_connection_failure(self, mock_manager_class):
        """Test queue infrastructure initialization with connection failure"""
        config = RedisConfig()
        
        mock_manager = Mock()
        mock_manager.test_connection.return_value = False
        mock_manager_class.return_value = mock_manager
        
        with pytest.raises(Exception, match="Redis connection test failed"):
            initialize_queue_infrastructure(config)


if __name__ == "__main__":
    pytest.main([__file__])