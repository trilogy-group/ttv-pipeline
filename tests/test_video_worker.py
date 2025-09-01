"""
Unit tests for video worker cancellation functionality.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from workers.video_worker import (
    CancellationToken,
    set_job_cancellation_flag,
    check_cancellation,
    process_manager,
    process_video_job
)
from api.models import JobStatus, JobData


class TestCancellationToken:
    """Test cancellation token functionality"""
    
    def test_init(self):
        """Test CancellationToken initialization"""
        token = CancellationToken("test-job")
        
        assert token.job_id == "test-job"
        assert not token.is_cancelled()
    
    def test_cancel(self):
        """Test cancellation"""
        token = CancellationToken("test-job")
        
        assert not token.is_cancelled()
        
        token.cancel()
        
        assert token.is_cancelled()
    
    def test_global_cancellation_flag(self):
        """Test global cancellation flag integration"""
        token = CancellationToken("test-job")
        
        # Set global flag
        set_job_cancellation_flag("test-job")
        
        assert token.is_cancelled()
        
        # Cleanup
        token.cleanup()
        
        # Create new token - should not be cancelled
        new_token = CancellationToken("test-job")
        assert not new_token.is_cancelled()
    
    def test_cleanup(self):
        """Test cancellation flag cleanup"""
        token = CancellationToken("test-job")
        
        # Set global flag
        set_job_cancellation_flag("test-job")
        assert token.is_cancelled()
        
        # Cleanup
        token.cleanup()
        
        # Flag should be cleared
        new_token = CancellationToken("test-job")
        assert not new_token.is_cancelled()


class TestCancellationFunctions:
    """Test cancellation utility functions"""
    
    def test_set_job_cancellation_flag(self):
        """Test setting cancellation flag"""
        job_id = "test-job"
        
        # Create token before setting flag
        token = CancellationToken(job_id)
        assert not token.is_cancelled()
        
        # Set flag
        set_job_cancellation_flag(job_id)
        
        # Token should now be cancelled
        assert token.is_cancelled()
        
        # Cleanup
        token.cleanup()
    
    @patch('workers.video_worker.get_job_queue')
    def test_check_cancellation_not_cancelled(self, mock_get_queue):
        """Test check_cancellation when job is not cancelled"""
        token = CancellationToken("test-job")
        
        result = check_cancellation(token, "test-job")
        
        assert result is False
        mock_get_queue.assert_not_called()
    
    @patch('workers.video_worker.get_job_queue')
    def test_check_cancellation_cancelled(self, mock_get_queue):
        """Test check_cancellation when job is cancelled"""
        token = CancellationToken("test-job")
        token.cancel()
        
        mock_queue = Mock()
        mock_get_queue.return_value = mock_queue
        
        result = check_cancellation(token, "test-job")
        
        assert result is True
        mock_queue.update_job_status.assert_called_once_with("test-job", JobStatus.CANCELED)
        mock_queue.add_job_log.assert_called_once_with("test-job", "Job cancelled during processing")


class TestProcessManager:
    """Test process manager context"""
    
    def test_process_manager_basic(self):
        """Test basic process manager functionality"""
        with process_manager("test-job") as (token, processes):
            assert isinstance(token, CancellationToken)
            assert token.job_id == "test-job"
            assert isinstance(processes, list)
            assert len(processes) == 0
    
    def test_process_manager_cleanup(self):
        """Test process manager cleanup"""
        job_id = "test-job"
        
        # Set cancellation flag before entering context
        set_job_cancellation_flag(job_id)
        
        with process_manager(job_id) as (token, processes):
            assert token.is_cancelled()
        
        # After context exit, flag should be cleaned up
        new_token = CancellationToken(job_id)
        assert not new_token.is_cancelled()
    
    @patch('subprocess.Popen')
    def test_process_manager_with_processes(self, mock_popen):
        """Test process manager with mock processes"""
        mock_proc = Mock()
        mock_proc.poll.return_value = None  # Process is running
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc
        
        with process_manager("test-job") as (token, processes):
            # Simulate adding a process
            processes.append(mock_proc)
        
        # Process should be terminated on exit
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called()


class TestProcessVideoJob:
    """Test main video processing function with cancellation"""
    
    @patch('workers.video_worker.execute_pipeline_with_config')
    @patch('workers.video_worker.get_job_queue')
    def test_process_video_job_success(self, mock_get_queue, mock_execute_pipeline):
        """Test successful video job processing"""
        job_id = "test-job"
        
        # Mock job data
        job_data = JobData(
            id=job_id,
            status=JobStatus.QUEUED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt",
            config={"test": "config"}
        )
        
        mock_queue = Mock()
        mock_queue.get_job.return_value = job_data
        mock_get_queue.return_value = mock_queue
        
        # Mock pipeline execution
        expected_gcs_uri = f"gs://ttv-api-artifacts/2025-08/{job_id}/final_video.mp4"
        mock_execute_pipeline.return_value = expected_gcs_uri
        
        # Process job
        result = process_video_job(job_id)
        
        # Verify result
        assert result == expected_gcs_uri
        
        # Verify pipeline was called with correct arguments
        mock_execute_pipeline.assert_called_once()
        call_args = mock_execute_pipeline.call_args
        assert call_args[1]['job_id'] == job_id
        assert call_args[1]['prompt'] == "Test prompt"
        assert call_args[1]['config'] == {"test": "config"}
        
        # Verify status updates were called
        assert mock_queue.update_job_status.call_count >= 2  # Started, Finished
        assert mock_queue.add_job_log.call_count >= 2
        
        # Verify final status update
        final_call = mock_queue.update_job_status.call_args_list[-1]
        assert final_call[0] == (job_id, JobStatus.FINISHED)
        assert final_call[1]['progress'] == 100
        assert final_call[1]['gcs_uri'] == expected_gcs_uri
    
    @patch('workers.video_worker.get_job_queue')
    def test_process_video_job_not_found(self, mock_get_queue):
        """Test video job processing when job not found"""
        job_id = "nonexistent-job"
        
        mock_queue = Mock()
        mock_queue.get_job.return_value = None
        mock_get_queue.return_value = mock_queue
        
        with pytest.raises(ValueError, match="Job nonexistent-job not found"):
            process_video_job(job_id)
    
    @patch('workers.video_worker.get_job_queue')
    def test_process_video_job_early_cancellation(self, mock_get_queue):
        """Test video job processing with early cancellation"""
        job_id = "test-job"
        
        # Mock job data
        job_data = JobData(
            id=job_id,
            status=JobStatus.QUEUED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt",
            config={"test": "config"}
        )
        
        mock_queue = Mock()
        mock_queue.get_job.return_value = job_data
        mock_get_queue.return_value = mock_queue
        
        # Set cancellation flag before processing
        set_job_cancellation_flag(job_id)
        
        with pytest.raises(InterruptedError, match="Job cancelled before processing started"):
            process_video_job(job_id)
    
    @patch('workers.video_worker.execute_pipeline_with_config')
    @patch('workers.video_worker.get_job_queue')
    def test_process_video_job_mid_processing_cancellation(self, mock_get_queue, mock_execute_pipeline):
        """Test video job processing with cancellation during processing"""
        job_id = "test-job"
        
        # Mock job data
        job_data = JobData(
            id=job_id,
            status=JobStatus.QUEUED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt",
            config={"test": "config"}
        )
        
        mock_queue = Mock()
        mock_queue.get_job.return_value = job_data
        mock_get_queue.return_value = mock_queue
        
        # Mock pipeline execution to raise cancellation error
        mock_execute_pipeline.side_effect = InterruptedError("Job cancelled during pipeline execution")
        
        with pytest.raises(InterruptedError, match="Job cancelled during"):
            process_video_job(job_id)
        
        # Verify job was marked as started before cancellation
        status_calls = [call for call in mock_queue.update_job_status.call_args_list 
                       if call[0][1] == JobStatus.STARTED]
        assert len(status_calls) >= 1
    
    @patch('workers.video_worker.get_job_queue')
    def test_process_video_job_exception_handling(self, mock_get_queue):
        """Test video job processing exception handling"""
        job_id = "test-job"
        
        # Mock job data
        job_data = JobData(
            id=job_id,
            status=JobStatus.QUEUED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt",
            config={"test": "config"}
        )
        
        mock_queue = Mock()
        mock_queue.get_job.return_value = job_data
        # Simulate error during status update
        mock_queue.update_job_status.side_effect = [None, Exception("Redis error")]
        mock_get_queue.return_value = mock_queue
        
        with pytest.raises(Exception, match="Redis error"):
            process_video_job(job_id)
        
        # Verify error handling was attempted
        assert mock_queue.update_job_status.call_count >= 2


if __name__ == "__main__":
    pytest.main([__file__])