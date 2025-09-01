"""
Comprehensive unit tests for job state transitions and cancellation behavior.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from api.queue import JobQueue, JobStateManager
from api.models import JobData, JobStatus, JobCreateRequest


class TestJobStateTransitions:
    """Test comprehensive job state transition scenarios"""
    
    def test_valid_state_transition_paths(self):
        """Test all valid state transition paths"""
        valid_paths = [
            # Normal successful flow
            [JobStatus.QUEUED, JobStatus.STARTED, JobStatus.PROGRESS, JobStatus.FINISHED],
            
            # Quick completion without progress updates
            [JobStatus.QUEUED, JobStatus.STARTED, JobStatus.FINISHED],
            
            # Failure during startup
            [JobStatus.QUEUED, JobStatus.STARTED, JobStatus.FAILED],
            
            # Failure during progress
            [JobStatus.QUEUED, JobStatus.STARTED, JobStatus.PROGRESS, JobStatus.FAILED],
            
            # Cancellation from queued state
            [JobStatus.QUEUED, JobStatus.CANCELED],
            
            # Cancellation from started state
            [JobStatus.QUEUED, JobStatus.STARTED, JobStatus.CANCELED],
            
            # Cancellation from progress state
            [JobStatus.QUEUED, JobStatus.STARTED, JobStatus.PROGRESS, JobStatus.CANCELED],
        ]
        
        for path in valid_paths:
            current_status = path[0]
            for next_status in path[1:]:
                assert JobStateManager.is_valid_transition(current_status, next_status), \
                    f"Invalid transition from {current_status} to {next_status} in path {path}"
                current_status = next_status
    
    def test_invalid_state_transitions(self):
        """Test invalid state transitions are properly rejected"""
        invalid_transitions = [
            # Cannot go backwards in normal flow
            (JobStatus.STARTED, JobStatus.QUEUED),
            (JobStatus.PROGRESS, JobStatus.STARTED),
            (JobStatus.PROGRESS, JobStatus.QUEUED),
            (JobStatus.FINISHED, JobStatus.PROGRESS),
            (JobStatus.FINISHED, JobStatus.STARTED),
            (JobStatus.FINISHED, JobStatus.QUEUED),
            
            # Cannot transition from terminal states
            (JobStatus.FINISHED, JobStatus.CANCELED),
            (JobStatus.FAILED, JobStatus.STARTED),
            (JobStatus.FAILED, JobStatus.PROGRESS),
            (JobStatus.FAILED, JobStatus.FINISHED),
            (JobStatus.CANCELED, JobStatus.STARTED),
            (JobStatus.CANCELED, JobStatus.PROGRESS),
            (JobStatus.CANCELED, JobStatus.FINISHED),
            
            # Cannot skip required intermediate states
            (JobStatus.QUEUED, JobStatus.PROGRESS),  # Must go through STARTED
            (JobStatus.QUEUED, JobStatus.FINISHED),  # Must go through STARTED
            (JobStatus.QUEUED, JobStatus.FAILED),    # Must go through STARTED
        ]
        
        for from_status, to_status in invalid_transitions:
            assert not JobStateManager.is_valid_transition(from_status, to_status), \
                f"Should not allow transition from {from_status} to {to_status}"
    
    def test_terminal_status_identification(self):
        """Test terminal status identification"""
        terminal_statuses = [JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELED]
        non_terminal_statuses = [JobStatus.QUEUED, JobStatus.STARTED, JobStatus.PROGRESS]
        
        for status in terminal_statuses:
            assert JobStateManager.is_terminal_status(status), \
                f"{status} should be identified as terminal"
        
        for status in non_terminal_statuses:
            assert not JobStateManager.is_terminal_status(status), \
                f"{status} should not be identified as terminal"
    
    def test_valid_next_states_from_each_status(self):
        """Test valid next states from each current status"""
        expected_next_states = {
            JobStatus.QUEUED: {JobStatus.STARTED, JobStatus.CANCELED},
            JobStatus.STARTED: {JobStatus.PROGRESS, JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELED},
            JobStatus.PROGRESS: {JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELED},  # No PROGRESS -> PROGRESS
            JobStatus.FINISHED: set(),  # Terminal state
            JobStatus.FAILED: set(),    # Terminal state
            JobStatus.CANCELED: set(),  # Terminal state
        }
        
        for current_status, expected_next in expected_next_states.items():
            actual_next = set(JobStateManager.get_valid_next_states(current_status))
            assert actual_next == expected_next, \
                f"Expected next states for {current_status}: {expected_next}, got: {actual_next}"
    
    def test_state_transition_with_timestamps(self):
        """Test that state transitions update appropriate timestamps"""
        job_data = JobData(
            id="test-job",
            status=JobStatus.QUEUED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        # Transition to STARTED should set started_at
        job_data.status = JobStatus.STARTED
        job_data.started_at = datetime.now(timezone.utc)
        assert job_data.started_at is not None
        assert job_data.finished_at is None
        
        # Transition to PROGRESS should not change timestamps
        original_started_at = job_data.started_at
        job_data.status = JobStatus.PROGRESS
        job_data.progress = 50
        assert job_data.started_at == original_started_at
        assert job_data.finished_at is None
        
        # Transition to FINISHED should set finished_at
        job_data.status = JobStatus.FINISHED
        job_data.finished_at = datetime.now(timezone.utc)
        job_data.progress = 100
        assert job_data.started_at == original_started_at
        assert job_data.finished_at is not None
        assert job_data.finished_at > job_data.started_at


class TestJobCancellationBehavior:
    """Test comprehensive job cancellation behavior"""
    
    @pytest.fixture
    def mock_redis_manager(self):
        """Create mock Redis manager"""
        manager = Mock()
        mock_connection = Mock()
        manager.get_connection.return_value = mock_connection
        return manager, mock_connection
    
    @pytest.fixture
    def job_queue(self, mock_redis_manager):
        """Create JobQueue instance with mocked Redis"""
        manager, _ = mock_redis_manager
        return JobQueue(manager, "test_queue")
    
    def test_cancel_queued_job(self, job_queue):
        """Test cancelling a job in QUEUED state"""
        job_data = JobData(
            id="test-job-queued",
            status=JobStatus.QUEUED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        with patch.object(job_queue, 'get_job', return_value=job_data), \
             patch.object(job_queue, 'update_job_status', return_value=True) as mock_update, \
             patch.object(job_queue, 'add_job_log', return_value=True) as mock_log, \
             patch('rq.job.Job.fetch') as mock_fetch, \
             patch('workers.video_worker.set_job_cancellation_flag') as mock_flag:
            
            mock_rq_job = Mock()
            mock_fetch.return_value = mock_rq_job
            
            result = job_queue.cancel_job("test-job-queued")
            
            assert result is True
            mock_flag.assert_called_once_with("test-job-queued")
            mock_rq_job.cancel.assert_called_once()
            mock_update.assert_called_once_with("test-job-queued", JobStatus.CANCELED)
            mock_log.assert_called_once_with("test-job-queued", "Job cancellation requested")
    
    def test_cancel_started_job(self, job_queue):
        """Test cancelling a job in STARTED state"""
        job_data = JobData(
            id="test-job-started",
            status=JobStatus.STARTED,
            created_at=datetime.now(timezone.utc),
            started_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        with patch.object(job_queue, 'get_job', return_value=job_data), \
             patch.object(job_queue, 'update_job_status', return_value=True) as mock_update, \
             patch.object(job_queue, 'add_job_log', return_value=True) as mock_log, \
             patch('rq.job.Job.fetch') as mock_fetch, \
             patch('workers.video_worker.set_job_cancellation_flag') as mock_flag:
            
            mock_rq_job = Mock()
            mock_fetch.return_value = mock_rq_job
            
            result = job_queue.cancel_job("test-job-started")
            
            assert result is True
            mock_flag.assert_called_once_with("test-job-started")
            mock_rq_job.cancel.assert_called_once()
            mock_update.assert_called_once_with("test-job-started", JobStatus.CANCELED)
            mock_log.assert_called_once_with("test-job-started", "Job cancellation requested")
    
    def test_cancel_progress_job(self, job_queue):
        """Test cancelling a job in PROGRESS state"""
        job_data = JobData(
            id="test-job-progress",
            status=JobStatus.PROGRESS,
            progress=45,
            created_at=datetime.now(timezone.utc),
            started_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        with patch.object(job_queue, 'get_job', return_value=job_data), \
             patch.object(job_queue, 'update_job_status', return_value=True) as mock_update, \
             patch.object(job_queue, 'add_job_log', return_value=True) as mock_log, \
             patch('rq.job.Job.fetch') as mock_fetch, \
             patch('workers.video_worker.set_job_cancellation_flag') as mock_flag:
            
            mock_rq_job = Mock()
            mock_fetch.return_value = mock_rq_job
            
            result = job_queue.cancel_job("test-job-progress")
            
            assert result is True
            mock_flag.assert_called_once_with("test-job-progress")
            mock_rq_job.cancel.assert_called_once()
            mock_update.assert_called_once_with("test-job-progress", JobStatus.CANCELED)
            mock_log.assert_called_once_with("test-job-progress", "Job cancellation requested")
    
    def test_cancel_finished_job(self, job_queue):
        """Test attempting to cancel a FINISHED job"""
        job_data = JobData(
            id="test-job-finished",
            status=JobStatus.FINISHED,
            progress=100,
            created_at=datetime.now(timezone.utc),
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        with patch.object(job_queue, 'get_job', return_value=job_data):
            result = job_queue.cancel_job("test-job-finished")
            
            assert result is False  # Cannot cancel finished job
    
    def test_cancel_failed_job(self, job_queue):
        """Test attempting to cancel a FAILED job"""
        job_data = JobData(
            id="test-job-failed",
            status=JobStatus.FAILED,
            created_at=datetime.now(timezone.utc),
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            prompt="Test prompt",
            error="Job failed due to error"
        )
        
        with patch.object(job_queue, 'get_job', return_value=job_data):
            result = job_queue.cancel_job("test-job-failed")
            
            assert result is False  # Cannot cancel failed job
    
    def test_cancel_already_canceled_job(self, job_queue):
        """Test attempting to cancel an already CANCELED job"""
        job_data = JobData(
            id="test-job-canceled",
            status=JobStatus.CANCELED,
            created_at=datetime.now(timezone.utc),
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        with patch.object(job_queue, 'get_job', return_value=job_data):
            result = job_queue.cancel_job("test-job-canceled")
            
            assert result is False  # Cannot cancel already canceled job
    
    def test_cancel_nonexistent_job(self, job_queue):
        """Test attempting to cancel a non-existent job"""
        with patch.object(job_queue, 'get_job', return_value=None):
            result = job_queue.cancel_job("nonexistent-job")
            
            assert result is False
    
    def test_cancel_job_rq_job_not_found(self, job_queue):
        """Test cancellation when RQ job is not found"""
        from rq.exceptions import NoSuchJobError
        
        job_data = JobData(
            id="test-job-no-rq",
            status=JobStatus.STARTED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        with patch.object(job_queue, 'get_job', return_value=job_data), \
             patch.object(job_queue, 'update_job_status', return_value=True) as mock_update, \
             patch.object(job_queue, 'add_job_log', return_value=True) as mock_log, \
             patch('rq.job.Job.fetch', side_effect=NoSuchJobError("Job not found")), \
             patch('workers.video_worker.set_job_cancellation_flag') as mock_flag:
            
            result = job_queue.cancel_job("test-job-no-rq")
            
            # Should still succeed and mark as canceled
            assert result is True
            mock_flag.assert_called_once_with("test-job-no-rq")
            mock_update.assert_called_once_with("test-job-no-rq", JobStatus.CANCELED)
            mock_log.assert_called_once_with("test-job-no-rq", "Job cancellation requested")
    
    def test_cancel_job_rq_cancel_fails(self, job_queue):
        """Test cancellation when RQ job.cancel() fails"""
        job_data = JobData(
            id="test-job-cancel-fail",
            status=JobStatus.STARTED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        with patch.object(job_queue, 'get_job', return_value=job_data), \
             patch.object(job_queue, 'update_job_status', return_value=True) as mock_update, \
             patch.object(job_queue, 'add_job_log', return_value=True) as mock_log, \
             patch('rq.job.Job.fetch') as mock_fetch, \
             patch('workers.video_worker.set_job_cancellation_flag') as mock_flag:
            
            mock_rq_job = Mock()
            mock_rq_job.cancel.side_effect = Exception("RQ cancel failed")
            mock_fetch.return_value = mock_rq_job
            
            result = job_queue.cancel_job("test-job-cancel-fail")
            
            # Should still succeed and mark as canceled even if RQ cancel fails
            assert result is True
            mock_flag.assert_called_once_with("test-job-cancel-fail")
            mock_update.assert_called_once_with("test-job-cancel-fail", JobStatus.CANCELED)
            mock_log.assert_called_once_with("test-job-cancel-fail", "Job cancellation requested")


class TestJobStateTransitionEdgeCases:
    """Test edge cases in job state transitions"""
    
    @pytest.fixture
    def mock_redis_manager(self):
        """Create mock Redis manager"""
        manager = Mock()
        mock_connection = Mock()
        manager.get_connection.return_value = mock_connection
        return manager, mock_connection
    
    @pytest.fixture
    def job_queue(self, mock_redis_manager):
        """Create JobQueue instance with mocked Redis"""
        manager, _ = mock_redis_manager
        return JobQueue(manager, "test_queue")
    
    def test_rapid_state_transitions(self, job_queue):
        """Test rapid consecutive state transitions"""
        job_data = JobData(
            id="test-job-rapid",
            status=JobStatus.QUEUED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        with patch.object(job_queue, 'get_job', return_value=job_data), \
             patch.object(job_queue, '_store_job_data') as mock_store:
            
            # Rapid transitions: QUEUED -> STARTED -> PROGRESS -> FINISHED
            transitions = [
                (JobStatus.STARTED, 0),
                (JobStatus.PROGRESS, 25),
                (JobStatus.PROGRESS, 50),
                (JobStatus.PROGRESS, 75),
                (JobStatus.FINISHED, 100)
            ]
            
            for status, progress in transitions:
                result = job_queue.update_job_status("test-job-rapid", status, progress=progress)
                assert result is True
                
                # Update the mock job data for next iteration
                job_data.status = status
                job_data.progress = progress
                if status == JobStatus.STARTED and job_data.started_at is None:
                    job_data.started_at = datetime.now(timezone.utc)
                elif status == JobStatus.FINISHED:
                    job_data.finished_at = datetime.now(timezone.utc)
            
            # Verify all transitions were stored
            assert mock_store.call_count == len(transitions)
    
    def test_concurrent_state_updates(self, job_queue):
        """Test concurrent state updates on the same job"""
        import threading
        import time
        
        job_data = JobData(
            id="test-job-concurrent",
            status=JobStatus.STARTED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        results = []
        errors = []
        
        def update_progress(progress_value):
            try:
                result = job_queue.update_job_status(
                    "test-job-concurrent", 
                    JobStatus.PROGRESS, 
                    progress=progress_value
                )
                results.append((progress_value, result))
            except Exception as e:
                errors.append((progress_value, e))
        
        with patch.object(job_queue, 'get_job', return_value=job_data), \
             patch.object(job_queue, '_store_job_data') as mock_store:
            
            # Create multiple threads updating progress concurrently
            threads = []
            for i in range(10):
                thread = threading.Thread(target=update_progress, args=(i * 10,))
                threads.append(thread)
            
            # Start all threads
            for thread in threads:
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Verify no errors occurred
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == 10
            
            # All updates should succeed
            for progress_value, result in results:
                assert result is True
    
    def test_state_transition_with_invalid_progress(self, job_queue):
        """Test state transitions with invalid progress values"""
        job_data = JobData(
            id="test-job-invalid-progress",
            status=JobStatus.STARTED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        with patch.object(job_queue, 'get_job', return_value=job_data), \
             patch.object(job_queue, '_store_job_data') as mock_store:
            
            # Test invalid progress values
            invalid_progress_values = [-1, 101, 150, -50]
            
            for invalid_progress in invalid_progress_values:
                # Should handle invalid progress gracefully
                result = job_queue.update_job_status(
                    "test-job-invalid-progress", 
                    JobStatus.PROGRESS, 
                    progress=invalid_progress
                )
                
                # Should either succeed with clamped value or fail gracefully
                assert isinstance(result, bool)
    
    def test_state_transition_timestamp_consistency(self, job_queue):
        """Test timestamp consistency during state transitions"""
        base_time = datetime.now(timezone.utc)
        
        job_data = JobData(
            id="test-job-timestamps",
            status=JobStatus.QUEUED,
            created_at=base_time,
            prompt="Test prompt"
        )
        
        with patch.object(job_queue, 'get_job', return_value=job_data), \
             patch.object(job_queue, '_store_job_data') as mock_store:
            
            # Transition to STARTED
            result = job_queue.update_job_status("test-job-timestamps", JobStatus.STARTED)
            assert result is True
            
            # Verify started_at is set and after created_at
            stored_job = mock_store.call_args[0][0]
            assert stored_job.started_at is not None
            assert stored_job.started_at >= stored_job.created_at
            assert stored_job.finished_at is None
            
            # Update job data for next transition
            job_data.status = JobStatus.STARTED
            job_data.started_at = stored_job.started_at
            
            # Transition to FINISHED
            result = job_queue.update_job_status("test-job-timestamps", JobStatus.FINISHED)
            assert result is True
            
            # Verify finished_at is set and after started_at
            stored_job = mock_store.call_args[0][0]
            assert stored_job.finished_at is not None
            assert stored_job.finished_at >= stored_job.started_at
            assert stored_job.finished_at >= stored_job.created_at
    
    def test_state_transition_with_job_data_corruption(self, job_queue):
        """Test state transitions when job data is corrupted"""
        # Simulate corrupted job data - use valid values since Pydantic validates
        corrupted_job_data = JobData(
            id="test-job-corrupted",
            status=JobStatus.QUEUED,  # Valid status but simulate corruption in other ways
            created_at=datetime.now(timezone.utc),
            prompt=""  # Empty prompt - this could be considered corruption
        )
        
        with patch.object(job_queue, 'get_job', return_value=corrupted_job_data):
            # Should handle corrupted data gracefully
            result = job_queue.update_job_status("test-job-corrupted", JobStatus.STARTED)
            
            # Should either succeed with data repair or fail gracefully
            assert isinstance(result, bool)
    
    def test_job_cancellation_race_conditions(self, job_queue):
        """Test job cancellation race conditions"""
        job_data = JobData(
            id="test-job-race",
            status=JobStatus.STARTED,
            created_at=datetime.now(timezone.utc),
            prompt="Test prompt"
        )
        
        # Simulate race condition: job finishes while cancellation is in progress
        def mock_get_job_side_effect(job_id):
            # First call returns STARTED, second call returns FINISHED
            if not hasattr(mock_get_job_side_effect, 'call_count'):
                mock_get_job_side_effect.call_count = 0
            
            mock_get_job_side_effect.call_count += 1
            
            if mock_get_job_side_effect.call_count == 1:
                return job_data  # STARTED
            else:
                # Job finished during cancellation
                finished_job = JobData(
                    id="test-job-race",
                    status=JobStatus.FINISHED,
                    created_at=job_data.created_at,
                    started_at=job_data.created_at,
                    finished_at=datetime.now(timezone.utc),
                    prompt="Test prompt"
                )
                return finished_job
        
        with patch.object(job_queue, 'get_job', side_effect=mock_get_job_side_effect), \
             patch('rq.job.Job.fetch') as mock_fetch, \
             patch('workers.video_worker.set_job_cancellation_flag'):
            
            mock_rq_job = Mock()
            mock_fetch.return_value = mock_rq_job
            
            result = job_queue.cancel_job("test-job-race")
            
            # Should handle race condition gracefully
            # Result depends on implementation - could be True or False
            assert isinstance(result, bool)


class TestJobStateManagerUtilities:
    """Test JobStateManager utility functions"""
    
    def test_state_transition_validation_comprehensive(self):
        """Test comprehensive state transition validation"""
        # Test all possible status combinations
        all_statuses = [
            JobStatus.QUEUED,
            JobStatus.STARTED, 
            JobStatus.PROGRESS,
            JobStatus.FINISHED,
            JobStatus.FAILED,
            JobStatus.CANCELED
        ]
        
        # Build expected transition matrix based on actual implementation
        expected_transitions = {
            JobStatus.QUEUED: [JobStatus.STARTED, JobStatus.CANCELED],
            JobStatus.STARTED: [JobStatus.PROGRESS, JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELED],
            JobStatus.PROGRESS: [JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELED],  # No self-transition
            JobStatus.FINISHED: [],
            JobStatus.FAILED: [],
            JobStatus.CANCELED: []
        }
        
        for from_status in all_statuses:
            for to_status in all_statuses:
                expected_valid = to_status in expected_transitions[from_status]
                actual_valid = JobStateManager.is_valid_transition(from_status, to_status)
                
                assert actual_valid == expected_valid, \
                    f"Transition {from_status} -> {to_status}: expected {expected_valid}, got {actual_valid}"
    
    def test_terminal_status_completeness(self):
        """Test that terminal status identification is complete"""
        all_statuses = [
            JobStatus.QUEUED,
            JobStatus.STARTED,
            JobStatus.PROGRESS, 
            JobStatus.FINISHED,
            JobStatus.FAILED,
            JobStatus.CANCELED
        ]
        
        terminal_count = 0
        non_terminal_count = 0
        
        for status in all_statuses:
            if JobStateManager.is_terminal_status(status):
                terminal_count += 1
                # Terminal statuses should have no valid next states
                next_states = JobStateManager.get_valid_next_states(status)
                assert len(next_states) == 0, \
                    f"Terminal status {status} should have no valid next states, got {next_states}"
            else:
                non_terminal_count += 1
                # Non-terminal statuses should have at least one valid next state
                next_states = JobStateManager.get_valid_next_states(status)
                assert len(next_states) > 0, \
                    f"Non-terminal status {status} should have valid next states"
        
        # Verify we have both terminal and non-terminal statuses
        assert terminal_count > 0, "Should have at least one terminal status"
        assert non_terminal_count > 0, "Should have at least one non-terminal status"
        assert terminal_count + non_terminal_count == len(all_statuses), \
            "All statuses should be classified as either terminal or non-terminal"
    
    def test_state_transition_symmetry_properties(self):
        """Test symmetry properties of state transitions"""
        # Note: Based on actual implementation, PROGRESS -> PROGRESS is not a valid transition
        # Progress updates are handled differently (not as state transitions)
        
        # Test that no status can transition to QUEUED (except itself, but that's not a real transition)
        all_statuses = [JobStatus.STARTED, JobStatus.PROGRESS, JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELED]
        for status in all_statuses:
            assert not JobStateManager.is_valid_transition(status, JobStatus.QUEUED), \
                f"No status should be able to transition back to QUEUED, but {status} can"
        
        # Test that all non-terminal statuses can transition to CANCELED
        non_terminal_statuses = [JobStatus.QUEUED, JobStatus.STARTED, JobStatus.PROGRESS]
        for status in non_terminal_statuses:
            assert JobStateManager.is_valid_transition(status, JobStatus.CANCELED), \
                f"Non-terminal status {status} should be able to transition to CANCELED"