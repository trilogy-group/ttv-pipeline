"""
Integration tests for Trio structured concurrency implementation.

This module tests the integration between Trio executor and video worker
to ensure structured concurrency works correctly with the existing system.
"""

import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

import pytest
import trio

from workers.trio_executor import get_trio_executor, initialize_trio_executor
from workers.video_worker import process_video_job
from api.models import JobData, JobStatus


class TestTrioIntegration:
    """Test integration between Trio and existing video worker"""
    
    def test_trio_availability_check(self):
        """Test that Trio is properly detected as available"""
        from workers.video_worker import TRIO_AVAILABLE
        assert TRIO_AVAILABLE is True
    
    def test_trio_executor_initialization(self):
        """Test that Trio executor can be initialized"""
        executor = initialize_trio_executor()
        assert executor is not None
        
        # Should return same instance
        executor2 = get_trio_executor()
        assert executor is executor2
    
    @patch('workers.video_worker.process_video_job_trio_wrapper')
    @patch('workers.video_worker.get_job_queue')
    def test_process_video_job_uses_trio(self, mock_get_queue, mock_trio_wrapper):
        """Test that process_video_job uses Trio when available"""
        job_id = "test_trio_job"
        expected_result = "gs://bucket/video.mp4"
        
        mock_trio_wrapper.return_value = expected_result
        
        result = process_video_job(job_id, use_trio=True)
        
        assert result == expected_result
        mock_trio_wrapper.assert_called_once_with(job_id)
    
    @patch('workers.video_worker.process_video_job_threading')
    @patch('workers.video_worker.process_video_job_trio_wrapper')
    @patch('workers.video_worker.get_job_queue')
    def test_process_video_job_fallback_to_threading(self, mock_get_queue, mock_trio_wrapper, mock_threading):
        """Test fallback to threading when Trio fails"""
        job_id = "test_fallback_job"
        expected_result = "gs://bucket/video.mp4"
        
        # Trio wrapper fails
        mock_trio_wrapper.side_effect = Exception("Trio failed")
        mock_threading.return_value = expected_result
        
        result = process_video_job(job_id, use_trio=True)
        
        assert result == expected_result
        mock_trio_wrapper.assert_called_once_with(job_id)
        mock_threading.assert_called_once_with(job_id)
    
    @patch('workers.video_worker.process_video_job_threading')
    @patch('workers.video_worker.get_job_queue')
    def test_process_video_job_threading_mode(self, mock_get_queue, mock_threading):
        """Test that threading mode is used when requested"""
        job_id = "test_threading_job"
        expected_result = "gs://bucket/video.mp4"
        
        mock_threading.return_value = expected_result
        
        result = process_video_job(job_id, use_trio=False)
        
        assert result == expected_result
        mock_threading.assert_called_once_with(job_id)


class TestTrioCancellationIntegration:
    """Test cancellation integration between Trio and queue system"""
    
    @patch('workers.trio_executor.get_job_queue')
    def test_trio_executor_cancellation(self, mock_get_queue):
        """Test that Trio executor can cancel jobs"""
        executor = get_trio_executor()
        
        # Mock a job in the executor
        with trio.CancelScope() as cancel_scope:
            from workers.trio_executor import TrioCancellationToken
            token = TrioCancellationToken("test_job", cancel_scope)
            executor.active_jobs["test_job"] = token
            
            # Cancel the job
            result = executor.cancel_job("test_job")
            
            assert result is True
            assert token.is_cancelled()
    
    @patch('workers.trio_executor.get_trio_executor')
    @patch('workers.video_worker.set_job_cancellation_flag')
    @patch('api.queue.Job.fetch')
    def test_queue_cancellation_calls_trio(self, mock_job_fetch, mock_set_flag, mock_get_executor):
        """Test that queue cancellation calls Trio executor"""
        from api.queue import JobQueue, RedisConnectionManager
        from api.config import RedisConfig
        
        # Mock Trio executor
        mock_executor = Mock()
        mock_executor.cancel_job.return_value = True
        mock_get_executor.return_value = mock_executor
        
        # Mock Redis components
        redis_config = RedisConfig(host="localhost", port=6379, db=0)
        redis_manager = Mock()
        redis_manager.get_connection.return_value = Mock()
        
        job_queue = JobQueue(redis_manager)
        
        # Mock job data
        mock_job_data = JobData(
            id="test_job",
            status=JobStatus.STARTED,
            prompt="test prompt",
            config={}
        )
        
        with patch.object(job_queue, 'get_job', return_value=mock_job_data), \
             patch.object(job_queue, 'update_job_status'), \
             patch.object(job_queue, 'add_job_log'):
            
            result = job_queue.cancel_job("test_job")
            
            assert result is True
            mock_executor.cancel_job.assert_called_once_with("test_job")


class TestTrioResourceManagement:
    """Test resource management with Trio structured concurrency"""
    
    @pytest.mark.trio
    async def test_temp_file_cleanup(self):
        """Test that temporary files are cleaned up properly"""
        from workers.trio_executor import cleanup_temp_files_async
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp(prefix="test_trio_cleanup_")
        
        # Verify it exists
        assert os.path.exists(temp_dir)
        
        # Clean it up using Trio
        await cleanup_temp_files_async(temp_dir)
        
        # Verify it's gone
        assert not os.path.exists(temp_dir)
    
    @pytest.mark.trio
    async def test_job_context_manager(self):
        """Test Trio job context manager"""
        from workers.trio_executor import trio_job_context
        
        cleanup_called = False
        
        def cleanup_callback():
            nonlocal cleanup_called
            cleanup_called = True
        
        async with trio_job_context("test_job") as (token, manager):
            assert token.job_id == "test_job"
            assert manager.job_id == "test_job"
            
            # Add cleanup callback
            token.add_cleanup_callback(cleanup_callback)
        
        # Cleanup should have been called
        assert cleanup_called
    
    @pytest.mark.trio
    async def test_process_manager_subprocess_handling(self):
        """Test process manager with actual subprocess"""
        from workers.trio_executor import ProcessManager, TrioCancellationToken
        
        with trio.CancelScope() as cancel_scope:
            token = TrioCancellationToken("test_job", cancel_scope)
            manager = ProcessManager("test_job", token)
            
            # Start a simple process
            process = await manager.start_process(["echo", "hello world"], timeout=5.0)
            
            assert process.returncode == 0
            assert len(manager.processes) == 1
            
            # Clean up
            manager.cleanup_all_processes()
            assert len(manager.processes) == 0


class TestTrioAsyncioFallback:
    """Test asyncio fallback mechanisms"""
    
    @pytest.mark.trio
    async def test_asyncio_bridge_sync_function(self):
        """Test running sync function through asyncio bridge"""
        from workers.trio_executor import AsyncioTrioBridge
        
        def sync_function(x, y):
            return x + y
        
        result = await AsyncioTrioBridge.run_asyncio_function(sync_function, 5, 3)
        assert result == 8
    
    @pytest.mark.trio
    async def test_asyncio_bridge_async_function(self):
        """Test running async function through asyncio bridge"""
        from workers.trio_executor import AsyncioTrioBridge
        import asyncio
        
        async def async_function(x, y):
            await asyncio.sleep(0.01)  # Small delay
            return x * y
        
        result = await AsyncioTrioBridge.run_asyncio_function(async_function, 4, 6)
        assert result == 24


class TestTrioErrorHandling:
    """Test error handling in Trio implementation"""
    
    @pytest.mark.trio
    async def test_cancellation_during_execution(self):
        """Test proper cancellation handling during job execution"""
        from workers.trio_executor import TrioJobExecutor
        
        executor = TrioJobExecutor()
        
        job_cancelled = False
        
        async def cancellable_job(job_id, token, manager):
            nonlocal job_cancelled
            
            # Simulate work
            for i in range(10):
                if token.is_cancelled():
                    job_cancelled = True
                    raise trio.Cancelled()
                await trio.sleep(0.01)
            
            return "completed"
        
        with patch('workers.trio_executor.get_job_queue') as mock_get_queue:
            mock_queue = Mock()
            mock_get_queue.return_value = mock_queue
            
            async with trio.open_nursery() as nursery:
                # Start job
                nursery.start_soon(executor.execute_job, "test_job", cancellable_job, 5.0)
                
                # Give it time to start
                await trio.sleep(0.05)
                
                # Cancel it
                executor.cancel_job("test_job")
                
                # Wait for cancellation to propagate
                await trio.sleep(0.1)
        
        assert job_cancelled
    
    @pytest.mark.trio
    async def test_exception_handling_in_job(self):
        """Test exception handling in Trio job execution"""
        from workers.trio_executor import TrioJobExecutor
        
        executor = TrioJobExecutor()
        
        async def failing_job(job_id, token, manager):
            await trio.sleep(0.01)
            raise ValueError("Job failed intentionally")
        
        with patch('workers.trio_executor.get_job_queue') as mock_get_queue:
            mock_queue = Mock()
            mock_get_queue.return_value = mock_queue
            
            with pytest.raises(ValueError, match="Job failed intentionally"):
                await executor.execute_job("test_job", failing_job, 5.0)
            
            # Should have updated job status to failed
            mock_queue.update_job_status.assert_called()
            mock_queue.add_job_log.assert_called()


class TestTrioPerformance:
    """Test performance characteristics of Trio implementation"""
    
    @pytest.mark.trio
    async def test_concurrent_job_execution(self):
        """Test that multiple jobs can run concurrently"""
        from workers.trio_executor import TrioJobExecutor
        
        executor = TrioJobExecutor()
        
        job_results = {}
        
        async def test_job(job_id, token, manager):
            await trio.sleep(0.1)  # Simulate work
            job_results[job_id] = f"result_{job_id}"
            return job_results[job_id]
        
        with patch('workers.trio_executor.get_job_queue') as mock_get_queue:
            mock_queue = Mock()
            mock_get_queue.return_value = mock_queue
            
            # Start multiple jobs concurrently
            async with trio.open_nursery() as nursery:
                for i in range(3):
                    job_id = f"job_{i}"
                    nursery.start_soon(executor.execute_job, job_id, test_job, 5.0)
        
        # All jobs should have completed
        assert len(job_results) == 3
        assert "job_0" in job_results
        assert "job_1" in job_results
        assert "job_2" in job_results
    
    @pytest.mark.trio
    async def test_structured_concurrency_cleanup(self):
        """Test that structured concurrency properly cleans up resources"""
        from workers.trio_executor import TrioJobExecutor
        
        executor = TrioJobExecutor()
        
        cleanup_count = 0
        
        async def job_with_cleanup(job_id, token, manager):
            nonlocal cleanup_count
            
            def cleanup():
                nonlocal cleanup_count
                cleanup_count += 1
            
            token.add_cleanup_callback(cleanup)
            
            # Simulate some work
            await trio.sleep(0.05)
            return "completed"
        
        with patch('workers.trio_executor.get_job_queue') as mock_get_queue:
            mock_queue = Mock()
            mock_get_queue.return_value = mock_queue
            
            # Execute job
            result = await executor.execute_job("test_job", job_with_cleanup, 5.0)
            
            assert result == "completed"
            assert cleanup_count == 1  # Cleanup should have been called