"""
Tests for Trio-based video worker implementation.

This module tests the Trio video worker with structured concurrency,
cancellation handling, and pipeline integration.
"""

import os
import tempfile
from unittest.mock import Mock, patch, MagicMock, AsyncMock

import pytest
import trio

from workers.trio_video_worker import (
    process_video_job_trio,
    execute_pipeline_with_trio,
    enhance_prompt_trio,
    generate_keyframes_trio,
    generate_video_segments_trio,
    stitch_video_segments_trio,
    upload_video_to_gcs_trio,
    write_config_file,
    import_pipeline_modules,
    process_video_job_trio_wrapper
)
from workers.trio_executor import TrioCancellationToken
from api.models import JobData, JobStatus


class TestTrioVideoWorker:
    """Test Trio video worker functionality"""
    
    @pytest.mark.trio
    async def test_process_video_job_trio_success(self):
        """Test successful video job processing with Trio"""
        job_id = "test_job_123"
        
        # Mock job data
        mock_job_data = JobData(
            id=job_id,
            status=JobStatus.QUEUED,
            prompt="A beautiful sunset over mountains",
            config={"test_config": "value"}
        )
        
        with patch('workers.trio_video_worker.get_job_queue') as mock_get_queue, \
             patch('workers.trio_video_worker.execute_pipeline_with_trio') as mock_execute:
            
            mock_queue = Mock()
            mock_queue.get_job.return_value = mock_job_data
            mock_get_queue.return_value = mock_queue
            
            mock_execute.return_value = "gs://bucket/path/video.mp4"
            
            result = await process_video_job_trio(job_id)
            
            assert result == "gs://bucket/path/video.mp4"
            
            # Verify job status updates
            mock_queue.update_job_status.assert_any_call(job_id, JobStatus.STARTED)
            mock_queue.update_job_status.assert_any_call(
                job_id, JobStatus.FINISHED, progress=100, gcs_uri="gs://bucket/path/video.mp4"
            )
            
            # Verify logs
            mock_queue.add_job_log.assert_any_call(job_id, "Video generation started with Trio")
    
    @pytest.mark.trio
    async def test_process_video_job_trio_not_found(self):
        """Test handling of non-existent job"""
        job_id = "nonexistent_job"
        
        with patch('workers.trio_video_worker.get_job_queue') as mock_get_queue:
            mock_queue = Mock()
            mock_queue.get_job.return_value = None
            mock_get_queue.return_value = mock_queue
            
            with pytest.raises(ValueError, match="Job nonexistent_job not found"):
                await process_video_job_trio(job_id)
    
    @pytest.mark.trio
    async def test_process_video_job_trio_cancellation(self):
        """Test job cancellation handling"""
        job_id = "test_job_cancel"
        
        mock_job_data = JobData(
            id=job_id,
            status=JobStatus.QUEUED,
            prompt="Test prompt",
            config={}
        )
        
        with patch('workers.trio_video_worker.get_job_queue') as mock_get_queue, \
             patch('workers.trio_video_worker.check_cancellation_async') as mock_check_cancel:
            
            mock_queue = Mock()
            mock_queue.get_job.return_value = mock_job_data
            mock_get_queue.return_value = mock_queue
            
            # Simulate cancellation
            mock_check_cancel.return_value = True
            
            with pytest.raises(trio.Cancelled):
                await process_video_job_trio(job_id)
    
    @pytest.mark.trio
    async def test_process_video_job_trio_failure(self):
        """Test job failure handling"""
        job_id = "test_job_fail"
        
        mock_job_data = JobData(
            id=job_id,
            status=JobStatus.QUEUED,
            prompt="Test prompt",
            config={}
        )
        
        with patch('workers.trio_video_worker.get_job_queue') as mock_get_queue, \
             patch('workers.trio_video_worker.execute_pipeline_with_trio') as mock_execute:
            
            mock_queue = Mock()
            mock_queue.get_job.return_value = mock_job_data
            mock_get_queue.return_value = mock_queue
            
            # Simulate pipeline failure
            mock_execute.side_effect = Exception("Pipeline failed")
            
            with pytest.raises(Exception, match="Pipeline failed"):
                await process_video_job_trio(job_id)
            
            # Should update job status to failed
            mock_queue.update_job_status.assert_any_call(
                job_id, JobStatus.FAILED, error="Pipeline failed"
            )


class TestPipelineExecution:
    """Test pipeline execution with Trio"""
    
    @pytest.mark.trio
    async def test_execute_pipeline_with_trio(self):
        """Test pipeline execution with structured concurrency"""
        job_id = "test_pipeline"
        prompt = "A majestic eagle soaring"
        config = {
            "openai_api_key": "test_key",
            "gcs_bucket": "test_bucket"
        }
        
        with trio.CancelScope() as cancel_scope:
            token = TrioCancellationToken(job_id, cancel_scope)
            
            mock_queue = Mock()
            
            with patch('workers.trio_video_worker.trio.to_thread.run_sync') as mock_run_sync, \
                 patch('workers.trio_video_worker.enhance_prompt_trio') as mock_enhance, \
                 patch('workers.trio_video_worker.generate_keyframes_trio') as mock_keyframes, \
                 patch('workers.trio_video_worker.generate_video_segments_trio') as mock_videos, \
                 patch('workers.trio_video_worker.stitch_video_segments_trio') as mock_stitch, \
                 patch('workers.trio_video_worker.upload_video_to_gcs_trio') as mock_upload, \
                 patch('workers.trio_video_worker.check_cancellation_async') as mock_check_cancel:
                
                # Mock all the async operations
                mock_check_cancel.return_value = False
                mock_enhance.return_value = {
                    'keyframe_prompts': [{'prompt': 'keyframe1'}, {'prompt': 'keyframe2'}],
                    'video_prompts': [{'prompt': 'video1'}, {'prompt': 'video2'}]
                }
                mock_keyframes.return_value = ['keyframe1.png', 'keyframe2.png']
                mock_videos.return_value = ['video1.mp4', 'video2.mp4']
                mock_stitch.return_value = '/tmp/final_video.mp4'
                mock_upload.return_value = 'gs://bucket/final_video.mp4'
                
                # Mock file operations
                mock_run_sync.return_value = None
                
                async with trio.open_nursery() as nursery:
                    result = await execute_pipeline_with_trio(
                        job_id, prompt, config, token, mock_queue, nursery
                    )
                
                assert result == 'gs://bucket/final_video.mp4'
                
                # Verify all phases were called
                mock_enhance.assert_called_once()
                mock_keyframes.assert_called_once()
                mock_videos.assert_called_once()
                mock_stitch.assert_called_once()
                mock_upload.assert_called_once()


class TestPipelineComponents:
    """Test individual pipeline components"""
    
    def test_write_config_file(self):
        """Test writing configuration to file"""
        config = {"key1": "value1", "key2": {"nested": "value"}}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
            config_path = f.name
        
        try:
            write_config_file(config_path, config)
            
            # Verify file was written
            assert os.path.exists(config_path)
            
            # Verify content
            import yaml
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            assert loaded_config == config
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_import_pipeline_modules(self):
        """Test importing pipeline modules"""
        with patch('workers.trio_video_worker.pipeline') as mock_pipeline:
            mock_pipeline.load_config = Mock()
            mock_pipeline.PromptEnhancer = Mock()
            mock_pipeline.generate_keyframes = Mock()
            mock_pipeline.generate_video_segments = Mock()
            mock_pipeline.stitch_video_segments = Mock()
            mock_pipeline.PROMPT_ENHANCEMENT_INSTRUCTIONS = "test_instructions"
            
            modules = import_pipeline_modules()
            
            assert 'load_config' in modules
            assert 'PromptEnhancer' in modules
            assert 'generate_keyframes' in modules
            assert 'generate_video_segments' in modules
            assert 'stitch_video_segments' in modules
            assert 'PROMPT_ENHANCEMENT_INSTRUCTIONS' in modules
    
    @pytest.mark.trio
    async def test_enhance_prompt_trio(self):
        """Test prompt enhancement with Trio"""
        prompt = "A beautiful landscape"
        config = {"openai_api_key": "test_key"}
        
        with trio.CancelScope() as cancel_scope:
            token = TrioCancellationToken("test_job", cancel_scope)
            
            expected_result = {
                'keyframe_prompts': [{'prompt': 'enhanced1'}],
                'video_prompts': [{'prompt': 'video1'}]
            }
            
            with patch('workers.trio_video_worker.trio.to_thread.run_sync') as mock_run_sync, \
                 patch('workers.trio_video_worker.check_cancellation_async') as mock_check_cancel:
                
                mock_check_cancel.return_value = False
                mock_run_sync.return_value = expected_result
                
                async with trio.open_nursery() as nursery:
                    result = await enhance_prompt_trio(prompt, config, token, nursery)
                
                assert result == expected_result
                mock_run_sync.assert_called_once()
    
    @pytest.mark.trio
    async def test_generate_keyframes_trio(self):
        """Test keyframe generation with Trio"""
        keyframe_prompts = ["prompt1", "prompt2"]
        config = {"image_generation_model": "test_model"}
        output_dir = "/tmp/output"
        
        with trio.CancelScope() as cancel_scope:
            token = TrioCancellationToken("test_job", cancel_scope)
            mock_queue = Mock()
            
            expected_paths = ["keyframe1.png", "keyframe2.png"]
            
            with patch('workers.trio_video_worker.trio.to_thread.run_sync') as mock_run_sync, \
                 patch('workers.trio_video_worker.check_cancellation_async') as mock_check_cancel:
                
                mock_check_cancel.return_value = False
                mock_run_sync.return_value = expected_paths
                
                async with trio.open_nursery() as nursery:
                    result = await generate_keyframes_trio(
                        keyframe_prompts, config, output_dir, token, 
                        mock_queue, "test_job", 30, 50, nursery
                    )
                
                assert result == expected_paths
                mock_queue.update_job_status.assert_called()
    
    @pytest.mark.trio
    async def test_generate_video_segments_trio(self):
        """Test video segment generation with Trio"""
        video_prompts = [{"prompt": "video1"}, {"prompt": "video2"}]
        config = {"single_keyframe_mode": False}
        output_dir = "/tmp/output"
        
        with trio.CancelScope() as cancel_scope:
            token = TrioCancellationToken("test_job", cancel_scope)
            mock_queue = Mock()
            
            expected_paths = ["video1.mp4", "video2.mp4"]
            
            with patch('workers.trio_video_worker.trio.to_thread.run_sync') as mock_run_sync, \
                 patch('workers.trio_video_worker.check_cancellation_async') as mock_check_cancel:
                
                mock_check_cancel.return_value = False
                mock_run_sync.return_value = expected_paths
                
                async with trio.open_nursery() as nursery:
                    result = await generate_video_segments_trio(
                        video_prompts, config, output_dir, token,
                        mock_queue, "test_job", 50, 80, nursery
                    )
                
                assert result == expected_paths
                mock_queue.update_job_status.assert_called()
    
    @pytest.mark.trio
    async def test_stitch_video_segments_trio(self):
        """Test video stitching with Trio"""
        video_paths = ["video1.mp4", "video2.mp4"]
        final_video_path = "/tmp/final_video.mp4"
        
        with trio.CancelScope() as cancel_scope:
            token = TrioCancellationToken("test_job", cancel_scope)
            
            with patch('workers.trio_video_worker.trio.to_thread.run_sync') as mock_run_sync:
                mock_run_sync.return_value = final_video_path
                
                async with trio.open_nursery() as nursery:
                    result = await stitch_video_segments_trio(
                        video_paths, final_video_path, token, nursery
                    )
                
                assert result == final_video_path
                mock_run_sync.assert_called_once()
    
    @pytest.mark.trio
    async def test_upload_video_to_gcs_trio(self):
        """Test GCS upload with Trio"""
        video_path = "/tmp/final_video.mp4"
        job_id = "test_job"
        config = {
            "gcs_bucket": "test_bucket",
            "gcs_prefix": "test_prefix",
            "credentials_path": "creds.json"
        }
        
        with trio.CancelScope() as cancel_scope:
            token = TrioCancellationToken(job_id, cancel_scope)
            
            expected_uri = "gs://test_bucket/test_prefix/final_video.mp4"
            
            with patch('workers.trio_video_worker.trio.to_thread.run_sync') as mock_run_sync:
                mock_run_sync.return_value = expected_uri
                
                async with trio.open_nursery() as nursery:
                    result = await upload_video_to_gcs_trio(
                        video_path, job_id, config, token, nursery
                    )
                
                assert result == expected_uri
                mock_run_sync.assert_called_once()


class TestTrioWrapper:
    """Test Trio wrapper function"""
    
    def test_process_video_job_trio_wrapper(self):
        """Test the sync wrapper for Trio job processing"""
        job_id = "test_wrapper_job"
        expected_result = "gs://bucket/video.mp4"
        
        with patch('workers.trio_video_worker.trio.run') as mock_trio_run:
            mock_trio_run.return_value = expected_result
            
            result = process_video_job_trio_wrapper(job_id)
            
            assert result == expected_result
            mock_trio_run.assert_called_once_with(
                process_video_job_trio, job_id
            )


class TestCancellationHandling:
    """Test cancellation handling in Trio video worker"""
    
    @pytest.mark.trio
    async def test_cancellation_during_enhancement(self):
        """Test cancellation during prompt enhancement"""
        job_id = "test_cancel_enhance"
        
        mock_job_data = JobData(
            id=job_id,
            status=JobStatus.QUEUED,
            prompt="Test prompt",
            config={}
        )
        
        with patch('workers.trio_video_worker.get_job_queue') as mock_get_queue, \
             patch('workers.trio_video_worker.check_cancellation_async') as mock_check_cancel:
            
            mock_queue = Mock()
            mock_queue.get_job.return_value = mock_job_data
            mock_get_queue.return_value = mock_queue
            
            # First call returns False, second returns True (cancellation)
            mock_check_cancel.side_effect = [False, True]
            
            with pytest.raises(trio.Cancelled):
                await process_video_job_trio(job_id)
    
    @pytest.mark.trio
    async def test_cancellation_cleanup(self):
        """Test that cancellation properly cleans up resources"""
        job_id = "test_cancel_cleanup"
        
        mock_job_data = JobData(
            id=job_id,
            status=JobStatus.QUEUED,
            prompt="Test prompt",
            config={}
        )
        
        cleanup_called = False
        
        def mock_cleanup():
            nonlocal cleanup_called
            cleanup_called = True
        
        with patch('workers.trio_video_worker.get_job_queue') as mock_get_queue, \
             patch('workers.trio_video_worker.cleanup_temp_files_async', side_effect=mock_cleanup) as mock_cleanup_func:
            
            mock_queue = Mock()
            mock_queue.get_job.return_value = mock_job_data
            mock_get_queue.return_value = mock_queue
            
            # Mock execute_pipeline_with_trio to raise cancellation
            with patch('workers.trio_video_worker.execute_pipeline_with_trio') as mock_execute:
                mock_execute.side_effect = trio.Cancelled()
                
                with pytest.raises(trio.Cancelled):
                    await process_video_job_trio(job_id)
        
        # Cleanup should have been called
        assert cleanup_called


class TestErrorHandling:
    """Test error handling in Trio video worker"""
    
    @pytest.mark.trio
    async def test_pipeline_execution_error(self):
        """Test handling of pipeline execution errors"""
        job_id = "test_error"
        
        mock_job_data = JobData(
            id=job_id,
            status=JobStatus.QUEUED,
            prompt="Test prompt",
            config={}
        )
        
        with patch('workers.trio_video_worker.get_job_queue') as mock_get_queue:
            mock_queue = Mock()
            mock_queue.get_job.return_value = mock_job_data
            mock_get_queue.return_value = mock_queue
            
            with patch('workers.trio_video_worker.execute_pipeline_with_trio') as mock_execute:
                mock_execute.side_effect = RuntimeError("Pipeline error")
                
                with pytest.raises(RuntimeError, match="Pipeline error"):
                    await process_video_job_trio(job_id)
                
                # Should update job status to failed
                mock_queue.update_job_status.assert_any_call(
                    job_id, JobStatus.FAILED, error="Pipeline error"
                )
    
    @pytest.mark.trio
    async def test_job_status_update_error(self):
        """Test handling of job status update errors"""
        job_id = "test_status_error"
        
        mock_job_data = JobData(
            id=job_id,
            status=JobStatus.QUEUED,
            prompt="Test prompt",
            config={}
        )
        
        with patch('workers.trio_video_worker.get_job_queue') as mock_get_queue:
            mock_queue = Mock()
            mock_queue.get_job.return_value = mock_job_data
            mock_queue.update_job_status.side_effect = Exception("Status update failed")
            mock_get_queue.return_value = mock_queue
            
            with patch('workers.trio_video_worker.execute_pipeline_with_trio') as mock_execute:
                mock_execute.side_effect = RuntimeError("Pipeline error")
                
                # Should not raise the status update error, only the original error
                with pytest.raises(RuntimeError, match="Pipeline error"):
                    await process_video_job_trio(job_id)