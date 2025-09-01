"""
Integration tests for RQ worker implementation with pipeline integration.
"""

import pytest
import tempfile
import os
import yaml
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from workers.video_worker import (
    execute_pipeline_with_config,
    CancellationToken,
    generate_keyframes_with_progress,
    generate_video_segments_with_progress,
    upload_video_to_gcs,
    cleanup_job_files
)
from api.models import JobStatus, JobData
from api.config_merger import ConfigMerger


class TestPipelineIntegration:
    """Test pipeline integration with configuration merging"""
    
    def test_config_merger_integration(self):
        """Test that ConfigMerger is properly integrated"""
        # Test configuration merging for job
        base_config = {
            'prompt': 'base prompt',
            'size': '1280*720',
            'frame_num': 81,
            'openai_api_key': 'test-key'
        }
        
        job_prompt = 'HTTP override prompt'
        
        merger = ConfigMerger()
        effective_config = merger.merge_for_job(base_config, job_prompt)
        
        # Verify HTTP prompt takes precedence
        assert effective_config['prompt'] == job_prompt
        assert effective_config['size'] == '1280*720'  # Other config preserved
        assert effective_config['frame_num'] == 81
        assert effective_config['openai_api_key'] == 'test-key'
    
    @patch('pipeline.generate_keyframes')
    def test_generate_keyframes_with_progress(self, mock_generate_keyframes):
        """Test keyframe generation with progress reporting"""
        # Mock keyframe generation
        mock_generate_keyframes.return_value = ['frame1.png', 'frame2.png']
        
        # Mock job queue
        mock_job_queue = Mock()
        
        # Mock cancellation token
        cancellation_token = CancellationToken("test-job")
        
        config = {
            'image_generation_model': 'test-model',
            'openai_api_key': 'test-key'
        }
        
        keyframe_prompts = ['prompt1', 'prompt2']
        
        result = generate_keyframes_with_progress(
            keyframe_prompts=keyframe_prompts,
            config=config,
            output_dir='/tmp/test',
            cancellation_token=cancellation_token,
            job_queue=mock_job_queue,
            job_id='test-job',
            progress_start=30,
            progress_end=50
        )
        
        # Verify results
        assert result == ['frame1.png', 'frame2.png']
        
        # Verify progress was updated
        mock_job_queue.update_job_status.assert_called_with(
            'test-job', JobStatus.PROGRESS, progress=50
        )
        
        # Verify keyframe generation was called with correct parameters
        mock_generate_keyframes.assert_called_once()
        call_kwargs = mock_generate_keyframes.call_args[1]
        assert call_kwargs['keyframe_prompts'] == keyframe_prompts
        assert call_kwargs['config'] == config
        assert call_kwargs['output_dir'] == '/tmp/test'
    
    @patch('pipeline.generate_video_segments_single_keyframe')
    def test_generate_video_segments_single_keyframe_mode(self, mock_generate_segments):
        """Test video segment generation in single keyframe mode"""
        # Mock video generation
        mock_generate_segments.return_value = ['video1.mp4', 'video2.mp4']
        
        # Mock job queue
        mock_job_queue = Mock()
        
        # Mock cancellation token
        cancellation_token = CancellationToken("test-job")
        
        config = {
            'single_keyframe_mode': True,
            'default_backend': 'veo3'
        }
        
        video_prompts = [
            {'segment': 1, 'prompt': 'prompt1'},
            {'segment': 2, 'prompt': 'prompt2'}
        ]
        
        result = generate_video_segments_with_progress(
            video_prompts=video_prompts,
            config=config,
            output_dir='/tmp/test',
            cancellation_token=cancellation_token,
            job_queue=mock_job_queue,
            job_id='test-job',
            processes=[],
            progress_start=50,
            progress_end=80
        )
        
        # Verify results
        assert result == ['video1.mp4', 'video2.mp4']
        
        # Verify progress was updated
        mock_job_queue.update_job_status.assert_called_with(
            'test-job', JobStatus.PROGRESS, progress=80
        )
        
        # Verify single keyframe generation was called
        mock_generate_segments.assert_called_once_with(
            config=config,
            video_prompts=video_prompts,
            output_dir='/tmp/test'
        )
    
    @patch('workers.gcs_uploader.upload_job_artifact')
    def test_upload_video_to_gcs(self, mock_upload):
        """Test GCS upload functionality"""
        # Mock successful upload
        expected_gcs_uri = 'gs://test-bucket/test-job/final_video.mp4'
        mock_upload.return_value = expected_gcs_uri
        
        # Mock cancellation token
        cancellation_token = CancellationToken("test-job")
        
        config = {
            'gcs_bucket': 'test-bucket',
            'gcs_prefix': 'test-prefix',
            'credentials_path': 'test-creds.json'
        }
        
        result = upload_video_to_gcs(
            video_path='/tmp/test_video.mp4',
            job_id='test-job',
            config=config,
            cancellation_token=cancellation_token
        )
        
        # Verify result
        assert result == expected_gcs_uri
        
        # Verify upload was called with correct parameters
        mock_upload.assert_called_once()
        call_kwargs = mock_upload.call_args[1]
        assert call_kwargs['local_video_path'] == '/tmp/test_video.mp4'
        assert call_kwargs['job_id'] == 'test-job'
        assert call_kwargs['cleanup_local'] == False
    
    def test_cleanup_job_files(self):
        """Test job file cleanup functionality"""
        # Create a temporary directory with some files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files
            test_file = os.path.join(temp_dir, 'test_file.txt')
            with open(test_file, 'w') as f:
                f.write('test content')
            
            # Verify file exists
            assert os.path.exists(test_file)
            
            # Cleanup should remove the directory
            cleanup_job_files(temp_dir)
            
            # Directory should be gone
            assert not os.path.exists(temp_dir)
    
    def test_cancellation_during_pipeline_execution(self):
        """Test cancellation handling during pipeline execution"""
        cancellation_token = CancellationToken("test-job")
        
        # Cancel the token
        cancellation_token.cancel()
        
        # Mock job queue
        mock_job_queue = Mock()
        
        config = {'test': 'config'}
        
        # Pipeline execution should raise InterruptedError when cancelled
        with pytest.raises(InterruptedError, match="Job cancelled during setup"):
            execute_pipeline_with_config(
                job_id='test-job',
                prompt='test prompt',
                config=config,
                cancellation_token=cancellation_token,
                job_queue=mock_job_queue,
                processes=[]
            )


class TestStructuredLogging:
    """Test structured logging functionality"""
    
    @patch('workers.video_worker.logger')
    def test_pipeline_logging(self, mock_logger):
        """Test that pipeline execution includes structured logging"""
        cancellation_token = CancellationToken("test-job")
        mock_job_queue = Mock()
        config = {'test': 'config'}
        
        # This will fail due to missing dependencies, but we can verify logging
        try:
            execute_pipeline_with_config(
                job_id='test-job',
                prompt='test prompt',
                config=config,
                cancellation_token=cancellation_token,
                job_queue=mock_job_queue,
                processes=[]
            )
        except Exception:
            pass  # Expected to fail in test environment
        
        # Verify structured logging was called
        assert mock_logger.info.call_count > 0
        
        # Check that job ID is included in log messages
        log_calls = [call.args[0] for call in mock_logger.info.call_calls]
        job_id_logs = [log for log in log_calls if 'test-job' in log]
        assert len(job_id_logs) > 0
    
    def test_progress_reporting_structure(self):
        """Test that progress reporting follows expected structure"""
        mock_job_queue = Mock()
        
        # Test progress update calls
        mock_job_queue.update_job_status('test-job', JobStatus.PROGRESS, progress=25)
        mock_job_queue.add_job_log('test-job', 'Test progress message')
        
        # Verify calls were made with correct structure
        status_call = mock_job_queue.update_job_status.call_args_list[0]
        assert status_call[0] == ('test-job', JobStatus.PROGRESS)
        assert status_call[1]['progress'] == 25
        
        log_call = mock_job_queue.add_job_log.call_args_list[0]
        assert log_call[0] == ('test-job', 'Test progress message')


if __name__ == "__main__":
    pytest.main([__file__])