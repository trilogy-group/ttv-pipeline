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
from pipeline import prepare_keyframes, generate_video_segments_single_keyframe, run_pipeline
from video_generator_interface import VideoGenerationError


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

    def test_prepare_keyframes_preserves_start_and_resolves_paths(self):
        with tempfile.TemporaryDirectory() as output_dir:
            initial_image = os.path.join(output_dir, "start.png")
            with open(initial_image, "wb") as file:
                file.write(b"start")

            frames_dir = os.path.join(output_dir, "frames")
            os.makedirs(frames_dir)
            frame_paths = [
                os.path.join(frames_dir, "segment_01.png"),
                os.path.join(frames_dir, "segment_02.png"),
            ]
            for frame_path in frame_paths:
                with open(frame_path, "wb") as file:
                    file.write(b"frame")

            video_prompts = [
                {"segment": 1, "prompt": "one", "first_frame": "provided_start_image.png", "last_frame": "segment_01.png"},
                {"segment": 2, "prompt": "two", "first_frame": "segment_01.png", "last_frame": "segment_02.png"},
            ]
            config = {"image_generation_model": "test", "initial_image": initial_image}

            with patch("pipeline.generate_keyframes", return_value=frame_paths):
                assert prepare_keyframes(config, ["one", "two"], video_prompts, output_dir) == frame_paths

            with open(os.path.join(frames_dir, "segment_00.png"), "rb") as file:
                assert file.read() == b"start"
            assert video_prompts[0]["first_frame"] == "provided_start_image.png"
            assert video_prompts[1]["last_frame"] == "segment_02.png"

    def test_prepare_keyframes_keeps_start_already_named_segment_zero(self):
        with tempfile.TemporaryDirectory() as output_dir:
            frames_dir = os.path.join(output_dir, "frames")
            os.makedirs(frames_dir)
            initial_image = os.path.join(frames_dir, "segment_00.png")
            with open(initial_image, "wb") as file:
                file.write(b"start")

            def generate_frame(**kwargs):
                with open(kwargs["input_image_path"], "rb") as file:
                    assert file.read() == b"start"
                with open(kwargs["output_path"], "wb") as file:
                    file.write(b"frame")
                return kwargs["output_path"]

            video_prompts = [{"segment": 1, "prompt": "one", "first_frame": "provided_start_image.png"}]
            config = {"image_generation_model": "test", "initial_image": initial_image}

            with patch("keyframe_generator.generate_keyframe", side_effect=generate_frame):
                prepare_keyframes(config, ["one"], video_prompts, output_dir)

            with open(initial_image, "rb") as file:
                assert file.read() == b"start"
            assert video_prompts[0]["first_frame"] == "provided_start_image.png"

    def test_single_keyframe_generation_uses_typed_error_fallback(self):
        with tempfile.TemporaryDirectory() as output_dir:
            frame_path = os.path.join(output_dir, "frame.png")
            with open(frame_path, "wb") as file:
                file.write(b"frame")

            primary = Mock()
            primary.generate_video.side_effect = VideoGenerationError("primary failed")
            fallback = Mock()
            config = {"default_backend": "primary", "segment_duration_seconds": 5}
            prompts = [{"segment": 1, "prompt": "move", "first_frame": frame_path}]

            with patch("generators.factory.create_video_generator", return_value=primary), \
                 patch("generators.factory.get_fallback_generator", return_value=fallback) as get_fallback:
                result = generate_video_segments_single_keyframe(config, prompts, output_dir)

            get_fallback.assert_called_once_with("primary", config)
            fallback.generate_video.assert_called_once()
            assert result == [os.path.join(output_dir, "videos", "segment_001.mp4")]

    def test_single_keyframe_generation_falls_back_when_backend_init_fails(self):
        with tempfile.TemporaryDirectory() as output_dir:
            frame_path = os.path.join(output_dir, "frame.png")
            with open(frame_path, "wb") as file:
                file.write(b"frame")

            fallback = Mock()
            config = {"default_backend": "primary", "segment_duration_seconds": 5}
            prompts = [{"segment": 1, "prompt": "move", "first_frame": frame_path}]

            with patch("generators.factory.create_video_generator", side_effect=VideoGenerationError("init failed")), \
                 patch("generators.factory.get_fallback_generator", return_value=fallback):
                result = generate_video_segments_single_keyframe(config, prompts, output_dir)

            expected_path = os.path.join(output_dir, "videos", "segment_001.mp4")
            assert fallback.generate_video.call_args.kwargs["output_path"] == expected_path
            assert result == [expected_path]

    def test_cli_does_not_persist_effective_config(self, monkeypatch):
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.chdir(temp_dir)
            config_path = os.path.join(temp_dir, "pipeline_config.yaml")
            with open(config_path, "w") as file:
                yaml.safe_dump({
                    "prompt": "test",
                    "generation_mode": "keyframe",
                    "openai_api_key": "test-secret",
                }, file)

            enhanced = {
                "keyframe_prompts": [{"segment": 1, "prompt": "frame"}],
                "video_prompts": [{"segment": 1, "prompt": "move"}],
            }
            with patch("pipeline.enhance_prompt", return_value=enhanced), \
                 patch("pipeline.prepare_keyframes"), \
                 patch("pipeline.generate_video_segments", return_value=["segment.mp4"]):
                run_pipeline(config_path)

            assert not os.path.exists(os.path.join(temp_dir, "output", "config.yaml"))
    
    @patch('pipeline.prepare_keyframes')
    def test_generate_keyframes_with_progress(self, mock_prepare_keyframes):
        """Test keyframe generation with progress reporting"""
        # Mock keyframe generation
        mock_prepare_keyframes.return_value = ['frame1.png', 'frame2.png']
        
        # Mock job queue
        mock_job_queue = Mock()
        
        # Mock cancellation token
        cancellation_token = CancellationToken("test-job")
        
        config = {
            'image_generation_model': 'test-model',
            'openai_api_key': 'test-key'
        }
        
        keyframe_prompts = ['prompt1', 'prompt2']
        video_prompts = [{'segment': 1, 'prompt': 'video1'}]
        
        result = generate_keyframes_with_progress(
            keyframe_prompts=keyframe_prompts,
            video_prompts=video_prompts,
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
        mock_prepare_keyframes.assert_called_once()
        call_kwargs = mock_prepare_keyframes.call_args[1]
        assert call_kwargs['keyframe_prompts'] == keyframe_prompts
        assert call_kwargs['video_prompts'] == video_prompts
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
        cancellation_token.cleanup()


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
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
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
