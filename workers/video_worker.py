"""
Video generation worker for processing jobs from the Redis queue.

This module contains the worker function that will be called by RQ
to process video generation jobs with cancellation support and
pipeline integration with effective configuration merging.

Supports both traditional threading-based execution and Trio-based
structured concurrency for optimal resource management.
"""

import logging
import time
import signal
import os
import subprocess
import threading
import tempfile
import yaml
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from pathlib import Path

from api.queue import get_job_queue, initialize_queue_infrastructure
from api.models import JobStatus
from api.config_merger import ConfigMerger
from api.config import get_config_from_env

# Import Trio components with fallback
try:
    import trio
    from .trio_executor import get_trio_executor, initialize_trio_executor
    from .trio_video_worker import process_video_job_trio_wrapper
    TRIO_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Trio structured concurrency available")
except ImportError as e:
    TRIO_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Trio not available, falling back to threading: {e}")

logger = logging.getLogger(__name__)

# Global flag to track if queue infrastructure is initialized
_queue_initialized = False

def ensure_queue_initialized():
    """Ensure queue infrastructure is initialized for workers."""
    global _queue_initialized
    
    if not _queue_initialized:
        try:
            # Try to get existing queue first
            get_job_queue()
            _queue_initialized = True
            logger.info("Queue infrastructure already initialized")
        except RuntimeError:
            # Initialize if not already done
            logger.info("Initializing queue infrastructure for worker...")
            config = get_config_from_env()
            redis_manager, job_queue = initialize_queue_infrastructure(config.redis)
            _queue_initialized = True
            logger.info("Queue infrastructure initialized successfully")

# Global cancellation flag for cooperative cancellation
_cancellation_flags = {}
_cancellation_lock = threading.Lock()


def _record_job_metrics(processing_time: float, success: bool = True):
    """
    Record job processing metrics for observability.
    
    Args:
        processing_time: Time taken to process the job in seconds
        success: Whether the job completed successfully
    """
    try:
        from api.routes.health import record_job_processed
        record_job_processed(processing_time)
        
        logger.info(
            f"Job processing metrics recorded: {processing_time:.2f}s, success={success}",
            extra={
                'extra': {
                    'event': 'job_metrics',
                    'processing_time': processing_time,
                    'success': success,
                    'job_type': 'video_generation'
                }
            }
        )
    except ImportError:
        logger.debug("Health metrics module not available for job metrics recording")
    except Exception as e:
        logger.warning(f"Failed to record job metrics: {e}")


class CancellationToken:
    """Token for cooperative cancellation"""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self._cancelled = False
    
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested"""
        with _cancellation_lock:
            return _cancellation_flags.get(self.job_id, False) or self._cancelled
    
    def cancel(self):
        """Request cancellation"""
        with _cancellation_lock:
            _cancellation_flags[self.job_id] = True
            self._cancelled = True
    
    def cleanup(self):
        """Clean up cancellation flag"""
        with _cancellation_lock:
            _cancellation_flags.pop(self.job_id, None)


def set_job_cancellation_flag(job_id: str):
    """Set cancellation flag for a job (called from queue manager)"""
    with _cancellation_lock:
        _cancellation_flags[job_id] = True
        logger.info(f"Cancellation flag set for job {job_id}")


@contextmanager
def process_manager(job_id: str):
    """Context manager for subprocess management with cancellation support"""
    processes = []
    cancellation_token = CancellationToken(job_id)
    
    def signal_handler(signum, frame):
        """Handle cancellation signals"""
        logger.info(f"Received signal {signum} for job {job_id}")
        cancellation_token.cancel()
        
        # Terminate all child processes
        for proc in processes:
            if proc.poll() is None:  # Process is still running
                logger.info(f"Terminating process {proc.pid} for job {job_id}")
                try:
                    proc.terminate()
                    # Give process 5 seconds to terminate gracefully
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Process {proc.pid} did not terminate gracefully, killing")
                    proc.kill()
                    proc.wait()
                except Exception as e:
                    logger.error(f"Error terminating process {proc.pid}: {e}")
    
    # Set up signal handlers for graceful shutdown
    old_sigterm = signal.signal(signal.SIGTERM, signal_handler)
    old_sigint = signal.signal(signal.SIGINT, signal_handler)
    
    try:
        yield cancellation_token, processes
    finally:
        # Restore original signal handlers
        signal.signal(signal.SIGTERM, old_sigterm)
        signal.signal(signal.SIGINT, old_sigint)
        
        # Clean up any remaining processes
        for proc in processes:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except:
                    try:
                        proc.kill()
                        proc.wait()
                    except:
                        pass
        
        # Clean up cancellation flag
        cancellation_token.cleanup()


def check_cancellation(cancellation_token: CancellationToken, job_id: str) -> bool:
    """
    Check if job should be cancelled and handle it
    
    Returns:
        True if job was cancelled, False otherwise
    """
    if cancellation_token.is_cancelled():
        logger.info(f"Job {job_id} cancellation detected")
        
        try:
            job_queue = get_job_queue()
            job_queue.update_job_status(job_id, JobStatus.CANCELED)
            job_queue.add_job_log(job_id, "Job cancelled during processing")
        except Exception as e:
            logger.error(f"Failed to update cancelled job status for {job_id}: {e}")
        
        return True
    
    return False


def _ensure_queue_infrastructure():
    """
    Ensure queue infrastructure is initialized for worker processes.
    
    This function initializes the queue infrastructure if it hasn't been
    initialized yet. It's called at the start of each job to ensure
    the worker has access to the job queue.
    """
    from api.queue import get_job_queue, initialize_queue_infrastructure
    from api.config import RedisConfig, get_config_from_env
    
    try:
        # Try to get existing queue instance
        get_job_queue()
        logger.debug("Queue infrastructure already initialized")
        return
    except RuntimeError:
        # Queue not initialized, initialize it now
        logger.info("Initializing queue infrastructure for worker")
        
        try:
            # Get configuration from environment
            config = get_config_from_env()
            
            # Initialize queue infrastructure
            initialize_queue_infrastructure(config.redis)
            logger.info("Queue infrastructure initialized successfully in worker")
            
        except Exception as e:
            logger.error(f"Failed to initialize queue infrastructure in worker: {e}")
            raise


def process_video_job(job_id: str, use_trio: bool = True) -> str:
    """
    Process a video generation job with cancellation support and pipeline integration.
    
    This function is called by RQ workers to process jobs from the queue.
    It supports both Trio-based structured concurrency and traditional threading
    for optimal resource management and cancellation handling.
    
    Args:
        job_id: The unique job identifier
        use_trio: Whether to use Trio structured concurrency (default: True)
        
    Returns:
        The GCS URI of the generated video
        
    Raises:
        Exception: If job processing fails
    """
    logger.info(f"Starting video generation job {job_id}")
    start_time = time.time()
    
    # Ensure queue infrastructure is initialized
    ensure_queue_initialized()
    _ensure_queue_infrastructure()
    
    try:
        # Use Trio structured concurrency if available and requested
        if TRIO_AVAILABLE and use_trio:
            try:
                logger.info(f"Using Trio structured concurrency for job {job_id}")
                result = process_video_job_trio_wrapper(job_id)
                
                # Record successful job processing metrics
                processing_time = time.time() - start_time
                _record_job_metrics(processing_time, success=True)
                
                return result
            except Exception as e:
                logger.warning(f"Trio execution failed for job {job_id}, falling back to threading: {e}")
                # Fall through to threading-based execution
        
        # Fallback to traditional threading-based execution
        logger.info(f"Using threading-based execution for job {job_id}")
        result = process_video_job_threading(job_id)
        
        # Record successful job processing metrics
        processing_time = time.time() - start_time
        _record_job_metrics(processing_time, success=True)
        
        return result
        
    except Exception as e:
        # Record failed job processing metrics
        processing_time = time.time() - start_time
        _record_job_metrics(processing_time, success=False)
        raise


def process_video_job_threading(job_id: str) -> str:
    """
    Process a video generation job using traditional threading approach.
    
    This is the fallback implementation when Trio is not available or fails.
    
    Args:
        job_id: The unique job identifier
        
    Returns:
        The GCS URI of the generated video
        
    Raises:
        Exception: If job processing fails
    """
    logger.info(f"Starting threading-based video generation job {job_id}")
    
    # Ensure queue infrastructure is initialized
    ensure_queue_initialized()
    
    with process_manager(job_id) as (cancellation_token, processes):
        try:
            # Get job queue instance
            job_queue = get_job_queue()
            
            # Get job data
            job_data = job_queue.get_job(job_id)
            if not job_data:
                raise ValueError(f"Job {job_id} not found")
            
            # Check for early cancellation
            if check_cancellation(cancellation_token, job_id):
                raise InterruptedError("Job cancelled before processing started")
            
            # Update status to started
            job_queue.update_job_status(job_id, JobStatus.STARTED)
            job_queue.add_job_log(job_id, "Video generation started (threading mode)")
            
            # Extract configuration and prompt
            prompt = job_data.prompt
            effective_config = job_data.config
            
            logger.info(f"Processing job {job_id} with prompt: {prompt[:100]}...")
            logger.info(f"Using effective configuration with {len(effective_config)} parameters")
            
            # Execute the pipeline with the effective configuration
            gcs_uri = execute_pipeline_with_config(
                job_id=job_id,
                prompt=prompt,
                config=effective_config,
                cancellation_token=cancellation_token,
                job_queue=job_queue,
                processes=processes
            )
            
            # Mark as completed
            job_queue.update_job_status(
                job_id, 
                JobStatus.FINISHED, 
                progress=100, 
                gcs_uri=gcs_uri
            )
            job_queue.add_job_log(job_id, f"Video generation completed: {gcs_uri}")
            
            logger.info(f"Job {job_id} completed successfully: {gcs_uri}")
            return gcs_uri
            
        except InterruptedError as e:
            logger.info(f"Job {job_id} was cancelled: {e}")
            # Job status already updated in check_cancellation
            raise
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            
            # Update job status to failed (unless it was already cancelled)
            try:
                job_queue = get_job_queue()
                current_job = job_queue.get_job(job_id)
                if current_job and current_job.status != JobStatus.CANCELED:
                    job_queue.update_job_status(
                        job_id, 
                        JobStatus.FAILED, 
                        error=str(e)
                    )
                    job_queue.add_job_log(job_id, f"Job failed: {e}")
            except Exception as update_error:
                logger.error(f"Failed to update job status for {job_id}: {update_error}")
            
            raise


def execute_pipeline_with_config(
    job_id: str,
    prompt: str,
    config: Dict[str, Any],
    cancellation_token: CancellationToken,
    job_queue,
    processes: List
) -> str:
    """
    Execute the video generation pipeline with effective configuration merging.
    
    This function integrates with the existing pipeline.py to generate videos
    using the merged configuration with proper progress reporting and cancellation.
    
    Args:
        job_id: The unique job identifier
        prompt: The video generation prompt
        config: Effective configuration dictionary
        cancellation_token: Cancellation token for cooperative cancellation
        job_queue: Job queue instance for status updates
        processes: List to track subprocess for cancellation
        
    Returns:
        The GCS URI of the generated video
        
    Raises:
        InterruptedError: If job is cancelled
        Exception: If pipeline execution fails
    """
    logger.info(f"Executing pipeline for job {job_id}")
    
    # Create temporary working directory for this job
    with tempfile.TemporaryDirectory(prefix=f"job_{job_id}_") as temp_dir:
        logger.info(f"Using temporary directory: {temp_dir}")
        
        # Create job-specific output directory
        job_output_dir = os.path.join(temp_dir, "output")
        os.makedirs(job_output_dir, exist_ok=True)
        
        try:
            # Phase 1: Setup and validation (5%)
            if check_cancellation(cancellation_token, job_id):
                raise InterruptedError("Job cancelled during setup")
            
            job_queue.update_job_status(job_id, JobStatus.PROGRESS, progress=5)
            job_queue.add_job_log(job_id, "Setting up pipeline configuration")
            
            # Write job-specific configuration to temporary file for logging
            job_config_path = os.path.join(temp_dir, "job_config.yaml")
            with open(job_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Job configuration written to {job_config_path}")
            
            # Phase 2: Import and initialize pipeline (10%)
            if check_cancellation(cancellation_token, job_id):
                raise InterruptedError("Job cancelled during pipeline initialization")
            
            job_queue.update_job_status(job_id, JobStatus.PROGRESS, progress=10)
            job_queue.add_job_log(job_id, "Initializing video generation pipeline")
            
            # Import pipeline functions
            from pipeline import (
                load_config, PromptEnhancer, generate_keyframes, 
                generate_video_segments, stitch_video_segments,
                PROMPT_ENHANCEMENT_INSTRUCTIONS
            )
            
            # Load base configuration from mounted pipeline_config.yaml
            base_config_path = "/app/pipeline_config.yaml"
            if not os.path.exists(base_config_path):
                logger.error(f"Base pipeline config not found at {base_config_path}")
                raise FileNotFoundError(f"Pipeline configuration not found at {base_config_path}")
            
            base_config = load_config(base_config_path)
            
            # Use ConfigMerger to ensure proper precedence (HTTP prompt overrides config)
            config_merger = ConfigMerger()
            final_config = config_merger.merge_for_job(base_config, prompt)
            
            logger.info(f"Final configuration merged with prompt override")
            logger.info(f"Image generation model from config: {final_config.get('image_generation_model')}")
            logger.info(f"Gemini API key present: {'gemini_api_key' in final_config and bool(final_config.get('gemini_api_key'))}")
            logger.info(f"Available config keys: {list(final_config.keys())[:20]}...")  # Log first 20 keys
            
            # Phase 3: Prompt enhancement and segmentation (20%)
            if check_cancellation(cancellation_token, job_id):
                raise InterruptedError("Job cancelled during prompt enhancement")
            
            job_queue.update_job_status(job_id, JobStatus.PROGRESS, progress=20)
            job_queue.add_job_log(job_id, "Enhancing and segmenting prompt")
            
            # Initialize prompt enhancer
            enhancer = PromptEnhancer(
                api_key=final_config.get('openai_api_key'),
                base_url=final_config.get('openai_base_url', 'https://api.openai.com/v1'),
                model=final_config.get('prompt_enhancement_model', 'gpt-4o-mini')
            )
            
            # Enhance and segment the prompt
            enhancement_result = enhancer.enhance(PROMPT_ENHANCEMENT_INSTRUCTIONS, prompt)
            
            keyframe_prompts = enhancement_result['keyframe_prompts']
            video_prompts = enhancement_result['video_prompts']
            
            logger.info(f"Generated {len(keyframe_prompts)} keyframe prompts and {len(video_prompts)} video prompts")
            
            # Phase 4: Keyframe generation (50%)
            if check_cancellation(cancellation_token, job_id):
                raise InterruptedError("Job cancelled before keyframe generation")
            
            job_queue.update_job_status(job_id, JobStatus.PROGRESS, progress=30)
            job_queue.add_job_log(job_id, f"Generating {len(keyframe_prompts)} keyframes")
            
            # Generate keyframes with progress tracking
            keyframe_paths = generate_keyframes_with_progress(
                keyframe_prompts=keyframe_prompts,
                config=final_config,
                output_dir=job_output_dir,
                cancellation_token=cancellation_token,
                job_queue=job_queue,
                job_id=job_id,
                progress_start=30,
                progress_end=50
            )
            
            logger.info(f"Generated {len(keyframe_paths)} keyframes")
            
            # Phase 5: Video segment generation (80%)
            if check_cancellation(cancellation_token, job_id):
                raise InterruptedError("Job cancelled before video generation")
            
            job_queue.update_job_status(job_id, JobStatus.PROGRESS, progress=50)
            job_queue.add_job_log(job_id, f"Generating {len(video_prompts)} video segments")
            
            # Generate video segments with progress tracking
            video_paths = generate_video_segments_with_progress(
                video_prompts=video_prompts,
                config=final_config,
                output_dir=job_output_dir,
                cancellation_token=cancellation_token,
                job_queue=job_queue,
                job_id=job_id,
                processes=processes,
                progress_start=50,
                progress_end=80
            )
            
            logger.info(f"Generated {len(video_paths)} video segments")
            
            # Phase 6: Video stitching (90%)
            if check_cancellation(cancellation_token, job_id):
                raise InterruptedError("Job cancelled before video stitching")
            
            job_queue.update_job_status(job_id, JobStatus.PROGRESS, progress=80)
            job_queue.add_job_log(job_id, "Stitching video segments together")
            
            # Stitch segments together
            final_video_path = os.path.join(job_output_dir, "final_video.mp4")
            stitched_path = stitch_video_segments(video_paths, final_video_path)
            
            if not stitched_path or not os.path.exists(stitched_path):
                raise Exception("Failed to stitch video segments")
            
            logger.info(f"Final video created: {stitched_path}")
            
            # Phase 7: Upload to GCS (95%)
            if check_cancellation(cancellation_token, job_id):
                raise InterruptedError("Job cancelled before GCS upload")
            
            job_queue.update_job_status(job_id, JobStatus.PROGRESS, progress=90)
            job_queue.add_job_log(job_id, "Uploading final video to Google Cloud Storage")
            
            # Upload to GCS
            gcs_uri = upload_video_to_gcs(
                video_path=stitched_path,
                job_id=job_id,
                config=final_config,
                cancellation_token=cancellation_token
            )
            
            logger.info(f"Video uploaded to GCS: {gcs_uri}")
            
            job_queue.update_job_status(job_id, JobStatus.PROGRESS, progress=95)
            job_queue.add_job_log(job_id, f"Upload completed: {gcs_uri}")
            
            return gcs_uri
            
        except InterruptedError:
            # Cancellation - clean up and re-raise
            logger.info(f"Pipeline execution cancelled for job {job_id}")
            cleanup_job_files(temp_dir)
            raise
            
        except Exception as e:
            # Other errors - clean up and re-raise
            logger.error(f"Pipeline execution failed for job {job_id}: {e}")
            cleanup_job_files(temp_dir)
            raise


def generate_keyframes_with_progress(
    keyframe_prompts: List[str],
    config: Dict[str, Any],
    output_dir: str,
    cancellation_token: CancellationToken,
    job_queue,
    job_id: str,
    progress_start: int,
    progress_end: int
) -> List[str]:
    """
    Generate keyframes with progress reporting and cancellation support.
    
    Args:
        keyframe_prompts: List of keyframe prompts
        config: Configuration dictionary
        output_dir: Output directory for keyframes
        cancellation_token: Cancellation token
        job_queue: Job queue for progress updates
        job_id: Job identifier
        progress_start: Starting progress percentage
        progress_end: Ending progress percentage
        
    Returns:
        List of keyframe file paths
    """
    from pipeline import generate_keyframes
    
    # Check for cancellation before starting
    if check_cancellation(cancellation_token, job_id):
        raise InterruptedError("Job cancelled before keyframe generation")
    
    try:
        # Generate keyframes using existing pipeline function
        keyframe_paths = generate_keyframes(
            keyframe_prompts=keyframe_prompts,
            config=config,
            output_dir=output_dir,
            model_name=config.get('image_generation_model'),
            imageRouter_api_key=config.get('image_router_api_key'),
            stability_api_key=config.get('stability_api_key'),
            openai_api_key=config.get('openai_api_key'),
            gemini_api_key=config.get('gemini_api_key'),
            initial_image_path=config.get('initial_image'),
            image_size=config.get('image_size'),
            reference_images_dir=config.get('reference_images_dir'),
            max_retries=3
        )
        
        # Update progress
        job_queue.update_job_status(job_id, JobStatus.PROGRESS, progress=progress_end)
        
        return keyframe_paths
        
    except Exception as e:
        logger.error(f"Keyframe generation failed: {e}")
        raise


def generate_video_segments_with_progress(
    video_prompts: List[Dict],
    config: Dict[str, Any],
    output_dir: str,
    cancellation_token: CancellationToken,
    job_queue,
    job_id: str,
    processes: List,
    progress_start: int,
    progress_end: int
) -> List[str]:
    """
    Generate video segments with progress reporting and cancellation support.
    
    Args:
        video_prompts: List of video prompt dictionaries
        config: Configuration dictionary
        output_dir: Output directory for videos
        cancellation_token: Cancellation token
        job_queue: Job queue for progress updates
        job_id: Job identifier
        processes: List to track subprocesses
        progress_start: Starting progress percentage
        progress_end: Ending progress percentage
        
    Returns:
        List of video file paths
    """
    from pipeline import generate_video_segments
    
    # Check for cancellation before starting
    if check_cancellation(cancellation_token, job_id):
        raise InterruptedError("Job cancelled before video generation")
    
    try:
        # Determine if using single keyframe mode (for remote APIs like Veo3)
        single_keyframe_mode = config.get('single_keyframe_mode', False)
        
        if single_keyframe_mode:
            # Use remote API generation
            from pipeline import generate_video_segments_single_keyframe
            video_paths = generate_video_segments_single_keyframe(
                config=config,
                video_prompts=video_prompts,
                output_dir=output_dir
            )
        else:
            # Use local Wan2.1 generation
            video_paths = generate_video_segments(
                wan2_dir=config.get('wan2_dir', './Wan2.1'),
                config=config,
                video_prompts=video_prompts,
                output_dir=output_dir,
                flf2v_model_dir=config.get('flf2v_model_dir'),
                frame_num=config.get('frame_num', 81),
                single_keyframe_mode=single_keyframe_mode
            )
        
        # Update progress
        job_queue.update_job_status(job_id, JobStatus.PROGRESS, progress=progress_end)
        
        return video_paths
        
    except Exception as e:
        logger.error(f"Video segment generation failed: {e}")
        raise


def upload_video_to_gcs(
    video_path: str,
    job_id: str,
    config: Dict[str, Any],
    cancellation_token: CancellationToken
) -> str:
    """
    Upload final video to Google Cloud Storage.
    
    Args:
        video_path: Path to the final video file
        job_id: Job identifier
        config: Configuration dictionary
        cancellation_token: Cancellation token
        
    Returns:
        GCS URI of the uploaded video
    """
    # Check for cancellation before upload
    if cancellation_token.is_cancelled():
        raise InterruptedError("Job cancelled before GCS upload")
    
    try:
        # Create GCS configuration from config
        from api.config import GCSConfig
        
        gcs_config = GCSConfig(
            bucket=config.get('gcs_bucket', 'ttv-api-artifacts'),
            prefix=config.get('gcs_prefix', 'ttv-api'),
            credentials_path=config.get('credentials_path', 'credentials.json'),
            signed_url_expiration=config.get('signed_url_expiration', 3600)
        )
        
        # Use existing GCS uploader
        from workers.gcs_uploader import upload_job_artifact
        
        gcs_uri = upload_job_artifact(
            local_video_path=video_path,
            job_id=job_id,
            gcs_config=gcs_config,
            cleanup_local=False  # Keep local file for now
        )
        
        if not gcs_uri:
            raise Exception("GCS upload returned None - upload failed")
        
        return gcs_uri
        
    except Exception as e:
        logger.error(f"GCS upload failed: {e}")
        raise


def cleanup_job_files(temp_dir: str):
    """
    Clean up temporary files for a job.
    
    Args:
        temp_dir: Temporary directory to clean up
    """
    try:
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")


def get_job_progress(job_id: str) -> Dict[str, Any]:
    """
    Get current job progress information.
    
    Args:
        job_id: The job identifier
        
    Returns:
        Dictionary with job progress information
    """
    try:
        job_queue = get_job_queue()
        job_data = job_queue.get_job(job_id)
        
        if not job_data:
            return {"error": "Job not found"}
        
        return {
            "id": job_data.id,
            "status": job_data.status.value,
            "progress": job_data.progress,
            "created_at": job_data.created_at.isoformat(),
            "started_at": job_data.started_at.isoformat() if job_data.started_at else None,
            "finished_at": job_data.finished_at.isoformat() if job_data.finished_at else None,
            "gcs_uri": job_data.gcs_uri,
            "error": job_data.error,
            "log_count": len(job_data.logs)
        }
        
    except Exception as e:
        logger.error(f"Failed to get progress for job {job_id}: {e}")
        return {"error": str(e)}


def cancel_job_processing(job_id: str) -> bool:
    """
    Handle job cancellation from worker side.
    
    This function can be called to cooperatively cancel a running job.
    
    Args:
        job_id: The job identifier
        
    Returns:
        True if cancellation was handled, False otherwise
    """
    try:
        logger.info(f"Cancellation requested for job {job_id}")
        
        job_queue = get_job_queue()
        job_queue.add_job_log(job_id, "Cancellation signal received")
        
        # In a real implementation, this would:
        # 1. Set a cancellation flag
        # 2. Stop any running processes
        # 3. Clean up temporary files
        # 4. Update job status
        
        job_queue.update_job_status(job_id, JobStatus.CANCELED)
        job_queue.add_job_log(job_id, "Job cancelled by worker")
        
        logger.info(f"Job {job_id} cancelled successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        return False