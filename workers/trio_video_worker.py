"""
Trio-based video generation worker with structured concurrency.

This module provides a Trio-based implementation of the video worker
that uses structured task groups and cancellation scopes for proper
resource management and cooperative cancellation.
"""

import logging
import os
import tempfile
import yaml
from typing import Dict, Any, Optional

import trio

from api.queue import get_job_queue
from api.models import JobStatus
from api.config_merger import ConfigMerger
from .trio_executor import (
    TrioCancellationToken, 
    ProcessManager, 
    check_cancellation_async,
    cleanup_temp_files_async,
    AsyncioTrioBridge
)

logger = logging.getLogger(__name__)


def _record_job_metrics_trio(processing_time: float, success: bool = True):
    """
    Record job processing metrics for observability (Trio version).
    
    Args:
        processing_time: Time taken to process the job in seconds
        success: Whether the job completed successfully
    """
    try:
        from api.routes.health import record_job_processed
        record_job_processed(processing_time)
        
        logger.info(
            f"Trio job processing metrics recorded: {processing_time:.2f}s, success={success}",
            extra={
                'extra': {
                    'event': 'job_metrics_trio',
                    'processing_time': processing_time,
                    'success': success,
                    'job_type': 'video_generation',
                    'execution_mode': 'trio'
                }
            }
        )
    except ImportError:
        logger.debug("Health metrics module not available for Trio job metrics recording")
    except Exception as e:
        logger.warning(f"Failed to record Trio job metrics: {e}")


async def process_video_job_trio(job_id: str) -> str:
    """
    Trio-based video generation job processor with structured concurrency.
    
    This function uses Trio's structured concurrency features to manage
    the video generation pipeline with proper cancellation and resource cleanup.
    
    Args:
        job_id: The unique job identifier
        
    Returns:
        The GCS URI of the generated video
        
    Raises:
        trio.Cancelled: If job is cancelled
        Exception: If job processing fails
    """
    logger.info(f"Starting Trio-based video generation job {job_id}")
    
    # Use structured concurrency with nursery
    async with trio.open_nursery() as nursery:
        # Create cancellation scope for this job
        with trio.CancelScope() as cancel_scope:
            # Create cancellation token
            cancellation_token = TrioCancellationToken(job_id, cancel_scope)
            
            try:
                # Get job queue instance
                job_queue = get_job_queue()
                
                # Get job data
                job_data = job_queue.get_job(job_id)
                if not job_data:
                    raise ValueError(f"Job {job_id} not found")
                
                # Check for early cancellation
                if await check_cancellation_async(cancellation_token, job_id):
                    raise trio.Cancelled()
                
                # Update status to started
                job_queue.update_job_status(job_id, JobStatus.STARTED)
                job_queue.add_job_log(job_id, "Video generation started with Trio")
                
                # Extract configuration and prompt
                prompt = job_data.prompt
                effective_config = job_data.config
                
                logger.info(f"Processing job {job_id} with prompt: {prompt[:100]}...")
                logger.info(f"Using effective configuration with {len(effective_config)} parameters")
                
                # Execute the pipeline with structured concurrency
                gcs_uri = await execute_pipeline_with_trio(
                    job_id=job_id,
                    prompt=prompt,
                    config=effective_config,
                    cancellation_token=cancellation_token,
                    job_queue=job_queue,
                    nursery=nursery
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
                
            except trio.Cancelled:
                logger.info(f"Job {job_id} was cancelled")
                # Job status already updated in check_cancellation_async
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


async def execute_pipeline_with_trio(
    job_id: str,
    prompt: str,
    config: Dict[str, Any],
    cancellation_token: TrioCancellationToken,
    job_queue,
    nursery: trio.Nursery
) -> str:
    """
    Execute the video generation pipeline using Trio structured concurrency.
    
    This function integrates with the existing pipeline.py to generate videos
    using structured task groups and proper cancellation handling.
    
    Args:
        job_id: The unique job identifier
        prompt: The video generation prompt
        config: Effective configuration dictionary
        cancellation_token: Trio cancellation token
        job_queue: Job queue instance for status updates
        nursery: Trio nursery for task management
        
    Returns:
        The GCS URI of the generated video
        
    Raises:
        trio.Cancelled: If job is cancelled
        Exception: If pipeline execution fails
    """
    logger.info(f"Executing pipeline with Trio for job {job_id}")
    
    # Create temporary working directory for this job
    temp_dir = tempfile.mkdtemp(prefix=f"job_{job_id}_")
    
    # Add cleanup callback for temp directory
    cancellation_token.add_cleanup_callback(
        lambda: trio.from_thread.run_sync(cleanup_temp_files_async, temp_dir)
    )
    
    try:
        logger.info(f"Using temporary directory: {temp_dir}")
        
        # Create job-specific output directory
        job_output_dir = os.path.join(temp_dir, "output")
        os.makedirs(job_output_dir, exist_ok=True)
        
        # Phase 1: Setup and validation (5%)
        if await check_cancellation_async(cancellation_token, job_id):
            raise trio.Cancelled()
        
        job_queue.update_job_status(job_id, JobStatus.PROGRESS, progress=5)
        job_queue.add_job_log(job_id, "Setting up pipeline configuration")
        
        # Write job-specific configuration to temporary file for logging
        job_config_path = os.path.join(temp_dir, "job_config.yaml")
        await trio.to_thread.run_sync(write_config_file, job_config_path, config)
        
        logger.info(f"Job configuration written to {job_config_path}")
        
        # Phase 2: Import and initialize pipeline (10%)
        if await check_cancellation_async(cancellation_token, job_id):
            raise trio.Cancelled()
        
        job_queue.update_job_status(job_id, JobStatus.PROGRESS, progress=10)
        job_queue.add_job_log(job_id, "Initializing video generation pipeline")
        
        # Import pipeline functions (run in thread to avoid blocking)
        pipeline_modules = await trio.to_thread.run_sync(import_pipeline_modules)
        
        # Load base configuration from mounted pipeline_config.yaml
        base_config_path = "/app/pipeline_config.yaml"
        if not os.path.exists(base_config_path):
            logger.error(f"Base pipeline config not found at {base_config_path}")
            raise FileNotFoundError(f"Pipeline configuration not found at {base_config_path}")
        
        base_config = await trio.to_thread.run_sync(
            pipeline_modules['load_config'], base_config_path
        )
        
        # Use ConfigMerger to ensure proper precedence
        config_merger = ConfigMerger()
        final_config = config_merger.merge_for_job(base_config, prompt)
        
        logger.info(f"Final configuration merged with prompt override")
        
        # Phase 3: Prompt enhancement and segmentation (20%)
        if await check_cancellation_async(cancellation_token, job_id):
            raise trio.Cancelled()
        
        job_queue.update_job_status(job_id, JobStatus.PROGRESS, progress=20)
        job_queue.add_job_log(job_id, "Enhancing and segmenting prompt")
        
        # Run prompt enhancement in structured task
        enhancement_result = await enhance_prompt_trio(
            prompt, final_config, cancellation_token, nursery
        )
        
        keyframe_prompts = [item['prompt'] for item in enhancement_result['keyframe_prompts']]
        video_prompts = enhancement_result['video_prompts']
        
        logger.info(f"Generated {len(keyframe_prompts)} keyframe prompts and {len(video_prompts)} video prompts")
        
        # Phase 4: Keyframe generation (50%)
        if await check_cancellation_async(cancellation_token, job_id):
            raise trio.Cancelled()
        
        job_queue.update_job_status(job_id, JobStatus.PROGRESS, progress=30)
        job_queue.add_job_log(job_id, f"Generating {len(keyframe_prompts)} keyframes")
        
        # Generate keyframes with structured concurrency
        keyframe_paths = await generate_keyframes_trio(
            keyframe_prompts=keyframe_prompts,
            config=final_config,
            output_dir=job_output_dir,
            cancellation_token=cancellation_token,
            job_queue=job_queue,
            job_id=job_id,
            progress_start=30,
            progress_end=50,
            nursery=nursery
        )
        
        logger.info(f"Generated {len(keyframe_paths)} keyframes")
        
        # Phase 5: Video segment generation (80%)
        if await check_cancellation_async(cancellation_token, job_id):
            raise trio.Cancelled()
        
        job_queue.update_job_status(job_id, JobStatus.PROGRESS, progress=50)
        job_queue.add_job_log(job_id, f"Generating {len(video_prompts)} video segments")
        
        # Generate video segments with structured concurrency
        video_paths = await generate_video_segments_trio(
            video_prompts=video_prompts,
            config=final_config,
            output_dir=job_output_dir,
            cancellation_token=cancellation_token,
            job_queue=job_queue,
            job_id=job_id,
            progress_start=50,
            progress_end=80,
            nursery=nursery
        )
        
        logger.info(f"Generated {len(video_paths)} video segments")
        
        # Phase 6: Video stitching (90%)
        if await check_cancellation_async(cancellation_token, job_id):
            raise trio.Cancelled()
        
        job_queue.update_job_status(job_id, JobStatus.PROGRESS, progress=80)
        job_queue.add_job_log(job_id, "Stitching video segments together")
        
        # Stitch segments together
        final_video_path = os.path.join(job_output_dir, "final_video.mp4")
        stitched_path = await stitch_video_segments_trio(
            video_paths, final_video_path, cancellation_token, nursery
        )
        
        if not stitched_path or not os.path.exists(stitched_path):
            raise Exception("Failed to stitch video segments")
        
        logger.info(f"Final video created: {stitched_path}")
        
        # Phase 7: Upload to GCS (95%)
        if await check_cancellation_async(cancellation_token, job_id):
            raise trio.Cancelled()
        
        job_queue.update_job_status(job_id, JobStatus.PROGRESS, progress=90)
        job_queue.add_job_log(job_id, "Uploading final video to Google Cloud Storage")
        
        # Upload to GCS
        gcs_uri = await upload_video_to_gcs_trio(
            video_path=stitched_path,
            job_id=job_id,
            config=final_config,
            cancellation_token=cancellation_token,
            nursery=nursery
        )
        
        logger.info(f"Video uploaded to GCS: {gcs_uri}")
        
        job_queue.update_job_status(job_id, JobStatus.PROGRESS, progress=95)
        job_queue.add_job_log(job_id, f"Upload completed: {gcs_uri}")
        
        return gcs_uri
        
    except trio.Cancelled:
        # Cancellation - clean up and re-raise
        logger.info(f"Pipeline execution cancelled for job {job_id}")
        await cleanup_temp_files_async(temp_dir)
        raise
        
    except Exception as e:
        # Other errors - clean up and re-raise
        logger.error(f"Pipeline execution failed for job {job_id}: {e}")
        await cleanup_temp_files_async(temp_dir)
        raise


def write_config_file(config_path: str, config: Dict[str, Any]):
    """Write configuration to YAML file (sync function for thread execution)"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def import_pipeline_modules():
    """Import pipeline modules (sync function for thread execution)"""
    from pipeline import (
        load_config, PromptEnhancer, generate_keyframes, 
        generate_video_segments, stitch_video_segments,
        PROMPT_ENHANCEMENT_INSTRUCTIONS
    )
    
    return {
        'load_config': load_config,
        'PromptEnhancer': PromptEnhancer,
        'generate_keyframes': generate_keyframes,
        'generate_video_segments': generate_video_segments,
        'stitch_video_segments': stitch_video_segments,
        'PROMPT_ENHANCEMENT_INSTRUCTIONS': PROMPT_ENHANCEMENT_INSTRUCTIONS
    }


async def enhance_prompt_trio(
    prompt: str,
    config: Dict[str, Any],
    cancellation_token: TrioCancellationToken,
    nursery: trio.Nursery
) -> Dict[str, Any]:
    """
    Enhance prompt using Trio structured concurrency
    
    Args:
        prompt: Original prompt
        config: Configuration dictionary
        cancellation_token: Cancellation token
        nursery: Trio nursery for task management
        
    Returns:
        Enhancement result with keyframe and video prompts
    """
    def enhance_sync():
        from pipeline import PromptEnhancer, PROMPT_ENHANCEMENT_INSTRUCTIONS
        
        enhancer = PromptEnhancer(
            api_key=config.get('openai_api_key'),
            base_url=config.get('openai_base_url', 'https://api.openai.com/v1'),
            model=config.get('prompt_enhancement_model', 'gpt-4o-mini')
        )
        
        return enhancer.enhance(PROMPT_ENHANCEMENT_INSTRUCTIONS, prompt)
    
    # Run enhancement in thread with cancellation check
    if await check_cancellation_async(cancellation_token, job_id=""):
        raise trio.Cancelled()
    
    return await trio.to_thread.run_sync(enhance_sync)


async def generate_keyframes_trio(
    keyframe_prompts: list,
    config: Dict[str, Any],
    output_dir: str,
    cancellation_token: TrioCancellationToken,
    job_queue,
    job_id: str,
    progress_start: int,
    progress_end: int,
    nursery: trio.Nursery
) -> list:
    """
    Generate keyframes using Trio structured concurrency
    
    Args:
        keyframe_prompts: List of keyframe prompts
        config: Configuration dictionary
        output_dir: Output directory for keyframes
        cancellation_token: Cancellation token
        job_queue: Job queue for progress updates
        job_id: Job identifier
        progress_start: Starting progress percentage
        progress_end: Ending progress percentage
        nursery: Trio nursery for task management
        
    Returns:
        List of keyframe file paths
    """
    # Check for cancellation before starting
    if await check_cancellation_async(cancellation_token, job_id):
        raise trio.Cancelled()
    
    def generate_sync():
        from pipeline import generate_keyframes
        
        return generate_keyframes(
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
    
    try:
        # Run keyframe generation in thread
        keyframe_paths = await trio.to_thread.run_sync(generate_sync)
        
        # Update progress
        job_queue.update_job_status(job_id, JobStatus.PROGRESS, progress=progress_end)
        
        return keyframe_paths
        
    except Exception as e:
        logger.error(f"Keyframe generation failed: {e}")
        raise


async def generate_video_segments_trio(
    video_prompts: list,
    config: Dict[str, Any],
    output_dir: str,
    cancellation_token: TrioCancellationToken,
    job_queue,
    job_id: str,
    progress_start: int,
    progress_end: int,
    nursery: trio.Nursery
) -> list:
    """
    Generate video segments using Trio structured concurrency
    
    Args:
        video_prompts: List of video prompt dictionaries
        config: Configuration dictionary
        output_dir: Output directory for videos
        cancellation_token: Cancellation token
        job_queue: Job queue for progress updates
        job_id: Job identifier
        progress_start: Starting progress percentage
        progress_end: Ending progress percentage
        nursery: Trio nursery for task management
        
    Returns:
        List of video file paths
    """
    # Check for cancellation before starting
    if await check_cancellation_async(cancellation_token, job_id):
        raise trio.Cancelled()
    
    def generate_sync():
        from pipeline import generate_video_segments, generate_video_segments_single_keyframe
        
        # Determine if using single keyframe mode
        single_keyframe_mode = config.get('single_keyframe_mode', False)
        
        if single_keyframe_mode:
            # Use remote API generation
            return generate_video_segments_single_keyframe(
                config=config,
                video_prompts=video_prompts,
                output_dir=output_dir
            )
        else:
            # Use local Wan2.1 generation
            return generate_video_segments(
                wan2_dir=config.get('wan2_dir', './Wan2.1'),
                config=config,
                video_prompts=video_prompts,
                output_dir=output_dir,
                flf2v_model_dir=config.get('flf2v_model_dir'),
                frame_num=config.get('frame_num', 81),
                single_keyframe_mode=single_keyframe_mode
            )
    
    try:
        # Run video generation in thread
        video_paths = await trio.to_thread.run_sync(generate_sync)
        
        # Update progress
        job_queue.update_job_status(job_id, JobStatus.PROGRESS, progress=progress_end)
        
        return video_paths
        
    except Exception as e:
        logger.error(f"Video segment generation failed: {e}")
        raise


async def stitch_video_segments_trio(
    video_paths: list,
    final_video_path: str,
    cancellation_token: TrioCancellationToken,
    nursery: trio.Nursery
) -> str:
    """
    Stitch video segments using Trio structured concurrency
    
    Args:
        video_paths: List of video file paths
        final_video_path: Output path for final video
        cancellation_token: Cancellation token
        nursery: Trio nursery for task management
        
    Returns:
        Path to stitched video file
    """
    # Check for cancellation before starting
    if cancellation_token.is_cancelled():
        raise trio.Cancelled()
    
    def stitch_sync():
        from pipeline import stitch_video_segments
        return stitch_video_segments(video_paths, final_video_path)
    
    # Run stitching in thread
    return await trio.to_thread.run_sync(stitch_sync)


async def upload_video_to_gcs_trio(
    video_path: str,
    job_id: str,
    config: Dict[str, Any],
    cancellation_token: TrioCancellationToken,
    nursery: trio.Nursery
) -> str:
    """
    Upload final video to Google Cloud Storage using Trio
    
    Args:
        video_path: Path to the final video file
        job_id: Job identifier
        config: Configuration dictionary
        cancellation_token: Cancellation token
        nursery: Trio nursery for task management
        
    Returns:
        GCS URI of the uploaded video
    """
    # Check for cancellation before upload
    if cancellation_token.is_cancelled():
        raise trio.Cancelled()
    
    def upload_sync():
        from api.config import GCSConfig
        from workers.gcs_uploader import upload_job_artifact
        
        gcs_config = GCSConfig(
            bucket=config.get('gcs_bucket', 'ttv-api-artifacts'),
            prefix=config.get('gcs_prefix', 'ttv-api'),
            credentials_path=config.get('credentials_path', 'credentials.json'),
            signed_url_expiration=config.get('signed_url_expiration', 3600)
        )
        
        return upload_job_artifact(
            local_video_path=video_path,
            job_id=job_id,
            gcs_config=gcs_config,
            cleanup_local=False
        )
    
    try:
        gcs_uri = await trio.to_thread.run_sync(upload_sync)
        
        if not gcs_uri:
            raise Exception("GCS upload returned None - upload failed")
        
        return gcs_uri
        
    except Exception as e:
        logger.error(f"GCS upload failed: {e}")
        raise


# Wrapper function for RQ compatibility
def process_video_job_trio_wrapper(job_id: str) -> str:
    """
    Wrapper function to run Trio-based job processing from RQ
    
    This function bridges the sync RQ interface with async Trio execution.
    
    Args:
        job_id: The unique job identifier
        
    Returns:
        The GCS URI of the generated video
    """
    import time
    start_time = time.time()
    
    try:
        result = trio.run(process_video_job_trio, job_id)
        
        # Record successful job processing metrics
        processing_time = time.time() - start_time
        _record_job_metrics_trio(processing_time, success=True)
        
        return result
        
    except Exception as e:
        # Record failed job processing metrics
        processing_time = time.time() - start_time
        _record_job_metrics_trio(processing_time, success=False)
        raise