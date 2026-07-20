"""
Job management routes for the API server.

This module contains the job creation, status, and management endpoints.
"""

import mimetypes
import os
from datetime import datetime, timedelta, timezone
from typing import List

from fastapi import APIRouter, HTTPException, Request

from api.config_merger import ConfigMerger
from api.logging_config import get_logger
from api.models import (
    JobCreateRequest,
    JobCreateResponse,
    JobStatus,
    JobStatusResponse,
    PlanCreateRequest,
    PromptEnhancementResult,
)
from api.queue import JobQueue

logger = get_logger(__name__)
router = APIRouter(tags=["jobs"])
plans_router = APIRouter(tags=["plans"])


@plans_router.post(
    "",
    response_model=PromptEnhancementResult,
    response_model_exclude_unset=True,
)
def create_plan(
    request_obj: Request,
    request: PlanCreateRequest,
) -> PromptEnhancementResult:
    """Create an editable prompt plan without starting media generation."""
    api_config = getattr(request_obj.app.state, "config", None)
    if not api_config or not isinstance(api_config.pipeline_config, dict):
        raise HTTPException(status_code=503, detail="Pipeline configuration not available")

    effective_config = ConfigMerger().merge_for_job(
        api_config.pipeline_config,
        request.prompt,
        request.duration_seconds,
    )
    if not effective_config.get("openai_api_key") and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="Prompt enhancement credentials are not configured",
        )

    from pipeline import enhance_prompt_data

    try:
        return PromptEnhancementResult.model_validate(
            enhance_prompt_data(request.prompt, effective_config)
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("", response_model=JobCreateResponse, status_code=202)
async def create_job(request_obj: Request, request: JobCreateRequest) -> JobCreateResponse:
    """
    Create a new video generation job.

    Accepts a prompt and optional requested duration, then returns a task ID.
    The job is queued for processing and can be monitored via the status endpoint.
    """
    # Get job queue from app state
    job_queue: JobQueue = getattr(request_obj.app.state, "job_queue", None)
    if not job_queue:
        raise HTTPException(status_code=503, detail="Job queue not available")

    api_config = getattr(request_obj.app.state, "config", None)
    if not api_config or not isinstance(api_config.pipeline_config, dict):
        raise HTTPException(status_code=503, detail="Pipeline configuration not available")

    job_prompt = request.prompt or "Resumed from reviewed prompt plan"
    effective_config = ConfigMerger().merge_for_job(
        api_config.pipeline_config,
        job_prompt,
        request.duration_seconds,
    )
    if request.enhanced_prompt is not None:
        reviewed_plan = request.enhanced_prompt.model_dump(exclude_unset=True)
        if request.duration_seconds is None:
            requested_total = reviewed_plan["segmentation_logic"][
                "total_duration_seconds"
            ]
            generated_total = sum(
                segment["duration_seconds"]
                for segment in reviewed_plan["video_prompts"]
            )
            if requested_total != generated_total:
                effective_config["duration_seconds"] = requested_total
            else:
                effective_config.pop("duration_seconds", None)
        effective_config["enhanced_prompt"] = reviewed_plan
    if request.keyframes_only:
        effective_config["keyframes_only"] = True
        if effective_config.get("generation_mode", "keyframe").lower() != "keyframe":
            raise HTTPException(
                status_code=422,
                detail="keyframes_only requires generation_mode=keyframe",
            )
    effective_config.update(
        {
            "gcs_bucket": api_config.gcs.bucket,
            "gcs_prefix": api_config.gcs.prefix,
            "credentials_path": api_config.gcs.credentials_path,
            "signed_url_expiration": api_config.gcs.signed_url_expiration,
        }
    )

    from pipeline import (
        get_duration_tradeoff,
        get_requested_job_timeout,
        validate_prompt_enhancement,
    )

    try:
        if request.enhanced_prompt is not None:
            validate_prompt_enhancement(
                effective_config["enhanced_prompt"], effective_config
            )
        tradeoff = get_duration_tradeoff(effective_config)
        job_timeout = get_requested_job_timeout(effective_config)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    job = job_queue.enqueue_job(
        request=request,
        effective_config=effective_config,
        job_timeout=job_timeout,
    )

    logger.info(f"Created job {job.id} with prompt: {job_prompt[:50]}...")

    return JobCreateResponse(
        id=job.id,
        status=job.status,
        created_at=job.created_at,
        warnings=[tradeoff] if tradeoff else [],
    )


@router.get("", response_model=List[JobStatusResponse])
async def list_jobs(
    request_obj: Request, limit: int = 100, offset: int = 0
) -> List[JobStatusResponse]:
    """
    List recent jobs with pagination.

    Returns a list of job status objects ordered by creation time.
    """
    # Get job queue from app state
    job_queue: JobQueue = getattr(request_obj.app.state, "job_queue", None)
    if not job_queue:
        raise HTTPException(status_code=503, detail="Job queue not available")

    # Get jobs list from queue
    jobs = job_queue.list_jobs(limit=limit, offset=offset)

    return [
        JobStatusResponse(
            id=job.id,
            status=job.status,
            progress=job.progress,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            gcs_uri=job.gcs_uri,
            error=job.error
        )
        for job in jobs
    ]


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    request_obj: Request,
    job_id: str
) -> JobStatusResponse:
    """
    Get the status of a video generation job.

    Returns current status, progress, timestamps, and GCS URI when available.
    """
    # Get job queue from app state
    job_queue: JobQueue = getattr(request_obj.app.state, 'job_queue', None)
    if not job_queue:
        raise HTTPException(status_code=503, detail="Job queue not available")
    
    # Get job status
    job = job_queue.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(
        id=job.id,
        status=job.status,
        progress=job.progress,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        gcs_uri=job.gcs_uri,
        error=job.error
    )


@router.get("/{job_id}/video-url")
@router.get("/{job_id}/artifact-url")
async def get_job_artifact_url(
    request_obj: Request,
    job_id: str,
    expiration_seconds: int = 3600
) -> dict:
    """
    Get a signed URL and media metadata for a completed job artifact.
    
    Returns a time-limited signed URL that can be used to stream or embed the video
    directly from Google Cloud Storage without authentication.
    
    Args:
        job_id: The job identifier
        expiration_seconds: URL expiration time in seconds (default: 1 hour)
    
    Returns:
        A dictionary containing the signed URL and expiration time
    """
    from api.gcs_client import create_gcs_client
    from api.config import get_config_from_env
    
    # Get job queue from app state
    job_queue = getattr(request_obj.app.state, 'job_queue', None)
    if not job_queue:
        raise HTTPException(status_code=503, detail="Job queue not available")
    
    # Get job status
    job = job_queue.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.FINISHED:
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed. Current status: {job.status}"
        )
    
    if not job.gcs_uri:
        raise HTTPException(
            status_code=404, 
            detail="No artifact found for this job"
        )
    
    # Create GCS client and generate signed URL
    try:
        # Get GCS config from app state or load it
        config = getattr(request_obj.app.state, 'config', None)
        if not config:
            config = get_config_from_env()
        
        gcs_client = create_gcs_client(config.gcs)
        
        signed_url = gcs_client.generate_signed_url(
            gcs_uri=job.gcs_uri,
            expiration_seconds=expiration_seconds
        )
        
        expiration_time = datetime.now(timezone.utc) + timedelta(seconds=expiration_seconds)
        
        logger.info(f"Generated signed URL for job {job_id}, expires at {expiration_time}")
        
        artifact_name = os.path.basename(job.gcs_uri)
        mime_type = mimetypes.guess_type(artifact_name)[0] or "application/octet-stream"
        response = {
            "artifact_url": signed_url,
            "artifact_name": artifact_name,
            "expires_at": expiration_time.isoformat(),
            "expiration_seconds": expiration_seconds,
            "mime_type": mime_type,
        }
        if mime_type.startswith("video/"):
            response["video_url"] = signed_url
        return response
        
    except Exception as e:
        logger.error(f"Failed to generate signed URL for job {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate artifact URL"
        )
