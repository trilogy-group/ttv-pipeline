"""
GCS uploader utility for video generation workers.

This module provides a simple interface for workers to upload video artifacts
to Google Cloud Storage and update job status with GCS URIs.
"""

import logging
from pathlib import Path
from typing import Optional

from api.gcs_client import GCSClient, GCSClientError, GCSUploadError
from api.config import GCSConfig

logger = logging.getLogger(__name__)


class WorkerGCSUploader:
    """
    GCS uploader utility for video generation workers.
    
    Provides a simplified interface for workers to upload artifacts and handle
    common error scenarios.
    """
    
    def __init__(self, gcs_client: GCSClient):
        """
        Initialize the uploader with a GCS client.
        
        Args:
            gcs_client: Initialized GCS client
        """
        self.gcs_client = gcs_client
    
    def upload_video_artifact(
        self, 
        local_video_path: str, 
        job_id: str,
        cleanup_local: bool = False
    ) -> Optional[str]:
        """
        Upload a video artifact to GCS and optionally clean up the local file.
        
        Args:
            local_video_path: Path to the local video file
            job_id: Job identifier for the artifact
            cleanup_local: Whether to delete the local file after successful upload
            
        Returns:
            GCS URI of the uploaded artifact, or None if upload failed
        """
        local_path = Path(local_video_path)
        
        if not local_path.exists():
            logger.error(f"Local video file not found: {local_video_path}")
            return None
        
        try:
            logger.info(f"Uploading video artifact for job {job_id}: {local_video_path}")
            
            # Upload the video file
            gcs_uri = self.gcs_client.upload_artifact(
                local_file_path=str(local_path),
                job_id=job_id,
                filename="final_video.mp4"
            )
            
            logger.info(f"Successfully uploaded video artifact: {gcs_uri}")
            
            # Clean up local file if requested
            if cleanup_local:
                try:
                    local_path.unlink()
                    logger.info(f"Cleaned up local file: {local_video_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up local file {local_video_path}: {e}")
            
            return gcs_uri
            
        except GCSUploadError as e:
            logger.error(f"Failed to upload video artifact for job {job_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading video artifact for job {job_id}: {e}")
            return None
    
    def upload_additional_artifact(
        self, 
        local_file_path: str, 
        job_id: str, 
        artifact_name: str,
        cleanup_local: bool = False
    ) -> Optional[str]:
        """
        Upload an additional artifact (e.g., logs, metadata) to GCS.
        
        Args:
            local_file_path: Path to the local file
            job_id: Job identifier for the artifact
            artifact_name: Name for the uploaded artifact
            cleanup_local: Whether to delete the local file after successful upload
            
        Returns:
            GCS URI of the uploaded artifact, or None if upload failed
        """
        local_path = Path(local_file_path)
        
        if not local_path.exists():
            logger.error(f"Local file not found: {local_file_path}")
            return None
        
        try:
            logger.info(f"Uploading additional artifact for job {job_id}: {artifact_name}")
            
            # Upload the file
            gcs_uri = self.gcs_client.upload_artifact(
                local_file_path=str(local_path),
                job_id=job_id,
                filename=artifact_name
            )
            
            logger.info(f"Successfully uploaded additional artifact: {gcs_uri}")
            
            # Clean up local file if requested
            if cleanup_local:
                try:
                    local_path.unlink()
                    logger.info(f"Cleaned up local file: {local_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up local file {local_file_path}: {e}")
            
            return gcs_uri
            
        except GCSUploadError as e:
            logger.error(f"Failed to upload additional artifact for job {job_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading additional artifact for job {job_id}: {e}")
            return None
    
    def verify_upload(self, gcs_uri: str) -> bool:
        """
        Verify that an uploaded artifact exists and is accessible.
        
        Args:
            gcs_uri: GCS URI of the artifact to verify
            
        Returns:
            True if artifact exists and is accessible, False otherwise
        """
        try:
            # Try to generate a signed URL - this will fail if the artifact doesn't exist
            signed_url = self.gcs_client.generate_signed_url(gcs_uri, expiration_seconds=60)
            logger.debug(f"Verified artifact exists: {gcs_uri}")
            return True
            
        except GCSClientError as e:
            logger.warning(f"Artifact verification failed for {gcs_uri}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error verifying artifact {gcs_uri}: {e}")
            return False
    
    def cleanup_job_artifacts(self, job_id: str) -> int:
        """
        Clean up all artifacts for a specific job.
        
        This is useful for cleaning up failed jobs or implementing retention policies.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Number of artifacts deleted
        """
        try:
            # List all artifacts for the job
            artifacts = self.gcs_client.list_artifacts(job_id=job_id)
            
            deleted_count = 0
            for artifact in artifacts:
                try:
                    gcs_uri = artifact['gcs_uri']
                    if self.gcs_client.delete_artifact(gcs_uri):
                        deleted_count += 1
                        logger.info(f"Deleted artifact: {gcs_uri}")
                except Exception as e:
                    logger.warning(f"Failed to delete artifact {artifact['gcs_uri']}: {e}")
            
            logger.info(f"Cleaned up {deleted_count} artifacts for job {job_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup artifacts for job {job_id}: {e}")
            return 0


def create_worker_uploader(gcs_config: GCSConfig) -> Optional[WorkerGCSUploader]:
    """
    Factory function to create a worker GCS uploader.
    
    Args:
        gcs_config: GCS configuration
        
    Returns:
        WorkerGCSUploader instance, or None if initialization fails
    """
    try:
        from api.gcs_client import create_gcs_client
        
        gcs_client = create_gcs_client(gcs_config)
        return WorkerGCSUploader(gcs_client)
        
    except Exception as e:
        logger.error(f"Failed to create worker GCS uploader: {e}")
        return None


def upload_job_artifact(
    local_video_path: str, 
    job_id: str, 
    gcs_config: GCSConfig,
    cleanup_local: bool = False
) -> Optional[str]:
    """
    Convenience function to upload a job artifact with minimal setup.
    
    This is useful for simple upload scenarios where you don't need to maintain
    a persistent uploader instance.
    
    Args:
        local_video_path: Path to the local video file
        job_id: Job identifier
        gcs_config: GCS configuration
        cleanup_local: Whether to delete the local file after upload
        
    Returns:
        GCS URI of the uploaded artifact, or None if upload failed
    """
    uploader = create_worker_uploader(gcs_config)
    if not uploader:
        return None
    
    return uploader.upload_video_artifact(
        local_video_path=local_video_path,
        job_id=job_id,
        cleanup_local=cleanup_local
    )