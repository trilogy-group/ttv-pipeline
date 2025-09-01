"""
Google Cloud Storage client for artifact management.

This module provides GCS integration for uploading video artifacts and generating
signed URLs for download. It reuses existing pipeline credentials and follows
the path structure defined in the requirements.
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

from google.cloud import storage
from google.cloud.exceptions import NotFound, GoogleCloudError
from google.auth.exceptions import DefaultCredentialsError

from .config import GCSConfig

logger = logging.getLogger(__name__)


class GCSClientError(Exception):
    """Base exception for GCS client errors"""
    pass


class GCSUploadError(GCSClientError):
    """Exception raised when GCS upload fails"""
    pass


class GCSCredentialsError(GCSClientError):
    """Exception raised when GCS credentials are invalid or missing"""
    pass


class GCSClient:
    """
    Google Cloud Storage client for artifact management.
    
    Handles video artifact uploads, signed URL generation, and bucket management
    using existing pipeline credentials.
    """
    
    def __init__(self, config: GCSConfig):
        """
        Initialize GCS client with configuration.
        
        Args:
            config: GCS configuration containing bucket, credentials, etc.
            
        Raises:
            GCSCredentialsError: If credentials cannot be loaded
        """
        self.config = config
        self._client = None
        self._bucket = None
        
        # Initialize the client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the GCS client using existing pipeline credentials"""
        try:
            # Try to use credentials from config first
            if self.config.credentials_path and Path(self.config.credentials_path).exists():
                logger.info(f"Using GCS credentials from: {self.config.credentials_path}")
                self._client = storage.Client.from_service_account_json(
                    self.config.credentials_path
                )
            else:
                # Fall back to default credentials (environment variables, metadata server, etc.)
                logger.info("Using default GCS credentials")
                self._client = storage.Client()
            
            # Get or create the bucket
            self._bucket = self._get_or_create_bucket()
            
            logger.info(f"GCS client initialized successfully for bucket: {self.config.bucket}")
            
        except DefaultCredentialsError as e:
            error_msg = f"GCS credentials not found: {e}"
            logger.error(error_msg)
            raise GCSCredentialsError(error_msg)
        except Exception as e:
            error_msg = f"Failed to initialize GCS client: {e}"
            logger.error(error_msg)
            raise GCSCredentialsError(error_msg)
    
    def _get_or_create_bucket(self) -> storage.Bucket:
        """
        Get the configured bucket, creating it if it doesn't exist.
        
        Returns:
            GCS Bucket object
            
        Raises:
            GCSClientError: If bucket cannot be accessed or created
        """
        try:
            # Try to get existing bucket
            bucket = self._client.bucket(self.config.bucket)
            
            # Test if bucket exists by trying to get its metadata
            bucket.reload()
            logger.info(f"Using existing GCS bucket: {self.config.bucket}")
            return bucket
            
        except NotFound:
            # Bucket doesn't exist, try to create it
            logger.info(f"Bucket {self.config.bucket} not found, attempting to create it")
            try:
                bucket = self._client.create_bucket(self.config.bucket)
                logger.info(f"Created GCS bucket: {self.config.bucket}")
                return bucket
            except Exception as e:
                error_msg = f"Failed to create bucket {self.config.bucket}: {e}"
                logger.error(error_msg)
                raise GCSClientError(error_msg)
        except Exception as e:
            error_msg = f"Failed to access bucket {self.config.bucket}: {e}"
            logger.error(error_msg)
            raise GCSClientError(error_msg)
    
    def generate_artifact_path(self, job_id: str, filename: str = "final_video.mp4") -> str:
        """
        Generate the GCS path for an artifact following the required format.
        
        Format: gs://{bucket}/{prefix}/{YYYY-MM}/{job_id}/final_video.mp4
        
        Args:
            job_id: Unique job identifier
            filename: Name of the artifact file (default: final_video.mp4)
            
        Returns:
            Full GCS URI for the artifact
        """
        # Get current date for path structure
        now = datetime.utcnow()
        date_prefix = now.strftime("%Y-%m")
        
        # Build the path components
        path_parts = [
            self.config.prefix,
            date_prefix,
            job_id,
            filename
        ]
        
        # Join path parts and create full GCS URI
        blob_path = "/".join(path_parts)
        gcs_uri = f"gs://{self.config.bucket}/{blob_path}"
        
        logger.debug(f"Generated artifact path for job {job_id}: {gcs_uri}")
        return gcs_uri
    
    def upload_artifact(
        self, 
        local_file_path: str, 
        job_id: str, 
        filename: str = "final_video.mp4",
        max_retries: int = 3
    ) -> str:
        """
        Upload a video artifact to GCS with retry logic.
        
        Args:
            local_file_path: Path to the local file to upload
            job_id: Unique job identifier
            filename: Name for the uploaded file (default: final_video.mp4)
            max_retries: Maximum number of retry attempts
            
        Returns:
            GCS URI of the uploaded artifact
            
        Raises:
            GCSUploadError: If upload fails after all retries
            FileNotFoundError: If local file doesn't exist
        """
        local_path = Path(local_file_path)
        
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_file_path}")
        
        # Generate the GCS path
        gcs_uri = self.generate_artifact_path(job_id, filename)
        
        # Extract blob path from URI
        parsed = urlparse(gcs_uri)
        blob_path = parsed.path.lstrip('/')
        
        logger.info(f"Uploading {local_file_path} to {gcs_uri}")
        
        # Retry upload with exponential backoff
        for attempt in range(max_retries):
            try:
                # Create blob and upload
                blob = self._bucket.blob(blob_path)
                
                # Set content type for video files
                if filename.endswith('.mp4'):
                    blob.content_type = 'video/mp4'
                elif filename.endswith('.mov'):
                    blob.content_type = 'video/quicktime'
                else:
                    blob.content_type = 'application/octet-stream'
                
                # Upload the file
                with open(local_path, 'rb') as f:
                    blob.upload_from_file(f)
                
                # Verify upload
                blob.reload()
                file_size = local_path.stat().st_size
                uploaded_size = blob.size
                
                if uploaded_size != file_size:
                    raise GCSUploadError(
                        f"Upload verification failed: local size {file_size}, "
                        f"uploaded size {uploaded_size}"
                    )
                
                logger.info(f"Successfully uploaded {local_file_path} to {gcs_uri} "
                          f"({uploaded_size} bytes)")
                return gcs_uri
                
            except Exception as e:
                attempt_num = attempt + 1
                if attempt_num >= max_retries:
                    error_msg = f"Upload failed after {max_retries} attempts: {e}"
                    logger.error(error_msg)
                    raise GCSUploadError(error_msg)
                
                # Calculate exponential backoff delay
                delay = 2 ** attempt
                logger.warning(f"Upload attempt {attempt_num} failed: {e}. "
                             f"Retrying in {delay} seconds...")
                
                import time
                time.sleep(delay)
        
        # This should never be reached due to the exception handling above
        raise GCSUploadError("Upload failed for unknown reason")
    
    def generate_signed_url(
        self, 
        gcs_uri: str, 
        expiration_seconds: Optional[int] = None,
        method: str = "GET"
    ) -> str:
        """
        Generate a signed URL for downloading an artifact.
        
        Args:
            gcs_uri: Full GCS URI of the artifact
            expiration_seconds: URL expiration time (uses config default if None)
            method: HTTP method for the signed URL (default: GET)
            
        Returns:
            Signed HTTPS URL for downloading the artifact
            
        Raises:
            GCSClientError: If signed URL generation fails
            ValueError: If GCS URI is invalid
        """
        # Parse the GCS URI
        parsed = urlparse(gcs_uri)
        if parsed.scheme != 'gs':
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")
        
        bucket_name = parsed.netloc
        blob_path = parsed.path.lstrip('/')
        
        if bucket_name != self.config.bucket:
            raise ValueError(f"URI bucket {bucket_name} doesn't match config bucket {self.config.bucket}")
        
        # Use configured expiration if not specified
        if expiration_seconds is None:
            expiration_seconds = self.config.signed_url_expiration
        
        try:
            # Get the blob
            blob = self._bucket.blob(blob_path)
            
            # Check if blob exists
            if not blob.exists():
                raise GCSClientError(f"Artifact not found: {gcs_uri}")
            
            # Generate signed URL with proper headers for web embedding
            expiration = datetime.utcnow() + timedelta(seconds=expiration_seconds)
            
            # Set response headers for video streaming
            response_headers = {}
            if blob_path.endswith('.mp4'):
                response_headers['content-type'] = 'video/mp4'
            elif blob_path.endswith('.mov'):
                response_headers['content-type'] = 'video/quicktime'
            else:
                response_headers['content-type'] = 'application/octet-stream'
            
            # Set content-disposition to inline for web streaming (not download)
            response_headers['content-disposition'] = 'inline'
            
            signed_url = blob.generate_signed_url(
                expiration=expiration,
                method=method,
                version="v4",  # Use v4 signing for better security
                response_disposition=response_headers.get('content-disposition'),
                response_type=response_headers.get('content-type')
            )
            
            logger.info(f"Generated signed URL for {gcs_uri} (expires in {expiration_seconds}s)")
            return signed_url
            
        except Exception as e:
            error_msg = f"Failed to generate signed URL for {gcs_uri}: {e}"
            logger.error(error_msg)
            raise GCSClientError(error_msg)
    
    def delete_artifact(self, gcs_uri: str) -> bool:
        """
        Delete an artifact from GCS.
        
        Args:
            gcs_uri: Full GCS URI of the artifact to delete
            
        Returns:
            True if deletion was successful, False if artifact didn't exist
            
        Raises:
            GCSClientError: If deletion fails for reasons other than not found
        """
        # Parse the GCS URI
        parsed = urlparse(gcs_uri)
        if parsed.scheme != 'gs':
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")
        
        bucket_name = parsed.netloc
        blob_path = parsed.path.lstrip('/')
        
        if bucket_name != self.config.bucket:
            raise ValueError(f"URI bucket {bucket_name} doesn't match config bucket {self.config.bucket}")
        
        try:
            # Get the blob and delete it
            blob = self._bucket.blob(blob_path)
            
            if not blob.exists():
                logger.info(f"Artifact not found for deletion: {gcs_uri}")
                return False
            
            blob.delete()
            logger.info(f"Deleted artifact: {gcs_uri}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to delete artifact {gcs_uri}: {e}"
            logger.error(error_msg)
            raise GCSClientError(error_msg)
    
    def list_artifacts(self, job_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List artifacts in the bucket, optionally filtered by job ID.
        
        Args:
            job_id: Optional job ID to filter artifacts
            limit: Maximum number of artifacts to return
            
        Returns:
            List of artifact metadata dictionaries
        """
        try:
            # Build prefix for filtering
            if job_id:
                # List artifacts for specific job
                now = datetime.utcnow()
                date_prefix = now.strftime("%Y-%m")
                prefix = f"{self.config.prefix}/{date_prefix}/{job_id}/"
            else:
                # List all artifacts under the configured prefix
                prefix = f"{self.config.prefix}/"
            
            # List blobs with the prefix
            blobs = self._client.list_blobs(
                self._bucket,
                prefix=prefix,
                max_results=limit
            )
            
            artifacts = []
            for blob in blobs:
                artifacts.append({
                    'name': blob.name,
                    'gcs_uri': f"gs://{self.config.bucket}/{blob.name}",
                    'size': blob.size,
                    'created': blob.time_created,
                    'updated': blob.updated,
                    'content_type': blob.content_type
                })
            
            logger.info(f"Listed {len(artifacts)} artifacts with prefix: {prefix}")
            return artifacts
            
        except Exception as e:
            error_msg = f"Failed to list artifacts: {e}"
            logger.error(error_msg)
            raise GCSClientError(error_msg)
    
    def get_bucket_info(self) -> Dict[str, Any]:
        """
        Get information about the configured bucket.
        
        Returns:
            Dictionary containing bucket metadata
        """
        try:
            self._bucket.reload()
            
            return {
                'name': self._bucket.name,
                'location': self._bucket.location,
                'storage_class': self._bucket.storage_class,
                'created': self._bucket.time_created,
                'updated': self._bucket.updated,
                'versioning_enabled': self._bucket.versioning_enabled,
                'lifecycle_rules': len(self._bucket.lifecycle_rules) if self._bucket.lifecycle_rules else 0
            }
            
        except Exception as e:
            error_msg = f"Failed to get bucket info: {e}"
            logger.error(error_msg)
            raise GCSClientError(error_msg)
    
    async def test_connection(self) -> bool:
        """
        Test GCS connectivity by attempting to access the bucket.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Try to reload bucket metadata to test connectivity
            self._bucket.reload()
            logger.debug("GCS connection test successful")
            return True
        except Exception as e:
            logger.error(f"GCS connection test failed: {e}")
            return False
    
    def setup_lifecycle_rules(self):
        """
        Set up lifecycle rules for automatic cleanup of old artifacts.
        
        This configures the bucket to automatically delete artifacts older than 30 days
        to manage storage costs.
        """
        try:
            # Define lifecycle rule to delete objects older than 30 days
            lifecycle_rule = {
                'action': {'type': 'Delete'},
                'condition': {
                    'age': 30,  # Delete after 30 days
                    'matchesPrefix': [self.config.prefix + '/']  # Only affect our artifacts
                }
            }
            
            # Get current lifecycle rules
            current_rules = list(self._bucket.lifecycle_rules) if self._bucket.lifecycle_rules else []
            
            # Check if our rule already exists
            rule_exists = any(
                rule.get('condition', {}).get('matchesPrefix') == [self.config.prefix + '/']
                for rule in current_rules
            )
            
            if not rule_exists:
                # Add our lifecycle rule
                current_rules.append(lifecycle_rule)
                self._bucket.lifecycle_rules = current_rules
                self._bucket.patch()
                
                logger.info(f"Added lifecycle rule for prefix {self.config.prefix}/ (30 day retention)")
            else:
                logger.info("Lifecycle rule already exists for artifact cleanup")
                
        except Exception as e:
            # Log warning but don't fail - lifecycle rules are optional
            logger.warning(f"Failed to set up lifecycle rules: {e}")


def create_gcs_client(config: GCSConfig) -> GCSClient:
    """
    Factory function to create a GCS client with the given configuration.
    
    Args:
        config: GCS configuration
        
    Returns:
        Initialized GCS client
        
    Raises:
        GCSCredentialsError: If credentials cannot be loaded
        GCSClientError: If client initialization fails
    """
    return GCSClient(config)