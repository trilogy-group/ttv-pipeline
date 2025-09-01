"""
Unit tests for worker GCS uploader functionality.
"""

import os
import tempfile
from unittest.mock import Mock, patch
import pytest

from workers.gcs_uploader import (
    WorkerGCSUploader,
    create_worker_uploader,
    upload_job_artifact
)
from api.gcs_client import GCSClient, GCSUploadError, GCSClientError
from api.config import GCSConfig


@pytest.fixture
def mock_gcs_client():
    """Create a mock GCS client"""
    return Mock(spec=GCSClient)


@pytest.fixture
def gcs_config():
    """Create a test GCS configuration"""
    return GCSConfig(
        bucket="test-bucket",
        prefix="test-prefix",
        signed_url_expiration=3600
    )


class TestWorkerGCSUploader:
    """Test cases for WorkerGCSUploader"""
    
    def test_init(self, mock_gcs_client):
        """Test uploader initialization"""
        uploader = WorkerGCSUploader(mock_gcs_client)
        assert uploader.gcs_client == mock_gcs_client
    
    def test_upload_video_artifact_success(self, mock_gcs_client):
        """Test successful video artifact upload"""
        uploader = WorkerGCSUploader(mock_gcs_client)
        
        # Create a temporary video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(b"fake video content")
            temp_path = temp_file.name
        
        try:
            # Mock successful upload
            expected_uri = "gs://test-bucket/test-prefix/2025-08/job123/final_video.mp4"
            mock_gcs_client.upload_artifact.return_value = expected_uri
            
            result = uploader.upload_video_artifact(temp_path, "job123")
            
            assert result == expected_uri
            mock_gcs_client.upload_artifact.assert_called_once_with(
                local_file_path=temp_path,
                job_id="job123",
                filename="final_video.mp4"
            )
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_upload_video_artifact_with_cleanup(self, mock_gcs_client):
        """Test video artifact upload with local file cleanup"""
        uploader = WorkerGCSUploader(mock_gcs_client)
        
        # Create a temporary video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(b"fake video content")
            temp_path = temp_file.name
        
        # Mock successful upload
        expected_uri = "gs://test-bucket/test-prefix/2025-08/job123/final_video.mp4"
        mock_gcs_client.upload_artifact.return_value = expected_uri
        
        result = uploader.upload_video_artifact(temp_path, "job123", cleanup_local=True)
        
        assert result == expected_uri
        # File should be deleted after successful upload
        assert not os.path.exists(temp_path)
    
    def test_upload_video_artifact_file_not_found(self, mock_gcs_client):
        """Test upload with non-existent file"""
        uploader = WorkerGCSUploader(mock_gcs_client)
        
        result = uploader.upload_video_artifact("/nonexistent/file.mp4", "job123")
        
        assert result is None
        mock_gcs_client.upload_artifact.assert_not_called()
    
    def test_upload_video_artifact_upload_error(self, mock_gcs_client):
        """Test upload with GCS upload error"""
        uploader = WorkerGCSUploader(mock_gcs_client)
        
        # Create a temporary video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(b"fake video content")
            temp_path = temp_file.name
        
        try:
            # Mock upload error
            mock_gcs_client.upload_artifact.side_effect = GCSUploadError("Upload failed")
            
            result = uploader.upload_video_artifact(temp_path, "job123")
            
            assert result is None
            mock_gcs_client.upload_artifact.assert_called_once()
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])