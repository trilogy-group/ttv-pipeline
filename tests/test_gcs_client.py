"""
Unit tests for GCS client functionality.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from api.gcs_client import (
    GCSClient, 
    GCSClientError, 
    GCSUploadError, 
    GCSCredentialsError,
    create_gcs_client
)
from api.config import GCSConfig


@pytest.fixture
def gcs_config():
    """Create a test GCS configuration"""
    return GCSConfig(
        bucket="test-bucket",
        prefix="test-prefix",
        credentials_path="/path/to/credentials.json",
        signed_url_expiration=3600
    )


@pytest.fixture
def mock_storage_client():
    """Create a mock Google Cloud Storage client"""
    with patch('api.gcs_client.storage.Client') as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client_class.from_service_account_json.return_value = mock_client
        
        # Mock bucket
        mock_bucket = Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_client.create_bucket.return_value = mock_bucket
        
        yield mock_client, mock_bucket


class TestGCSClient:
    """Test cases for GCSClient"""
    
    def test_init_with_credentials_file(self, gcs_config, mock_storage_client):
        """Test initialization with credentials file"""
        mock_client, mock_bucket = mock_storage_client
        
        with patch('pathlib.Path.exists', return_value=True):
            client = GCSClient(gcs_config)
            
            assert client.config == gcs_config
            assert client._client == mock_client
            assert client._bucket == mock_bucket
    
    def test_init_with_default_credentials(self, mock_storage_client):
        """Test initialization with default credentials"""
        mock_client, mock_bucket = mock_storage_client
        
        config = GCSConfig(bucket="test-bucket")
        client = GCSClient(config)
        
        assert client._client == mock_client
        assert client._bucket == mock_bucket
    
    def test_init_credentials_error(self, gcs_config):
        """Test initialization with invalid credentials"""
        with patch('api.gcs_client.storage.Client') as mock_client_class:
            from google.auth.exceptions import DefaultCredentialsError
            mock_client_class.side_effect = DefaultCredentialsError("No credentials")
            
            with pytest.raises(GCSCredentialsError):
                GCSClient(gcs_config)
    
    def test_generate_artifact_path(self, gcs_config, mock_storage_client):
        """Test artifact path generation"""
        client = GCSClient(gcs_config)
        
        with patch('api.gcs_client.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2025, 8, 31, 12, 0, 0)
            
            path = client.generate_artifact_path("job123")
            expected = "gs://test-bucket/test-prefix/2025-08/job123/final_video.mp4"
            assert path == expected
    
    def test_generate_artifact_path_custom_filename(self, gcs_config, mock_storage_client):
        """Test artifact path generation with custom filename"""
        client = GCSClient(gcs_config)
        
        with patch('api.gcs_client.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2025, 8, 31, 12, 0, 0)
            
            path = client.generate_artifact_path("job123", "custom.mp4")
            expected = "gs://test-bucket/test-prefix/2025-08/job123/custom.mp4"
            assert path == expected
    
    def test_upload_artifact_success(self, gcs_config, mock_storage_client):
        """Test successful artifact upload"""
        mock_client, mock_bucket = mock_storage_client
        client = GCSClient(gcs_config)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test video content")
            temp_path = temp_file.name
        
        try:
            # Mock blob
            mock_blob = Mock()
            mock_bucket.blob.return_value = mock_blob
            mock_blob.size = 18  # Length of "test video content"
            
            with patch('api.gcs_client.datetime') as mock_datetime:
                mock_datetime.utcnow.return_value = datetime(2025, 8, 31, 12, 0, 0)
                
                gcs_uri = client.upload_artifact(temp_path, "job123")
                
                expected_uri = "gs://test-bucket/test-prefix/2025-08/job123/final_video.mp4"
                assert gcs_uri == expected_uri
                
                # Verify blob operations
                mock_blob.upload_from_file.assert_called_once()
                mock_blob.reload.assert_called_once()
                assert mock_blob.content_type == 'video/mp4'
        
        finally:
            # Clean up temp file
            os.unlink(temp_path)
    
    def test_upload_artifact_file_not_found(self, gcs_config, mock_storage_client):
        """Test upload with non-existent file"""
        client = GCSClient(gcs_config)
        
        with pytest.raises(FileNotFoundError):
            client.upload_artifact("/nonexistent/file.mp4", "job123")
    
    def test_upload_artifact_retry_logic(self, gcs_config, mock_storage_client):
        """Test upload retry logic on failure"""
        mock_client, mock_bucket = mock_storage_client
        client = GCSClient(gcs_config)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test content")
            temp_path = temp_file.name
        
        try:
            # Mock blob that fails twice then succeeds
            mock_blob = Mock()
            mock_bucket.blob.return_value = mock_blob
            mock_blob.upload_from_file.side_effect = [
                Exception("Network error"),
                Exception("Another error"),
                None  # Success on third attempt
            ]
            mock_blob.size = 12  # Length of "test content"
            
            with patch('time.sleep'):  # Speed up the test
                with patch('api.gcs_client.datetime') as mock_datetime:
                    mock_datetime.utcnow.return_value = datetime(2025, 8, 31, 12, 0, 0)
                    
                    gcs_uri = client.upload_artifact(temp_path, "job123", max_retries=3)
                    
                    expected_uri = "gs://test-bucket/test-prefix/2025-08/job123/final_video.mp4"
                    assert gcs_uri == expected_uri
                    
                    # Should have been called 3 times (2 failures + 1 success)
                    assert mock_blob.upload_from_file.call_count == 3
        
        finally:
            os.unlink(temp_path)
    
    def test_upload_artifact_max_retries_exceeded(self, gcs_config, mock_storage_client):
        """Test upload failure after max retries"""
        mock_client, mock_bucket = mock_storage_client
        client = GCSClient(gcs_config)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test content")
            temp_path = temp_file.name
        
        try:
            # Mock blob that always fails
            mock_blob = Mock()
            mock_bucket.blob.return_value = mock_blob
            mock_blob.upload_from_file.side_effect = Exception("Persistent error")
            
            with patch('time.sleep'):  # Speed up the test
                with pytest.raises(GCSUploadError):
                    client.upload_artifact(temp_path, "job123", max_retries=2)
        
        finally:
            os.unlink(temp_path)
    
    def test_generate_signed_url_success(self, gcs_config, mock_storage_client):
        """Test successful signed URL generation"""
        mock_client, mock_bucket = mock_storage_client
        client = GCSClient(gcs_config)
        
        # Mock blob
        mock_blob = Mock()
        mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = True
        mock_blob.generate_signed_url.return_value = "https://signed-url.example.com"
        
        gcs_uri = "gs://test-bucket/test-prefix/2025-08/job123/final_video.mp4"
        signed_url = client.generate_signed_url(gcs_uri)
        
        assert signed_url == "https://signed-url.example.com"
        mock_blob.exists.assert_called_once()
        mock_blob.generate_signed_url.assert_called_once()
    
    def test_generate_signed_url_invalid_uri(self, gcs_config, mock_storage_client):
        """Test signed URL generation with invalid URI"""
        client = GCSClient(gcs_config)
        
        with pytest.raises(ValueError):
            client.generate_signed_url("http://invalid-uri.com")
    
    def test_generate_signed_url_wrong_bucket(self, gcs_config, mock_storage_client):
        """Test signed URL generation with wrong bucket"""
        client = GCSClient(gcs_config)
        
        with pytest.raises(ValueError):
            client.generate_signed_url("gs://wrong-bucket/path/to/file.mp4")
    
    def test_generate_signed_url_blob_not_found(self, gcs_config, mock_storage_client):
        """Test signed URL generation for non-existent blob"""
        mock_client, mock_bucket = mock_storage_client
        client = GCSClient(gcs_config)
        
        # Mock blob that doesn't exist
        mock_blob = Mock()
        mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = False
        
        gcs_uri = "gs://test-bucket/test-prefix/2025-08/job123/final_video.mp4"
        
        with pytest.raises(GCSClientError):
            client.generate_signed_url(gcs_uri)
    
    def test_delete_artifact_success(self, gcs_config, mock_storage_client):
        """Test successful artifact deletion"""
        mock_client, mock_bucket = mock_storage_client
        client = GCSClient(gcs_config)
        
        # Mock blob
        mock_blob = Mock()
        mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = True
        
        gcs_uri = "gs://test-bucket/test-prefix/2025-08/job123/final_video.mp4"
        result = client.delete_artifact(gcs_uri)
        
        assert result is True
        mock_blob.delete.assert_called_once()
    
    def test_delete_artifact_not_found(self, gcs_config, mock_storage_client):
        """Test deletion of non-existent artifact"""
        mock_client, mock_bucket = mock_storage_client
        client = GCSClient(gcs_config)
        
        # Mock blob that doesn't exist
        mock_blob = Mock()
        mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = False
        
        gcs_uri = "gs://test-bucket/test-prefix/2025-08/job123/final_video.mp4"
        result = client.delete_artifact(gcs_uri)
        
        assert result is False
        mock_blob.delete.assert_not_called()
    
    def test_list_artifacts_all(self, gcs_config, mock_storage_client):
        """Test listing all artifacts"""
        mock_client, mock_bucket = mock_storage_client
        client = GCSClient(gcs_config)
        
        # Mock blobs
        mock_blob1 = Mock()
        mock_blob1.name = "test-prefix/2025-08/job1/final_video.mp4"
        mock_blob1.size = 1000
        mock_blob1.time_created = datetime(2025, 8, 31, 12, 0, 0)
        mock_blob1.updated = datetime(2025, 8, 31, 12, 0, 0)
        mock_blob1.content_type = "video/mp4"
        
        mock_blob2 = Mock()
        mock_blob2.name = "test-prefix/2025-08/job2/final_video.mp4"
        mock_blob2.size = 2000
        mock_blob2.time_created = datetime(2025, 8, 31, 13, 0, 0)
        mock_blob2.updated = datetime(2025, 8, 31, 13, 0, 0)
        mock_blob2.content_type = "video/mp4"
        
        mock_client.list_blobs.return_value = [mock_blob1, mock_blob2]
        
        artifacts = client.list_artifacts()
        
        assert len(artifacts) == 2
        assert artifacts[0]['name'] == "test-prefix/2025-08/job1/final_video.mp4"
        assert artifacts[0]['gcs_uri'] == "gs://test-bucket/test-prefix/2025-08/job1/final_video.mp4"
        assert artifacts[0]['size'] == 1000
        
        mock_client.list_blobs.assert_called_once_with(
            mock_bucket,
            prefix="test-prefix/",
            max_results=100
        )
    
    def test_list_artifacts_by_job_id(self, gcs_config, mock_storage_client):
        """Test listing artifacts for specific job"""
        mock_client, mock_bucket = mock_storage_client
        client = GCSClient(gcs_config)
        
        mock_client.list_blobs.return_value = []
        
        with patch('api.gcs_client.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2025, 8, 31, 12, 0, 0)
            
            artifacts = client.list_artifacts(job_id="job123")
            
            mock_client.list_blobs.assert_called_once_with(
                mock_bucket,
                prefix="test-prefix/2025-08/job123/",
                max_results=100
            )
    
    def test_get_bucket_info(self, gcs_config, mock_storage_client):
        """Test getting bucket information"""
        mock_client, mock_bucket = mock_storage_client
        client = GCSClient(gcs_config)
        
        # Mock bucket properties
        mock_bucket.name = "test-bucket"
        mock_bucket.location = "US"
        mock_bucket.storage_class = "STANDARD"
        mock_bucket.time_created = datetime(2025, 1, 1, 0, 0, 0)
        mock_bucket.updated = datetime(2025, 8, 31, 12, 0, 0)
        mock_bucket.versioning_enabled = False
        mock_bucket.lifecycle_rules = []
        
        info = client.get_bucket_info()
        
        assert info['name'] == "test-bucket"
        assert info['location'] == "US"
        assert info['storage_class'] == "STANDARD"
        assert info['versioning_enabled'] is False
        assert info['lifecycle_rules'] == 0
        
        # reload is called during initialization and in get_bucket_info
        assert mock_bucket.reload.call_count >= 1
    
    def test_setup_lifecycle_rules(self, gcs_config, mock_storage_client):
        """Test setting up lifecycle rules"""
        mock_client, mock_bucket = mock_storage_client
        client = GCSClient(gcs_config)
        
        # Mock existing lifecycle rules (empty)
        mock_bucket.lifecycle_rules = []
        
        client.setup_lifecycle_rules()
        
        # Should have added our lifecycle rule
        mock_bucket.patch.assert_called_once()
        
        # Check that lifecycle_rules was set
        assert hasattr(mock_bucket, 'lifecycle_rules')


class TestCreateGCSClient:
    """Test cases for create_gcs_client factory function"""
    
    def test_create_gcs_client_success(self, gcs_config):
        """Test successful client creation"""
        with patch('api.gcs_client.GCSClient') as mock_gcs_client_class:
            mock_client = Mock()
            mock_gcs_client_class.return_value = mock_client
            
            result = create_gcs_client(gcs_config)
            
            assert result == mock_client
            mock_gcs_client_class.assert_called_once_with(gcs_config)
    
    def test_create_gcs_client_error(self, gcs_config):
        """Test client creation with error"""
        with patch('api.gcs_client.GCSClient') as mock_gcs_client_class:
            mock_gcs_client_class.side_effect = GCSCredentialsError("No credentials")
            
            with pytest.raises(GCSCredentialsError):
                create_gcs_client(gcs_config)


if __name__ == "__main__":
    pytest.main([__file__])