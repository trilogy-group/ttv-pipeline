"""
Integration tests for GCS functionality with existing pipeline configuration.
"""

import os
import tempfile
from unittest.mock import Mock, patch
import pytest

from api.config import load_api_config, GCSConfig
from api.gcs_client import create_gcs_client, GCSClient
from workers.gcs_uploader import create_worker_uploader


class TestGCSIntegration:
    """Integration tests for GCS with pipeline configuration"""
    
    def test_load_gcs_config_from_pipeline(self):
        """Test loading GCS configuration from pipeline config"""
        # Mock pipeline config with Google Veo settings
        mock_pipeline_config = {
            'google_veo': {
                'project_id': 'test-project',
                'credentials_path': 'test-credentials.json',
                'output_bucket': 'test-output-bucket'
            }
        }
        
        # Mock API config
        mock_api_config = {
            'gcs': {
                'prefix': 'custom-prefix',
                'signed_url_expiration': 7200
            }
        }
        
        with patch('api.config.load_pipeline_config', return_value=mock_pipeline_config):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', mock_open_yaml(mock_api_config)):
                    config = load_api_config('api_config.yaml', 'pipeline_config.yaml')
        
        # Verify GCS config was properly merged
        assert config.gcs.bucket == 'test-output-bucket'  # From pipeline config
        assert config.gcs.credentials_path == 'test-credentials.json'  # From pipeline config
        assert config.gcs.prefix == 'custom-prefix'  # From API config override
        assert config.gcs.signed_url_expiration == 7200  # From API config override
    
    def test_load_gcs_config_defaults(self):
        """Test loading GCS configuration with defaults when no pipeline config"""
        mock_pipeline_config = {}
        
        with patch('api.config.load_pipeline_config', return_value=mock_pipeline_config):
            with patch('pathlib.Path.exists', return_value=False):
                config = load_api_config(None, 'pipeline_config.yaml')
        
        # Should use default bucket name
        assert config.gcs.bucket == 'ttv-api-artifacts'
        assert config.gcs.prefix == 'ttv-api'
        assert config.gcs.signed_url_expiration == 3600
    
    @patch('api.gcs_client.storage.Client')
    def test_create_gcs_client_with_pipeline_credentials(self, mock_storage_client):
        """Test creating GCS client with pipeline credentials"""
        # Mock storage client
        mock_client = Mock()
        mock_storage_client.from_service_account_json.return_value = mock_client
        mock_bucket = Mock()
        mock_client.bucket.return_value = mock_bucket
        
        # Create GCS config with credentials path
        gcs_config = GCSConfig(
            bucket='test-bucket',
            credentials_path='ai-coe-454404-df4ebc146821.json'
        )
        
        with patch('pathlib.Path.exists', return_value=True):
            client = create_gcs_client(gcs_config)
        
        # Verify client was created with service account credentials
        mock_storage_client.from_service_account_json.assert_called_once_with(
            'ai-coe-454404-df4ebc146821.json'
        )
        assert isinstance(client, GCSClient)
    
    @patch('api.gcs_client.storage.Client')
    def test_create_worker_uploader_integration(self, mock_storage_client):
        """Test creating worker uploader with GCS client"""
        # Mock storage client
        mock_client = Mock()
        mock_storage_client.return_value = mock_client
        mock_bucket = Mock()
        mock_client.bucket.return_value = mock_bucket
        
        # Create GCS config
        gcs_config = GCSConfig(
            bucket='ttv-api-artifacts',
            prefix='ttv-api'
        )
        
        uploader = create_worker_uploader(gcs_config)
        
        assert uploader is not None
        assert hasattr(uploader, 'gcs_client')
        assert isinstance(uploader.gcs_client, GCSClient)
    
    def test_artifact_path_generation_format(self):
        """Test that artifact paths follow the required format"""
        gcs_config = GCSConfig(
            bucket='ttv-api-artifacts',
            prefix='ttv-api'
        )
        
        with patch('api.gcs_client.storage.Client'):
            with patch('api.gcs_client.GCSClient._get_or_create_bucket'):
                client = GCSClient(gcs_config)
                
                with patch('api.gcs_client.datetime') as mock_datetime:
                    from datetime import datetime
                    mock_datetime.utcnow.return_value = datetime(2025, 8, 31, 12, 0, 0)
                    
                    path = client.generate_artifact_path('job-12345')
                    
                    # Verify format: gs://{bucket}/{prefix}/{YYYY-MM}/{job_id}/final_video.mp4
                    expected = 'gs://ttv-api-artifacts/ttv-api/2025-08/job-12345/final_video.mp4'
                    assert path == expected


def mock_open_yaml(data):
    """Helper to mock yaml file opening"""
    import yaml
    from unittest.mock import mock_open
    
    yaml_content = yaml.dump(data)
    return mock_open(read_data=yaml_content)


if __name__ == "__main__":
    pytest.main([__file__])