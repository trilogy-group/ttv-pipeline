"""
Pytest configuration and fixtures for the API test suite.
"""

import os
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any

import pytest

from api.config import APIConfig
from api.config_merger import ConfigMerger


@pytest.fixture
def sample_pipeline_config() -> Dict[str, Any]:
    """Sample pipeline configuration for testing"""
    return {
        "task": "flf2v-14B",
        "size": "1280*720",
        "prompt": "A sample video prompt for testing",
        "default_backend": "veo3",
        "generation_mode": "keyframe",
        "wan2_dir": "./Wan2.1",
        "flf2v_model_dir": "./models/Wan2.1-FLF2V-14B-720P",
        "total_gpus": 8,
        "parallel_segments": 1,
        "segment_duration_seconds": 5.0,
        "frame_num": 81,
        "sample_steps": 40,
        "guide_scale": 5.0,
        "base_seed": 42,
        "output_dir": "output",
        "google_veo": {
            "project_id": "test-project",
            "credentials_path": "test-credentials.json",
            "region": "us-central1",
            "output_bucket": "test-bucket"
        },
        "runway_ml": {
            "api_key": "test-runway-key",
            "model_version": "gen4_turbo",
            "max_duration": 5
        },
        "minimax": {
            "api_key": "test-minimax-key",
            "model": "I2V-01-Director",
            "max_duration": 6
        },
        "remote_api_settings": {
            "max_retries": 3,
            "timeout": 600
        }
    }


@pytest.fixture
def sample_api_config() -> Dict[str, Any]:
    """Sample API configuration for testing"""
    return {
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "quic_port": 8443,
            "workers": 2,
            "cors_origins": ["*"]
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0
        },
        "gcs": {
            "bucket": "test-api-bucket",
            "prefix": "ttv-api",
            "signed_url_expiration": 3600
        },
        "security": {
            "auth_token": "test-token",
            "rate_limit_per_minute": 60
        }
    }


@pytest.fixture
def temp_config_files(sample_pipeline_config, sample_api_config):
    """Create temporary configuration files for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create pipeline config file
        pipeline_config_path = temp_path / "pipeline_config.yaml"
        with open(pipeline_config_path, 'w') as f:
            yaml.dump(sample_pipeline_config, f)
        
        # Create API config file
        api_config_path = temp_path / "api_config.yaml"
        with open(api_config_path, 'w') as f:
            yaml.dump(sample_api_config, f)
        
        yield {
            "pipeline_config_path": str(pipeline_config_path),
            "api_config_path": str(api_config_path),
            "temp_dir": temp_dir
        }


@pytest.fixture
def config_merger():
    """ConfigMerger instance for testing"""
    return ConfigMerger()


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing"""
    env_vars = {
        "API_HOST": "test-host",
        "API_PORT": "9000",
        "REDIS_HOST": "test-redis",
        "REDIS_PORT": "6380",
        "GCS_BUCKET": "test-env-bucket",
        "AUTH_TOKEN": "test-env-token"
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    return env_vars