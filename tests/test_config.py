"""
Tests for configuration loading and validation.
"""

import os
import pytest
from pathlib import Path
from pydantic import ValidationError

from api.config import (
    load_pipeline_config, load_api_config, get_config_from_env,
    APIServerConfig, RedisConfig, GCSConfig, SecurityConfig, APIConfig
)


class TestPipelineConfigLoading:
    """Test pipeline configuration loading"""
    
    def test_load_existing_pipeline_config(self):
        """Test loading the actual pipeline configuration file"""
        config = load_pipeline_config("pipeline_config.yaml")
        
        assert isinstance(config, dict)
        assert "prompt" in config
        assert "default_backend" in config
        assert "size" in config
    
    def test_load_pipeline_config_from_temp_file(self, temp_config_files):
        """Test loading pipeline config from temporary file"""
        config = load_pipeline_config(temp_config_files["pipeline_config_path"])
        
        assert config["task"] == "flf2v-14B"
        assert config["size"] == "1280*720"
        assert config["default_backend"] == "veo3"
        assert config["total_gpus"] == 8
    
    def test_load_nonexistent_pipeline_config(self):
        """Test loading non-existent pipeline config raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_pipeline_config("nonexistent_config.yaml")


class TestAPIConfigValidation:
    """Test API configuration validation with Pydantic models"""
    
    def test_api_server_config_validation(self):
        """Test APIServerConfig validation"""
        # Valid config
        config = APIServerConfig(
            host="localhost",
            port=8000,
            quic_port=8443,
            workers=4
        )
        assert config.host == "localhost"
        assert config.port == 8000
        assert config.quic_port == 8443
        assert config.workers == 4
    
    def test_api_server_config_invalid_port(self):
        """Test APIServerConfig with invalid port"""
        with pytest.raises(ValidationError):
            APIServerConfig(port=70000)  # Port too high
        
        with pytest.raises(ValidationError):
            APIServerConfig(port=0)  # Port too low
    
    def test_redis_config_validation(self):
        """Test RedisConfig validation"""
        config = RedisConfig(
            host="redis-server",
            port=6379,
            db=1,
            password="secret"
        )
        assert config.host == "redis-server"
        assert config.port == 6379
        assert config.db == 1
        assert config.password == "secret"
    
    def test_gcs_config_validation(self):
        """Test GCSConfig validation"""
        config = GCSConfig(
            bucket="test-bucket",
            prefix="api-test",
            credentials_path="/path/to/creds.json",
            signed_url_expiration=7200
        )
        assert config.bucket == "test-bucket"
        assert config.prefix == "api-test"
        assert config.signed_url_expiration == 7200
    
    def test_gcs_config_invalid_expiration(self):
        """Test GCSConfig with invalid expiration"""
        with pytest.raises(ValidationError):
            GCSConfig(
                bucket="test-bucket",
                signed_url_expiration=-1  # Negative expiration
            )
    
    def test_security_config_validation(self):
        """Test SecurityConfig validation"""
        config = SecurityConfig(
            auth_token="bearer-token",
            rate_limit_per_minute=120
        )
        assert config.auth_token == "bearer-token"
        assert config.rate_limit_per_minute == 120


class TestAPIConfigLoading:
    """Test complete API configuration loading"""
    
    def test_load_api_config_with_files(self, temp_config_files):
        """Test loading API config with both pipeline and API config files"""
        config = load_api_config(
            api_config_path=temp_config_files["api_config_path"],
            pipeline_config_path=temp_config_files["pipeline_config_path"]
        )
        
        assert isinstance(config, APIConfig)
        assert config.server.port == 8000
        assert config.redis.host == "localhost"
        assert config.gcs.bucket == "test-api-bucket"
        assert config.security.auth_token == "test-token"
        assert config.pipeline_config["default_backend"] == "veo3"
    
    def test_load_api_config_pipeline_only(self, temp_config_files):
        """Test loading API config with only pipeline config"""
        config = load_api_config(
            api_config_path=None,
            pipeline_config_path=temp_config_files["pipeline_config_path"]
        )
        
        assert isinstance(config, APIConfig)
        # Should use defaults for API settings
        assert config.server.port == 8000
        assert config.redis.host == "localhost"
        # Should extract GCS config from pipeline
        assert config.gcs.bucket == "test-bucket"  # From google_veo.output_bucket
        assert config.pipeline_config["default_backend"] == "veo3"
    
    def test_load_api_config_gcs_fallback(self, sample_pipeline_config, temp_config_files):
        """Test GCS config fallback when no bucket specified"""
        # Remove GCS bucket from pipeline config
        pipeline_config = sample_pipeline_config.copy()
        if "google_veo" in pipeline_config:
            del pipeline_config["google_veo"]["output_bucket"]
        
        # Write modified config
        pipeline_path = Path(temp_config_files["temp_dir"]) / "pipeline_no_bucket.yaml"
        import yaml
        with open(pipeline_path, 'w') as f:
            yaml.dump(pipeline_config, f)
        
        config = load_api_config(
            api_config_path=None,
            pipeline_config_path=str(pipeline_path)
        )
        
        # Should use default bucket name
        assert config.gcs.bucket == "ttv-api-artifacts"


class TestEnvironmentVariableOverrides:
    """Test environment variable configuration overrides"""
    
    def test_env_var_overrides(self, temp_config_files, mock_env_vars, monkeypatch):
        """Test that environment variables override configuration"""
        # Set environment variables for config paths
        monkeypatch.setenv("API_CONFIG_PATH", temp_config_files["api_config_path"])
        monkeypatch.setenv("PIPELINE_CONFIG_PATH", temp_config_files["pipeline_config_path"])
        
        config = get_config_from_env()
        
        # Check that env vars override config values
        assert config.server.host == "test-host"  # From API_HOST
        assert config.server.port == 9000  # From API_PORT
        assert config.redis.host == "test-redis"  # From REDIS_HOST
        assert config.redis.port == 6380  # From REDIS_PORT
        assert config.gcs.bucket == "test-env-bucket"  # From GCS_BUCKET
        assert config.security.auth_token == "test-env-token"  # From AUTH_TOKEN
    
    def test_env_var_config_paths(self, temp_config_files, monkeypatch):
        """Test environment variables for config file paths"""
        monkeypatch.setenv("API_CONFIG_PATH", temp_config_files["api_config_path"])
        monkeypatch.setenv("PIPELINE_CONFIG_PATH", temp_config_files["pipeline_config_path"])
        
        config = get_config_from_env()
        
        assert isinstance(config, APIConfig)
        assert config.pipeline_config["default_backend"] == "veo3"
        assert config.gcs.bucket == "test-api-bucket"