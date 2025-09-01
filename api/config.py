"""
Configuration management for the API server.

This module provides configuration loading and validation that reuses
the existing pipeline configuration system while adding API-specific settings.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class APIServerConfig(BaseModel):
    """API server configuration"""
    host: str = Field(default="0.0.0.0", description="Host to bind the server to")
    port: int = Field(default=8000, description="HTTP port")
    quic_port: int = Field(default=8443, description="QUIC/HTTP3 port")
    workers: int = Field(default=2, description="Number of worker processes")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    
    @field_validator('port', 'quic_port')
    @classmethod
    def validate_ports(cls, v):
        if not (1 <= v <= 65535):
            raise ValueError('Port must be between 1 and 65535')
        return v


class RedisConfig(BaseModel):
    """Redis configuration for job queuing"""
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    
    @field_validator('port')
    @classmethod
    def validate_port(cls, v):
        if not (1 <= v <= 65535):
            raise ValueError('Port must be between 1 and 65535')
        return v


class GCSConfig(BaseModel):
    """Google Cloud Storage configuration"""
    bucket: str = Field(..., description="GCS bucket name for artifacts")
    prefix: str = Field(default="ttv-api", description="Path prefix for artifacts")
    credentials_path: Optional[str] = Field(default=None, description="Path to GCS credentials JSON")
    signed_url_expiration: int = Field(default=3600, description="Signed URL expiration in seconds")
    
    @field_validator('signed_url_expiration')
    @classmethod
    def validate_expiration(cls, v):
        if v <= 0:
            raise ValueError('Signed URL expiration must be positive')
        return v


class SecurityConfig(BaseModel):
    """Security configuration"""
    auth_token: Optional[str] = Field(default=None, description="Bearer token for authentication")
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per minute per client")
    
    @field_validator('rate_limit_per_minute')
    @classmethod
    def validate_rate_limit(cls, v):
        if v <= 0:
            raise ValueError('Rate limit must be positive')
        return v


class APIConfig(BaseModel):
    """Complete API configuration combining server, Redis, GCS, and security settings"""
    server: APIServerConfig = Field(default_factory=APIServerConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    gcs: GCSConfig
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # Pipeline configuration will be merged from existing config
    pipeline_config: Dict[str, Any] = Field(default_factory=dict)


def load_pipeline_config(config_path: str = "pipeline_config.yaml") -> Dict[str, Any]:
    """
    Load the existing pipeline configuration.
    
    Args:
        config_path: Path to the pipeline configuration file
        
    Returns:
        Dictionary containing the pipeline configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded pipeline configuration from {config_path}")
        return config or {}
        
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in pipeline config: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load pipeline config: {e}")
        raise


def load_api_config(
    api_config_path: Optional[str] = None,
    pipeline_config_path: str = "pipeline_config.yaml"
) -> APIConfig:
    """
    Load complete API configuration by combining API-specific settings
    with the existing pipeline configuration.
    
    Args:
        api_config_path: Path to API-specific configuration file (optional)
        pipeline_config_path: Path to pipeline configuration file
        
    Returns:
        Complete APIConfig instance
        
    Raises:
        FileNotFoundError: If required config files don't exist
        ValidationError: If configuration is invalid
    """
    # Load pipeline configuration (required)
    pipeline_config = load_pipeline_config(pipeline_config_path)
    
    # Load API-specific configuration (optional)
    api_settings = {}
    if api_config_path and Path(api_config_path).exists():
        try:
            with open(api_config_path, 'r') as f:
                api_settings = yaml.safe_load(f) or {}
            logger.info(f"Loaded API configuration from {api_config_path}")
        except Exception as e:
            logger.warning(f"Failed to load API config from {api_config_path}: {e}")
    
    # Extract GCS configuration from pipeline config if available
    gcs_config = {}
    if 'google_veo' in pipeline_config:
        veo_config = pipeline_config['google_veo']
        gcs_config = {
            'bucket': veo_config.get('output_bucket', 'ttv-api-artifacts'),
            'credentials_path': veo_config.get('credentials_path')
        }
    
    # Override with API-specific GCS settings if provided
    if 'gcs' in api_settings:
        gcs_config.update(api_settings['gcs'])
    
    # Ensure we have required GCS configuration
    if not gcs_config.get('bucket'):
        # Use a default bucket name if none specified
        gcs_config['bucket'] = 'ttv-api-artifacts'
        logger.warning("No GCS bucket specified, using default: ttv-api-artifacts")
    
    # Build complete configuration
    config_data = {
        'server': api_settings.get('server', {}),
        'redis': api_settings.get('redis', {}),
        'gcs': gcs_config,
        'security': api_settings.get('security', {}),
        'pipeline_config': pipeline_config
    }
    
    try:
        config = APIConfig(**config_data)
        logger.info("API configuration loaded and validated successfully")
        return config
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def get_config_from_env() -> APIConfig:
    """
    Load configuration with environment variable overrides.
    
    Environment variables:
    - API_CONFIG_PATH: Path to API configuration file
    - PIPELINE_CONFIG_PATH: Path to pipeline configuration file
    - API_HOST: Server host
    - API_PORT: Server port
    - API_QUIC_PORT: QUIC port
    - REDIS_HOST: Redis host
    - REDIS_PORT: Redis port
    - REDIS_PASSWORD: Redis password
    - GCS_BUCKET: GCS bucket name
    - GCS_CREDENTIALS_PATH: Path to GCS credentials
    - AUTH_TOKEN: Bearer token for authentication
    
    Returns:
        Complete APIConfig instance with environment overrides
    """
    # Load base configuration
    api_config_path = os.getenv('API_CONFIG_PATH')
    pipeline_config_path = os.getenv('PIPELINE_CONFIG_PATH', 'pipeline_config.yaml')
    
    config = load_api_config(api_config_path, pipeline_config_path)
    
    # Apply environment variable overrides
    if os.getenv('API_HOST'):
        config.server.host = os.getenv('API_HOST')
    
    if os.getenv('API_PORT'):
        config.server.port = int(os.getenv('API_PORT'))
    
    if os.getenv('API_QUIC_PORT'):
        config.server.quic_port = int(os.getenv('API_QUIC_PORT'))
    
    if os.getenv('REDIS_HOST'):
        config.redis.host = os.getenv('REDIS_HOST')
    
    if os.getenv('REDIS_PORT'):
        config.redis.port = int(os.getenv('REDIS_PORT'))
    
    if os.getenv('REDIS_PASSWORD'):
        config.redis.password = os.getenv('REDIS_PASSWORD')
    
    if os.getenv('GCS_BUCKET'):
        config.gcs.bucket = os.getenv('GCS_BUCKET')
    
    if os.getenv('GCS_CREDENTIALS_PATH'):
        config.gcs.credentials_path = os.getenv('GCS_CREDENTIALS_PATH')
    
    if os.getenv('AUTH_TOKEN'):
        config.security.auth_token = os.getenv('AUTH_TOKEN')
    
    logger.info("Configuration loaded with environment variable overrides")
    return config