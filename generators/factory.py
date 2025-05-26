"""
Factory for creating video generator instances
"""

import logging
from typing import Dict, Any, Optional
from video_generator_interface import VideoGeneratorInterface, VideoGenerationError

# Import local generators
from generators.local.wan21_generator import Wan21Generator

# Import remote generators
from generators.remote.runway_generator import RunwayMLGenerator
from generators.remote.veo3_generator import Veo3Generator
from generators.remote.minimax_generator import MinimaxGenerator

logger = logging.getLogger(__name__)

# Registry of available generators
GENERATOR_REGISTRY = {
    "wan2.1": Wan21Generator,
    "runway": RunwayMLGenerator,
    "veo3": Veo3Generator,
    "minimax": MinimaxGenerator,
}


def create_video_generator(backend: str, config: Dict[str, Any]) -> VideoGeneratorInterface:
    """
    Create a video generator instance based on the specified backend
    
    Args:
        backend: Name of the backend to use (e.g., "wan2.1", "runway", "veo3")
        config: Full configuration dictionary containing backend-specific settings
        
    Returns:
        Instance of VideoGeneratorInterface
        
    Raises:
        VideoGenerationError: If backend is not supported or configuration is invalid
    """
    backend = backend.lower()
    
    if backend not in GENERATOR_REGISTRY:
        available = ", ".join(GENERATOR_REGISTRY.keys())
        raise VideoGenerationError(
            f"Unknown video generation backend: '{backend}'. "
            f"Available backends: {available}"
        )
    
    generator_class = GENERATOR_REGISTRY[backend]
    
    try:
        # Extract backend-specific configuration
        backend_config = {}
        
        if backend == "wan2.1":
            # Local Wan2.1 configuration
            backend_config = {
                "wan2_dir": config.get("wan2_dir", "./Wan2.1"),
                "i2v_model_dir": config.get("i2v_model_dir"),
                "flf2v_model_dir": config.get("flf2v_model_dir"),
                "total_gpus": config.get("total_gpus", 1),
                "gpu_count": config.get("gpu_count", config.get("total_gpus", 1)),
                "size": config.get("size", "1280*720"),
                "guide_scale": config.get("guide_scale", 5.0),
                "sample_steps": config.get("sample_steps", 40),
                "sample_shift": config.get("sample_shift", 5.0),
                "frame_num": config.get("frame_num", 81),
                "chaining_max_retries": config.get("chaining_max_retries", 3),
                "chaining_use_fsdp_flags": config.get("chaining_use_fsdp_flags", True),
            }
            
        elif backend == "runway":
            # Runway ML configuration
            runway_config = config.get("runway_ml", {})
            remote_settings = config.get("remote_api_settings", {})
            
            backend_config = {
                "api_key": runway_config.get("api_key"),
                "model_version": runway_config.get("model_version", "gen-3-alpha"),
                "max_duration": runway_config.get("max_duration", 10),
                "motion_amount": runway_config.get("motion_amount", "auto"),
                "max_retries": remote_settings.get("max_retries", 3),
                "polling_interval": remote_settings.get("polling_interval", 10),
                "timeout": remote_settings.get("timeout", 600),
            }
            
            if not backend_config["api_key"]:
                raise VideoGenerationError("Runway ML API key is required but not provided")
                
        elif backend == "veo3":
            # Google Veo 3 configuration
            veo_config = config.get("google_veo", {})
            remote_settings = config.get("remote_api_settings", {})
            
            backend_config = {
                "project_id": veo_config.get("project_id"),
                "credentials_path": veo_config.get("credentials_path", "credentials.json"),
                "region": veo_config.get("region", "global"),
                "output_bucket": veo_config.get("output_bucket"),
                "max_retries": remote_settings.get("max_retries", 3),
                "polling_interval": remote_settings.get("polling_interval", 15),
                "timeout": remote_settings.get("timeout", 600),
            }
            
            if not backend_config["project_id"]:
                raise VideoGenerationError(
                    "Google Veo 3 requires a project ID"
                )
        
        elif backend == "minimax":
            # Minimax configuration
            minimax_config = config.get("minimax", {})
            remote_settings = config.get("remote_api_settings", {})
            
            backend_config = {
                "api_key": minimax_config.get("api_key"),
                "model": minimax_config.get("model", "I2V-01-Director"),
                "max_duration": minimax_config.get("max_duration", 6),
                "base_url": minimax_config.get("base_url", "https://api.minimaxi.chat/v1"),
                "max_retries": remote_settings.get("max_retries", 3),
                "polling_interval": remote_settings.get("polling_interval", 5),
                "timeout": remote_settings.get("timeout", 300),
            }
            
            if not backend_config["api_key"]:
                raise VideoGenerationError("Minimax API key is required but not provided")
        
        # Create the generator instance
        generator = generator_class(backend_config)
        
        # Verify the generator is available
        if not generator.is_available():
            raise VideoGenerationError(
                f"Backend '{backend}' is not available. "
                "Please check configuration and dependencies."
            )
        
        logger.info(f"Created {backend} video generator")
        return generator
        
    except Exception as e:
        logger.error(f"Failed to create {backend} generator: {e}")
        raise VideoGenerationError(f"Failed to initialize {backend} backend: {e}")


def get_fallback_generator(primary_backend: str, 
                           config: Dict[str, Any],
                           attempted_backends: Optional[set] = None) -> Optional[VideoGeneratorInterface]:
    """
    Get a fallback generator if the primary backend fails
    
    Args:
        primary_backend: The primary backend that failed
        config: Full configuration dictionary
        attempted_backends: Set of backends already attempted
        
    Returns:
        Fallback generator instance or None if no fallback available
    """
    if attempted_backends is None:
        attempted_backends = set()
    
    attempted_backends.add(primary_backend)
    
    # Check if a specific fallback is configured
    remote_settings = config.get("remote_api_settings", {})
    fallback_backend = remote_settings.get("fallback_backend")
    
    # Only use fallback if explicitly configured
    if fallback_backend and fallback_backend not in attempted_backends:
        try:
            logger.info(f"Attempting to use fallback backend: {fallback_backend}")
            return create_video_generator(fallback_backend, config)
        except Exception as e:
            logger.error(f"Fallback backend {fallback_backend} also failed: {e}")
    
    # If no fallback is configured or fallback failed, don't try other backends
    logger.info("No fallback backend configured or available")
    return None
