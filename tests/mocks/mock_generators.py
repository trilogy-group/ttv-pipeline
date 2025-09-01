"""
Mock video generators for integration testing.
"""

import os
import time
from typing import Dict, Any, List, Optional, Callable
from unittest.mock import Mock

from video_generator_interface import VideoGeneratorInterface, VideoGenerationError


class MockVideoGenerator(VideoGeneratorInterface):
    """Mock video generator that simulates real generator behavior"""
    
    def __init__(self, config: Dict[str, Any], 
                 generation_time: float = 2.0, 
                 should_fail: bool = False,
                 failure_at_progress: Optional[int] = None):
        super().__init__(config)
        self.generation_time = generation_time
        self.should_fail = should_fail
        self.failure_at_progress = failure_at_progress
        self.progress_callback = None
        self.generation_count = 0
        
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "max_duration": 10.0,
            "supported_resolutions": ["1920x1080", "1280x720", "720x1280"],
            "supports_image_to_video": True,
            "supports_text_to_video": True,
            "requires_gpu": False,
            "api_based": True,
            "models": {
                "mock-model-v1": True
            },
            "features": {
                "motion_control": True,
                "style_transfer": False,
                "camera_control": True
            }
        }
    
    def estimate_cost(self, duration: float, resolution: str = "1920x1080") -> float:
        base_cost = duration * 0.10  # $0.10 per second
        
        # Resolution multiplier
        if "1920x1080" in resolution:
            multiplier = 1.2
        elif "1280x720" in resolution:
            multiplier = 1.0
        else:
            multiplier = 0.8
            
        return base_cost * multiplier
    
    def validate_inputs(self, prompt: str, input_image_path: str = None, duration: float = 5.0) -> List[str]:
        errors = []
        
        if not prompt or len(prompt.strip()) == 0:
            errors.append("Prompt is required")
        elif len(prompt) > 1000:
            errors.append("Prompt too long (max 1000 characters)")
            
        if input_image_path and not os.path.exists(input_image_path):
            errors.append("Input image not found")
            
        if duration <= 0:
            errors.append("Duration must be positive")
        elif duration > 10:
            errors.append("Duration exceeds maximum (10 seconds)")
            
        return errors
    
    def generate_video(self, prompt: str, input_image_path: str = None, output_path: str = None, 
                      duration: float = 5.0, **kwargs) -> str:
        """Mock video generation with realistic progress reporting"""
        self.generation_count += 1
        
        # Validate inputs first
        validation_errors = self.validate_inputs(prompt, input_image_path, duration)
        if validation_errors:
            raise VideoGenerationError(f"Validation failed: {'; '.join(validation_errors)}")
        
        # Ensure output directory exists
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        else:
            output_path = f"/tmp/mock_video_{int(time.time())}.mp4"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Simulate generation with progress reporting
        progress_steps = [
            (0, "Initializing generation..."),
            (10, "Processing prompt..."),
            (25, "Generating keyframes..."),
            (40, "Creating video segments..."),
            (60, "Applying motion..."),
            (80, "Finalizing video..."),
            (95, "Encoding output..."),
            (100, "Generation complete")
        ]
        
        step_duration = self.generation_time / len(progress_steps)
        
        for progress, message in progress_steps:
            if self.progress_callback:
                self.progress_callback(progress, message)
            
            # Check for failure at specific progress
            if self.failure_at_progress and progress >= self.failure_at_progress:
                raise VideoGenerationError(f"Mock failure at {progress}% progress")
            
            time.sleep(step_duration)
        
        # Check for general failure
        if self.should_fail:
            raise VideoGenerationError("Mock generation failure")
        
        # Create a mock video file with realistic content
        self._create_mock_video_file(output_path, duration)
        
        return output_path
    
    def _create_mock_video_file(self, output_path: str, duration: float):
        """Create a mock video file with proper MP4 structure"""
        # Calculate approximate file size (1MB per second as rough estimate)
        file_size = int(duration * 1024 * 1024)
        
        with open(output_path, 'wb') as f:
            # Write minimal MP4 header (ftyp box)
            f.write(b'\x00\x00\x00\x20ftypmp42\x00\x00\x00\x00mp42isom')
            
            # Write mock mdat box header
            mdat_size = file_size - 32  # Subtract header size
            f.write(b'\x00\x00\x00\x08mdat')
            
            # Write mock video data
            chunk_size = 8192
            written = 32  # Already wrote header
            
            while written < file_size:
                remaining = min(chunk_size, file_size - written)
                # Write pseudo-random data that looks like compressed video
                chunk = bytes([(i * 37 + written) % 256 for i in range(remaining)])
                f.write(chunk)
                written += remaining
    
    def is_available(self) -> bool:
        return True
    
    def set_progress_callback(self, callback: Callable[[int, str], None]):
        """Set progress callback for testing"""
        self.progress_callback = callback
    
    def get_generation_count(self) -> int:
        """Get number of videos generated (for testing)"""
        return self.generation_count


class MockWan21Generator(MockVideoGenerator):
    """Mock Wan2.1 local generator"""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.backend_name = "wan2.1"
    
    def get_capabilities(self) -> Dict[str, Any]:
        capabilities = super().get_capabilities()
        capabilities.update({
            "requires_gpu": True,
            "api_based": False,
            "local_processing": True,
            "models": {
                "wan2.1-i2v": True,
                "wan2.1-flf2v": True
            }
        })
        return capabilities


class MockRunwayGenerator(MockVideoGenerator):
    """Mock Runway ML generator"""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.backend_name = "runway"
        self.api_key = config.get("api_key", "mock-api-key")
    
    def get_capabilities(self) -> Dict[str, Any]:
        capabilities = super().get_capabilities()
        capabilities.update({
            "max_duration": 16.0,
            "models": {
                "gen-3-alpha": True,
                "gen-3-turbo": True
            },
            "features": {
                "motion_control": True,
                "camera_control": True,
                "style_transfer": True
            }
        })
        return capabilities


class MockVeo3Generator(MockVideoGenerator):
    """Mock Google Veo 3 generator"""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.backend_name = "veo3"
        self.project_id = config.get("project_id", "mock-project")
    
    def get_capabilities(self) -> Dict[str, Any]:
        capabilities = super().get_capabilities()
        capabilities.update({
            "max_duration": 5.0,
            "models": {
                "veo-3.0-generate-preview": True
            },
            "features": {
                "motion_control": True,
                "temporal_consistency": True
            }
        })
        return capabilities


class MockMinimaxGenerator(MockVideoGenerator):
    """Mock Minimax generator"""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.backend_name = "minimax"
        self.api_key = config.get("api_key", "mock-api-key")
    
    def get_capabilities(self) -> Dict[str, Any]:
        capabilities = super().get_capabilities()
        capabilities.update({
            "max_duration": 6.0,
            "models": {
                "I2V-01-Director": True
            }
        })
        return capabilities


class MockGeneratorFactory:
    """Mock factory for creating video generators"""
    
    GENERATOR_REGISTRY = {
        "wan2.1": MockWan21Generator,
        "runway": MockRunwayGenerator,
        "veo3": MockVeo3Generator,
        "minimax": MockMinimaxGenerator,
    }
    
    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        self.default_config = default_config or {}
        self.created_generators = []
    
    def create_video_generator(self, backend: str, config: Dict[str, Any], **kwargs) -> VideoGeneratorInterface:
        """Create a mock video generator"""
        backend = backend.lower()
        
        if backend not in self.GENERATOR_REGISTRY:
            available = ", ".join(self.GENERATOR_REGISTRY.keys())
            raise VideoGenerationError(
                f"Unknown video generation backend: '{backend}'. "
                f"Available backends: {available}"
            )
        
        generator_class = self.GENERATOR_REGISTRY[backend]
        
        # Merge with default config
        merged_config = {**self.default_config, **config}
        
        # Create generator with any additional kwargs
        generator = generator_class(merged_config, **kwargs)
        self.created_generators.append(generator)
        
        return generator
    
    def get_created_generators(self) -> List[VideoGeneratorInterface]:
        """Get list of created generators (for testing)"""
        return self.created_generators.copy()
    
    def reset(self):
        """Reset factory state"""
        self.created_generators.clear()


# Utility functions for testing

def create_mock_generator_with_failure(backend: str, failure_type: str = "immediate") -> MockVideoGenerator:
    """Create a mock generator configured to fail in specific ways"""
    config = {}
    
    if failure_type == "immediate":
        return MockVideoGenerator(config, should_fail=True)
    elif failure_type == "at_50_percent":
        return MockVideoGenerator(config, failure_at_progress=50)
    elif failure_type == "timeout":
        return MockVideoGenerator(config, generation_time=300)  # Very long generation
    else:
        return MockVideoGenerator(config)


def create_mock_generator_with_progress(backend: str, generation_time: float = 1.0) -> MockVideoGenerator:
    """Create a mock generator with specific generation time"""
    config = {}
    return MockVideoGenerator(config, generation_time=generation_time)