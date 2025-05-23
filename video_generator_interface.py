"""
Video Generator Interface

This module defines the abstract interface for video generation backends,
enabling seamless switching between local models (like Wan2.1) and remote
APIs (like Runway ML and Google Veo 3).
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import logging

class VideoGeneratorInterface(ABC):
    """Abstract interface for video generation backends"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the video generator with configuration
        
        Args:
            config: Configuration dictionary containing backend-specific settings
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate_video(self, 
                      prompt: str, 
                      input_image_path: str,
                      output_path: str,
                      duration: float = 5.0,
                      **kwargs) -> str:
        """
        Generate a video segment
        
        Args:
            prompt: Text prompt describing the desired video
            input_image_path: Path to the input/reference image
            output_path: Path where the generated video should be saved
            duration: Desired duration of the video in seconds
            **kwargs: Additional backend-specific parameters
            
        Returns:
            Path to the generated video file
            
        Raises:
            VideoGenerationError: If video generation fails
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return backend capabilities and limits
        
        Returns:
            Dictionary containing:
            - max_duration: Maximum video duration in seconds
            - supported_resolutions: List of supported resolutions
            - supports_image_to_video: Whether backend supports image-to-video
            - supports_text_to_video: Whether backend supports text-to-video
            - requires_gpu: Whether local GPU is required
            - api_based: Whether this is an API-based backend
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, duration: float, resolution: str = "1280x720") -> float:
        """
        Estimate cost for video generation
        
        Args:
            duration: Video duration in seconds
            resolution: Video resolution (e.g., "1280x720")
            
        Returns:
            Estimated cost in USD (0.0 for local models)
        """
        pass
    
    @abstractmethod
    def validate_inputs(self, 
                       prompt: str, 
                       input_image_path: str,
                       duration: float) -> List[str]:
        """
        Validate inputs before generation
        
        Args:
            prompt: Text prompt
            input_image_path: Path to input image
            duration: Requested duration
            
        Returns:
            List of validation errors (empty if valid)
        """
        pass
    
    def is_available(self) -> bool:
        """
        Check if the backend is available and properly configured
        
        Returns:
            True if backend is ready to use, False otherwise
        """
        try:
            # Base implementation - can be overridden by subclasses
            capabilities = self.get_capabilities()
            return capabilities is not None
        except Exception as e:
            self.logger.error(f"Backend availability check failed: {e}")
            return False
    
    def get_backend_name(self) -> str:
        """
        Get the name of this backend
        
        Returns:
            Backend name for logging and display
        """
        return self.__class__.__name__.replace("Generator", "")


class VideoGenerationError(Exception):
    """Base exception for video generation errors"""
    pass


class APIError(VideoGenerationError):
    """Exception for API-related errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class GenerationTimeoutError(VideoGenerationError):
    """Exception for generation timeout"""
    pass


class InvalidInputError(VideoGenerationError):
    """Exception for invalid input parameters"""
    pass


class QuotaExceededError(VideoGenerationError):
    """Exception for quota/rate limit errors"""
    pass
