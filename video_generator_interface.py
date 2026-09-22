"""
Video Generator Interface

This module defines the abstract interface for video generation backends,
enabling seamless switching between local models (like Wan2.1) and remote
APIs (like Runway ML and Google Veo 3).
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import logging

from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
import inspect
import os
from pathlib import Path
from tempfile import gettempdir


@dataclass(frozen=True)
class BillingObservation:
    raw_unit_name: str | None = None
    raw_units: str | None = None
    estimated_usd: float | None = None
    actual_usd: float | None = None
    currency: str = "USD"


@dataclass(frozen=True)
class GenerationOutput:
    path: str
    provider: str
    model: str | None
    model_version: str | None
    provider_request_id: str | None
    seed: int | None
    parameters: dict[str, object]
    prompt: str
    started_at: datetime
    finished_at: datetime
    billing: BillingObservation | None
    reference_paths: tuple[str, ...] | None = None
    reference_sources: tuple[tuple[str, str], ...] = ()

    def __fspath__(self):
        return self.path


# Details belong to this call, including on failure; no adapter last-call lookup.
_call_details: ContextVar[dict | None] = ContextVar("generation_call_details", default=None)
generation_observer: ContextVar[Any] = ContextVar("generation_observer", default=None)


def set_generation_details(**details):
    current = _call_details.get()
    if current is not None:
        current.update(details)
        if observer := generation_observer.get():
            observer(current)


def replace_generation_reference(source, prepared):
    """Retain the exact bytes submitted after provider-specific image preparation."""
    current = _call_details.get()
    if current is not None:
        current["reference_paths"] = tuple(
            os.fspath(prepared) if path == os.fspath(source) else path
            for path in current.get("reference_paths", ())
        )
        current["reference_sources"] += ((os.fspath(source), os.fspath(prepared)),)


def cleanup_prepared_reference(path):
    prepared = Path(path)
    if prepared.parent == Path(gettempdir()) and prepared.name.startswith("ttv-prepared-"):
        prepared.unlink(missing_ok=True)


def recorded_generation(provider):
    def decorate(function):
        default_duration = inspect.signature(function).parameters["duration"].default
        @wraps(function)
        def wrapped(self, prompt, input_image_path, output_path, duration=default_duration, **kwargs):
            started = datetime.now(timezone.utc)
            details = dict(provider=provider, model=None, model_version=None,
                           provider_request_id=None, seed=None, parameters={}, prompt=prompt,
                           billing=BillingObservation(),
                           reference_paths=(os.fspath(input_image_path),), reference_sources=())
            token = _call_details.set(details)
            try:
                set_generation_details()
                path = function(self, prompt, input_image_path, output_path, duration, **kwargs)
                output = GenerationOutput(path=os.fspath(path), started_at=started,
                                        finished_at=datetime.now(timezone.utc), **details)
                from api.generation_ledger import active_ledger
                if ledger := active_ledger.get():
                    ledger.record(output, None, [input_image_path, kwargs.get("last_frame_path")])
                return output
            except BaseException as error:
                error.generation_output = GenerationOutput(
                    path=output_path, started_at=started,
                    finished_at=datetime.now(timezone.utc), **details)
                from api.generation_ledger import active_ledger
                if ledger := active_ledger.get():
                    ledger.record(error.generation_output, error, [input_image_path, kwargs.get("last_frame_path")])
                raise
            finally:
                _call_details.reset(token)
        return wrapped
    return decorate


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
                      **kwargs) -> GenerationOutput:
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
    def estimate_cost(self, duration: float, resolution: str = "1280x720") -> float | None:
        """
        Estimate cost for video generation
        
        Args:
            duration: Video duration in seconds
            resolution: Video resolution (e.g., "1280x720")
            
        Returns:
            Estimated cost in USD (0.0 for local models), or None when unknown
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
    def __init__(self, message: str, status_code: Optional[int] = None,
                 response_body: Optional[str] = None, error_type: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
        self.error_type = error_type


class GenerationTimeoutError(VideoGenerationError):
    """Exception for generation timeout"""
    pass


class InvalidInputError(VideoGenerationError):
    """Exception for invalid input parameters"""
    pass


class QuotaExceededError(VideoGenerationError):
    """Exception for quota/rate limit errors"""
    pass


def recorded_keyframe(function):
    """Capture one image invocation; orchestration sets retries=0 before invoking it."""
    @wraps(function)
    def wrapped(prompt, output_path, model_name=None, **kwargs):
        started = datetime.now(timezone.utc)
        details = dict(provider='unknown', model=None, model_version=None,
                       provider_request_id=None, seed=None, parameters={}, prompt=prompt,
                       billing=BillingObservation(),
                       reference_paths=tuple(os.fspath(p) for p in
                           [kwargs.get('input_image_path'), kwargs.get('mask_path')] if p))
        token = _call_details.set(details)
        try:
            set_generation_details()
            path = function(prompt, output_path, model_name=model_name, **kwargs)
            output = GenerationOutput(path=os.fspath(path), started_at=started,
                                    finished_at=datetime.now(timezone.utc), **details)
            from api.generation_ledger import active_ledger
            if ledger := active_ledger.get():
                ledger.record(output, None, [kwargs.get("input_image_path")])
            return output
        except BaseException as error:
            error.generation_output = GenerationOutput(path=str(output_path), started_at=started,
                finished_at=datetime.now(timezone.utc), **details)
            from api.generation_ledger import active_ledger
            if ledger := active_ledger.get():
                ledger.record(error.generation_output, error, [kwargs.get("input_image_path")])
            raise
        finally:
            _call_details.reset(token)
    return wrapped
