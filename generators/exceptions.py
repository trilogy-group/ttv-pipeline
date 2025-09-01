"""Custom exceptions for video generation backends."""


class VideoGenerationError(Exception):
    """Base exception for video generation errors."""
    pass


class InvalidInputError(VideoGenerationError):
    """Raised when input parameters are invalid."""
    pass


class GenerationError(VideoGenerationError):
    """Raised when generation fails."""
    pass


class ConfigurationError(VideoGenerationError):
    """Raised when configuration is invalid."""
    pass


class AuthenticationError(VideoGenerationError):
    """Raised when authentication fails."""
    pass


class RateLimitError(VideoGenerationError):
    """Raised when rate limit is exceeded."""
    pass


class ServiceUnavailableError(VideoGenerationError):
    """Raised when service is unavailable."""
    pass
