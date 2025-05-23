"""
Base utilities and helper functions for video generators
"""

import os
import time
import logging
from typing import Dict, Any, Optional, Callable
import requests
from PIL import Image

class RetryHandler:
    """Handle retries with exponential backoff"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def retry_with_backoff(self, func: Callable, *args, **kwargs):
        """
        Retry a function with exponential backoff
        
        Args:
            func: Function to retry
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"All {self.max_retries} attempts failed. Last error: {e}")
        
        raise last_exception


class ImageValidator:
    """Validate and prepare images for video generation"""
    
    @staticmethod
    def validate_image(image_path: str, max_size_mb: float = 10.0) -> Dict[str, Any]:
        """
        Validate an image file
        
        Args:
            image_path: Path to the image file
            max_size_mb: Maximum file size in MB
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        info = {}
        
        # Check if file exists
        if not os.path.exists(image_path):
            errors.append(f"Image file not found: {image_path}")
            return {"valid": False, "errors": errors, "info": info}
        
        # Check file size
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        info["file_size_mb"] = file_size_mb
        
        if file_size_mb > max_size_mb:
            errors.append(f"Image file too large: {file_size_mb:.2f}MB (max: {max_size_mb}MB)")
        
        # Check image format and dimensions
        try:
            with Image.open(image_path) as img:
                info["format"] = img.format
                info["dimensions"] = img.size
                info["mode"] = img.mode
                
                # Validate format
                supported_formats = ["PNG", "JPEG", "JPG", "WEBP"]
                if img.format not in supported_formats:
                    errors.append(f"Unsupported image format: {img.format}. Supported: {supported_formats}")
                
                # Check dimensions
                width, height = img.size
                if width < 64 or height < 64:
                    errors.append(f"Image too small: {width}x{height} (minimum: 64x64)")
                if width > 4096 or height > 4096:
                    errors.append(f"Image too large: {width}x{height} (maximum: 4096x4096)")
                    
        except Exception as e:
            errors.append(f"Failed to read image: {e}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "info": info
        }
    
    @staticmethod
    def prepare_image_for_api(image_path: str, 
                             target_size: Optional[tuple] = None,
                             max_size_mb: float = 5.0) -> str:
        """
        Prepare an image for API upload
        
        Args:
            image_path: Path to the original image
            target_size: Target dimensions (width, height) or None
            max_size_mb: Maximum file size in MB
            
        Returns:
            Path to the prepared image
        """
        import tempfile
        
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode not in ["RGB", "RGBA"]:
                img = img.convert("RGB")
            
            # Resize if target size specified
            if target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Save to temporary file with optimization
            temp_path = os.path.join(tempfile.gettempdir(), f"prepared_{os.path.basename(image_path)}")
            
            # Try different quality levels to meet size requirement
            for quality in [95, 90, 85, 80, 70]:
                img.save(temp_path, "JPEG", quality=quality, optimize=True)
                if os.path.getsize(temp_path) / (1024 * 1024) <= max_size_mb:
                    break
            
            return temp_path


class ProgressMonitor:
    """Monitor and report progress for long-running operations"""
    
    def __init__(self, total_steps: int = 100, callback: Optional[Callable] = None):
        self.total_steps = total_steps
        self.current_step = 0
        self.callback = callback
        self.start_time = time.time()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def update(self, step: int, message: str = ""):
        """Update progress"""
        self.current_step = step
        progress = (step / self.total_steps) * 100
        elapsed = time.time() - self.start_time
        
        if self.callback:
            self.callback(progress, message, elapsed)
        else:
            self.logger.info(f"Progress: {progress:.1f}% - {message} (elapsed: {elapsed:.1f}s)")
    
    def estimate_remaining_time(self) -> float:
        """Estimate remaining time based on current progress"""
        if self.current_step == 0:
            return 0
        
        elapsed = time.time() - self.start_time
        rate = self.current_step / elapsed
        remaining_steps = self.total_steps - self.current_step
        
        return remaining_steps / rate if rate > 0 else 0


def download_file(url: str, output_path: str, chunk_size: int = 8192) -> str:
    """
    Download a file from a URL with progress reporting
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        chunk_size: Download chunk size
        
    Returns:
        Path to the downloaded file
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\rDownloading: {progress:.1f}%", end='', flush=True)
    
    print()  # New line after progress
    return output_path


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"
