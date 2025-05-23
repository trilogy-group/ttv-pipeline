"""
Runway ML API video generator implementation
"""

import os
import time
import json
import base64
import logging
from typing import Dict, Any, List, Optional
from runwayml import RunwayML
from video_generator_interface import (
    VideoGeneratorInterface, 
    VideoGenerationError,
    APIError,
    GenerationTimeoutError,
    InvalidInputError,
    QuotaExceededError
)
from generators.base import (
    ImageValidator, 
    RetryHandler, 
    ProgressMonitor,
    download_file,
    format_duration
)

class RunwayMLGenerator(VideoGeneratorInterface):
    """Remote video generator using Runway ML API"""
    
    # Pricing per second (approximate values)
    PRICING = {
        "gen-3-alpha": 0.05,      # $0.05 per second
        "gen-3-alpha-turbo": 0.01, # $0.01 per second
        "gen-4": 0.08,           # $0.08 per second
        "gen4_turbo": 0.03       # $0.03 per second
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.model_version = config.get("model_version", "gen4_turbo")
        self.max_duration = config.get("max_duration", 5)
        self.default_ratio = config.get("default_ratio", "1280:720")  # 16:9 landscape
        self.seed = config.get("seed")  # Optional seed parameter
        self.max_retries = config.get("max_retries", 3)
        self.polling_interval = config.get("polling_interval", 10)
        self.timeout = config.get("timeout", 600)
        
        # Set environment variable for Runway SDK (only if config value is defined)
        if self.api_key:
            os.environ["RUNWAYML_API_SECRET"] = self.api_key
        
        # Check if API key is available (either from config or environment)
        if not self.api_key and not os.environ.get("RUNWAYML_API_SECRET"):
            raise VideoGenerationError("Runway ML API key is required (either in config or RUNWAYML_API_SECRET env var)")
        
        # Initialize Runway client (SDK will read from environment variable)
        try:
            self.client = RunwayML()
        except Exception as e:
            raise VideoGenerationError(f"Failed to initialize Runway client: {e}")
            
        # Initialize retry handler
        self.retry_handler = RetryHandler(max_retries=self.max_retries)
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Return Runway ML capabilities"""
        return {
            "max_duration": self.max_duration,
            "supported_resolutions": [
                "1280:720",   # 16:9 landscape
                "720:1280",   # 9:16 portrait 
                "1104:832",   # 4:3 landscape
                "832:1104",   # 3:4 portrait
                "960:960",    # 1:1 square
                "1584:672"    # 21:9 ultrawide
            ],
            "supports_image_to_video": True,
            "supports_text_to_video": False,  # Only image-to-video supported
            "requires_gpu": False,  # API-based
            "api_based": True,
            "models": {
                "gen4_turbo": True,  # Latest model
                "gen3a_turbo": True  # Fast/cheaper model
            },
            "features": {
                "seed_control": True,  # Reproducibility via seed
                "prompt_text": True,   # Text prompts supported
                "duration_control": True  # 5-10 second videos
            }
        }
    
    def estimate_cost(self, duration: float, resolution: str = "1280:720") -> float:
        """Estimate cost for video generation"""
        # Get price per second for the model
        price_per_second = self.PRICING.get(self.model_version, 0.05)
        
        # Resolution multiplier (higher res costs more)
        resolution_multipliers = {
            "960:960": 1.0,      # 1:1 square
            "1280:720": 1.0,     # 16:9 landscape
            "720:1280": 1.0,     # 9:16 portrait
            "1104:832": 1.2,     # 4:3 landscape
            "832:1104": 1.2,     # 3:4 portrait
            "1584:672": 1.5      # 21:9 ultrawide
        }
        multiplier = resolution_multipliers.get(resolution, 1.0)
        
        return duration * price_per_second * multiplier
    
    def validate_inputs(self, 
                       prompt: str, 
                       input_image_path: str,
                       duration: float) -> List[str]:
        """Validate inputs for Runway ML generation"""
        errors = []
        
        # Validate prompt
        if not prompt or len(prompt.strip()) == 0:
            errors.append("Prompt cannot be empty")
        elif len(prompt) > 500:
            errors.append("Prompt too long (max 500 characters for Runway ML)")
        
        # Validate image
        image_validation = ImageValidator.validate_image(input_image_path, max_size_mb=10.0)
        if not image_validation["valid"]:
            errors.extend(image_validation["errors"])
        
        # Validate duration
        if duration > self.max_duration:
            errors.append(f"Duration {duration}s exceeds maximum {self.max_duration}s")
        elif duration < 1:
            errors.append("Duration must be at least 1 second")
        
        return errors
    
    def generate_video(self, 
                      prompt: str, 
                      input_image_path: str,
                      output_path: str,
                      duration: float = 5.0,
                      **kwargs) -> str:
        """Generate video using Runway ML API"""
        # Validate inputs
        validation_errors = self.validate_inputs(prompt, input_image_path, duration)
        if validation_errors:
            raise InvalidInputError(f"Input validation failed: {'; '.join(validation_errors)}")
        
        # Log cost estimate
        estimated_cost = self.estimate_cost(duration)
        self.logger.info(f"Estimated cost: ${estimated_cost:.2f}")
        
        try:
            # Encode image to base64
            self.logger.info("Encoding image to base64...")
            with open(input_image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
            
            # Determine aspect ratio from kwargs or use default
            ratio = kwargs.get("aspect_ratio", self.default_ratio)
            
            # Create the image-to-video task
            self.logger.info(f"Creating video generation task with model {self.model_version}...")
            
            # Prepare task parameters
            task_params = {
                "model": self.model_version,
                "prompt_image": f"data:image/png;base64,{base64_image}",
                "prompt_text": prompt,
                "ratio": ratio,
                "duration": int(duration),
            }
            
            # Add seed if specified (either from config or kwargs)
            seed = kwargs.get("seed", self.seed)
            if seed is not None:
                task_params["seed"] = int(seed)
                self.logger.info(f"Using seed: {seed}")
            
            task = self.client.image_to_video.create(**task_params)
            
            task_id = task.id
            self.logger.info(f"Task created with ID: {task_id}")
            
            # Poll for completion with progress monitoring
            self.logger.info("Polling for task completion...")
            result = self._poll_task(task_id)
            
            # Handle the result
            if result.status == 'SUCCEEDED':
                if hasattr(result, 'output') and result.output:
                    video_url = result.output[0] if isinstance(result.output, list) else result.output
                    self.logger.info("Downloading generated video...")
                    return download_file(video_url, output_path)
                else:
                    raise VideoGenerationError("Task succeeded but no output URL found")
            else:
                error_msg = getattr(result, 'failure_reason', 'Unknown error')
                raise VideoGenerationError(f"Task failed: {error_msg}")
                
        except Exception as e:
            if isinstance(e, VideoGenerationError):
                raise
            raise VideoGenerationError(f"Runway ML generation failed: {e}")
    
    def _poll_task(self, task_id: str):
        """Poll for task completion with progress monitoring"""
        start_time = time.time()
        progress_monitor = ProgressMonitor(self.timeout)
        
        # Initial wait before polling
        time.sleep(10)
        
        while True:
            elapsed = time.time() - start_time
            
            # Check timeout
            if elapsed > self.timeout:
                raise GenerationTimeoutError(f"Task {task_id} timed out after {elapsed:.1f}s")
            
            try:
                # Retrieve task status
                task = self.client.tasks.retrieve(task_id)
                self.logger.info(f"Task {task_id} status: {task.status}")
                
                # Update progress
                progress_monitor.update(int(elapsed / self.timeout * 100))
                
                # Check if complete
                if task.status in ['SUCCEEDED', 'FAILED']:
                    return task
                
                # Wait before next poll
                time.sleep(self.polling_interval)
                
            except Exception as e:
                self.logger.warning(f"Error polling task {task_id}: {e}")
                time.sleep(self.polling_interval)
                continue
