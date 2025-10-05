"""
Higgsfield AI video generator implementation using DoP (Depth of Parallax) model

This module implements the VideoGeneratorInterface for Higgsfield's image-to-video API,
supporting the DoP-Turbo model for high-quality video generation.
"""

import os
import time
import json
import logging
import requests
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import timedelta

from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError

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

class HiggsfieldGenerator(VideoGeneratorInterface):
    """Remote video generator using Higgsfield AI API with DoP models"""
    
    # API base URL
    BASE_URL = "https://platform.higgsfield.ai/v1"
    
    # Pricing per generation (estimated)
    PRICING = {
        "dop-turbo": 0.10,   # ~$0.10 per generation
        "dop-preview": 0.15, # ~$0.15 per generation
        "dop-lite": 0.08,    # ~$0.08 per generation
    }
    
    # Supported models
    MODELS = ["dop-turbo", "dop-preview", "dop-lite"]
    
    # Job status values
    STATUS_QUEUED = "queued"
    STATUS_IN_PROGRESS = "in_progress"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    STATUS_NSFW = "nsfw"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.api_secret = config.get("api_secret")
        self.model = config.get("model", "dop-turbo")
        self.base_url = config.get("base_url", self.BASE_URL)
        self.max_retries = config.get("max_retries", 3)
        self.polling_interval = config.get("polling_interval", 10)  # Poll every 10 seconds
        self.timeout = config.get("timeout", 600)  # 10 minutes default
        
        # GCS configuration for uploading keyframe images
        self.gcs_bucket = config.get("gcs_bucket")
        self.gcs_credentials_path = config.get("gcs_credentials_path", "credentials.json")
        self.gcs_prefix = config.get("gcs_prefix", "higgsfield-inputs")
        
        # Validate configuration
        if not self.api_key:
            self.api_key = os.getenv("HIGGSFIELD_API_KEY")
            if not self.api_key:
                raise VideoGenerationError(
                    "Higgsfield API key not found in config or HIGGSFIELD_API_KEY environment variable"
                )
        
        if not self.api_secret:
            self.api_secret = os.getenv("HIGGSFIELD_API_SECRET")
            if not self.api_secret:
                raise VideoGenerationError(
                    "Higgsfield API secret not found in config or HIGGSFIELD_API_SECRET environment variable"
                )
        
        if self.model not in self.MODELS:
            raise VideoGenerationError(
                f"Invalid model '{self.model}'. Supported models: {', '.join(self.MODELS)}"
            )
        
        # Validate GCS configuration
        if not self.gcs_bucket:
            raise VideoGenerationError(
                "GCS bucket is required for Higgsfield (config key: gcs_bucket)"
            )
        
        self.headers = {
            'hf-api-key': self.api_key,
            'hf-secret': self.api_secret,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        # Initialize GCS client for uploading images
        self._init_gcs_client()
        
        self.logger.info(f"Initialized Higgsfield generator with model: {self.model}")
    
    def _init_gcs_client(self):
        """Initialize Google Cloud Storage client for uploading keyframe images"""
        try:
            # Try to use credentials from config first
            if self.gcs_credentials_path and Path(self.gcs_credentials_path).exists():
                self.logger.info(f"Using GCS credentials from: {self.gcs_credentials_path}")
                self.storage_client = storage.Client.from_service_account_json(
                    self.gcs_credentials_path
                )
            else:
                # Fall back to default credentials
                self.logger.info("Using default GCS credentials")
                self.storage_client = storage.Client()
            
            # Ensure bucket exists
            self._ensure_bucket_exists(self.gcs_bucket)
            
            self.logger.info(f"GCS client initialized for bucket: {self.gcs_bucket}")
            
        except DefaultCredentialsError as e:
            raise VideoGenerationError(
                f"GCS credentials not found. Please provide credentials via "
                f"gcs_credentials_path or set up application default credentials: {e}"
            )
        except Exception as e:
            raise VideoGenerationError(f"Failed to initialize GCS client: {e}")
    
    def _ensure_bucket_exists(self, bucket_name: str):
        """Ensure the GCS bucket exists, create if it doesn't"""
        try:
            bucket = self.storage_client.bucket(bucket_name)
            if not bucket.exists():
                self.logger.info(f"Creating GCS bucket: {bucket_name}")
                bucket = self.storage_client.create_bucket(bucket_name)
        except Exception as e:
            self.logger.warning(f"Error checking/creating bucket {bucket_name}: {e}")
    
    def _upload_image_to_gcs(self, image_path: str) -> str:
        """
        Upload image to GCS and return a signed URL
        
        Args:
            image_path: Path to the local image file
            
        Returns:
            Signed URL for the uploaded image
        """
        try:
            # Generate unique blob name
            timestamp = int(time.time())
            filename = os.path.basename(image_path)
            blob_name = f"{self.gcs_prefix}/{timestamp}_{filename}"
            
            # Upload to GCS
            bucket = self.storage_client.bucket(self.gcs_bucket)
            blob = bucket.blob(blob_name)
            
            self.logger.info(f"Uploading image to GCS: gs://{self.gcs_bucket}/{blob_name}")
            
            with open(image_path, "rb") as f:
                blob.upload_from_file(f, content_type="image/jpeg")
            
            # Generate signed URL (valid for 24 hours)
            signed_url = blob.generate_signed_url(
                expiration=timedelta(hours=24),
                method="GET",
                version="v4"
            )
            
            self.logger.info(f"Image uploaded successfully. Signed URL generated.")
            return signed_url
            
        except Exception as e:
            raise VideoGenerationError(f"Failed to upload image to GCS: {e}")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return Higgsfield backend capabilities"""
        return {
            "max_duration": 5.0,  # Higgsfield generates ~5 second clips
            "supported_resolutions": ["720x1280", "1280x720", "1024x1024"],
            "supported_formats": ["mp4"],
            "supports_image_to_video": True,
            "supports_text_to_video": False,
            "requires_gpu": False,
            "api_based": True,
            "models": self.MODELS,
            "supports_motion_presets": True,
            "supports_end_frame": True,
            "cost_per_generation": self.PRICING.get(self.model, 0.10)
        }
    
    def estimate_cost(self, duration: float, resolution: str = "1280x720") -> float:
        """
        Estimate cost for video generation
        
        Note: Higgsfield charges per generation, not per second
        """
        cost_per_gen = self.PRICING.get(self.model, 0.10)
        return cost_per_gen
    
    def validate_inputs(self, 
                       prompt: str, 
                       input_image_path: str,
                       duration: float) -> List[str]:
        """Validate inputs for Higgsfield generation"""
        errors = []
        
        # Validate prompt
        if not prompt or len(prompt.strip()) == 0:
            errors.append("Prompt cannot be empty")
        elif len(prompt) > 1000:
            errors.append("Prompt too long (max 1000 characters for Higgsfield)")
        
        # Validate image
        image_validation = ImageValidator.validate_image(input_image_path, max_size_mb=10.0)
        if not image_validation["valid"]:
            errors.extend(image_validation["errors"])
        
        # Validate GCS configuration
        if not self.gcs_bucket:
            errors.append("GCS bucket is required for Higgsfield image upload")
        
        return errors
    
    def _submit_generation_request(self, prompt: str, image_url: str, **kwargs) -> Dict[str, Any]:
        """
        Submit video generation request to Higgsfield API
        
        Args:
            prompt: Text prompt for video generation
            image_url: Publicly accessible URL of the input image
            **kwargs: Additional parameters (seed, motions, etc.)
            
        Returns:
            API response containing job_set_id
        """
        url = f"{self.base_url}/image2video/dop"
        
        # Build request payload
        payload = {
            "params": {
                "model": self.model,
                "prompt": prompt,
                "input_images": [
                    {
                        "type": "image_url",
                        "image_url": image_url
                    }
                ],
                "enhance_prompt": kwargs.get("enhance_prompt", True)
            }
        }
        
        # Add optional parameters
        if "seed" in kwargs:
            payload["params"]["seed"] = kwargs["seed"]
        
        if "motions" in kwargs:
            payload["params"]["motions"] = kwargs["motions"]
        
        if "input_images_end" in kwargs:
            payload["params"]["input_images_end"] = kwargs["input_images_end"]
        
        # Add webhook if provided
        if "webhook_url" in kwargs and "webhook_secret" in kwargs:
            payload["webhook"] = {
                "url": kwargs["webhook_url"],
                "secret": kwargs["webhook_secret"]
            }
        
        self.logger.info(f"Submitting generation request to Higgsfield API...")
        self.logger.debug(f"Request URL: {url}")
        self.logger.debug(f"Prompt: {prompt}")
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            self.logger.info(f"Generation request submitted successfully. Job set ID: {result.get('id')}")
            return result
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise APIError("Invalid API key or secret")
            elif response.status_code == 403:
                raise QuotaExceededError("API quota exceeded or access denied")
            elif response.status_code == 400:
                error_msg = "Bad request"
                try:
                    error_detail = response.json()
                    if "detail" in error_detail:
                        error_msg = f"Bad request: {error_detail['detail']}"
                except:
                    pass
                raise InvalidInputError(error_msg)
            elif response.status_code == 422:
                error_msg = "Validation error"
                try:
                    error_detail = response.json()
                    if "detail" in error_detail:
                        error_msg = f"Validation error: {error_detail['detail']}"
                except:
                    pass
                raise InvalidInputError(error_msg)
            else:
                raise APIError(f"HTTP {response.status_code}: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise APIError(f"Request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise APIError(f"Invalid JSON response: {str(e)}")
    
    def _poll_for_completion(self, job_set_id: str) -> Dict[str, Any]:
        """
        Poll Higgsfield API for job completion status
        
        Args:
            job_set_id: The job set ID returned from submission
            
        Returns:
            Job set result with video URLs
        """
        url = f"{self.base_url}/job-sets/{job_set_id}"
        start_time = time.time()
        
        progress_monitor = ProgressMonitor(100)
        progress_monitor.update(0, "Waiting for video generation...")
        
        while time.time() - start_time < self.timeout:
            try:
                # Query job status
                self.logger.info(f"Checking status for job set: {job_set_id}")
                
                response = requests.get(
                    url,
                    headers=self.headers,
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Check job status
                jobs = result.get("jobs", [])
                if not jobs:
                    raise APIError("No jobs found in job set")
                
                # Get the first job (typically only one job per set for single video)
                job = jobs[0]
                status = job.get("status", "unknown")
                
                elapsed = time.time() - start_time
                
                if status == self.STATUS_COMPLETED:
                    # Job completed successfully
                    progress_monitor.update(100, "Video generation complete!")
                    self.logger.info(f"Video generation completed in {format_duration(elapsed)}")
                    return result
                    
                elif status == self.STATUS_FAILED:
                    error_msg = "Video generation failed"
                    raise APIError(error_msg)
                    
                elif status == self.STATUS_NSFW:
                    raise InvalidInputError("Content was flagged as NSFW and rejected")
                    
                elif status in [self.STATUS_QUEUED, self.STATUS_IN_PROGRESS]:
                    # Still processing
                    progress_pct = 30 if status == self.STATUS_QUEUED else 60
                    progress_monitor.update(
                        progress_pct, 
                        f"Status: {status} (elapsed: {format_duration(elapsed)})"
                    )
                    self.logger.info(f"Job status: {status}")
                    time.sleep(self.polling_interval)
                    continue
                    
                else:
                    self.logger.warning(f"Unknown status: {status}")
                    time.sleep(self.polling_interval)
                    continue
                
            except requests.exceptions.HTTPError as e:
                if response.status_code == 404:
                    raise APIError(f"Job set not found: {job_set_id}")
                elif response.status_code == 401:
                    raise APIError("Authentication failed - check API credentials")
                else:
                    self.logger.warning(f"Status check failed with code {response.status_code}: {e}")
                    time.sleep(self.polling_interval)
            except Exception as e:
                self.logger.warning(f"Error checking status: {str(e)}")
                time.sleep(self.polling_interval)
        
        raise GenerationTimeoutError(f"Generation timed out after {self.timeout} seconds")
    
    def _extract_video_url(self, result: Dict[str, Any]) -> str:
        """
        Extract video download URL from API response
        
        Args:
            result: API response from job status query
            
        Returns:
            Video download URL
        """
        try:
            jobs = result.get("jobs", [])
            if not jobs:
                raise VideoGenerationError("No jobs found in result")
            
            job = jobs[0]
            results = job.get("results")
            
            if not results:
                raise VideoGenerationError("No results found in job")
            
            # Try to get the raw/full quality video first
            if "raw" in results and "url" in results["raw"]:
                video_url = results["raw"]["url"]
                self.logger.info(f"Using raw quality video URL")
                return video_url
            
            # Fall back to min quality if raw is not available
            if "min" in results and "url" in results["min"]:
                video_url = results["min"]["url"]
                self.logger.info(f"Using min quality video URL")
                return video_url
            
            # Log the response structure for debugging
            self.logger.error(f"Could not find video URL in response: {json.dumps(result, indent=2)}")
            raise VideoGenerationError("No video URL found in API response")
            
        except Exception as e:
            if isinstance(e, VideoGenerationError):
                raise
            raise VideoGenerationError(f"Failed to extract video URL: {e}")
    
    def generate_video(self, 
                      prompt: str, 
                      input_image_path: str,
                      output_path: str,
                      duration: float = 5.0,
                      **kwargs) -> str:
        """
        Generate video using Higgsfield API
        
        Args:
            prompt: Text prompt describing the desired video
            input_image_path: Path to the input/reference image
            output_path: Path where the generated video should be saved
            duration: Desired duration (note: Higgsfield generates fixed ~5s clips)
            **kwargs: Additional parameters:
                - seed: Random seed for reproducibility (1-1000000)
                - motions: List of motion presets with strength
                - enhance_prompt: Whether to enhance the prompt (default: True)
                
        Returns:
            Path to the generated video file
        """
        # Validate inputs
        validation_errors = self.validate_inputs(prompt, input_image_path, duration)
        if validation_errors:
            raise InvalidInputError(f"Input validation failed: {'; '.join(validation_errors)}")
        
        # Log cost estimate
        estimated_cost = self.estimate_cost(duration)
        self.logger.info(f"Estimated cost: ${estimated_cost:.2f}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Use RetryHandler for the entire generation process
        retry_handler = RetryHandler(max_retries=self.max_retries)
        
        def _generate_attempt():
            # Upload image to GCS and get signed URL
            self.logger.info(f"Uploading keyframe image to GCS...")
            image_url = self._upload_image_to_gcs(input_image_path)
            self.logger.info(f"Image uploaded. URL: {image_url[:100]}...")
            
            # Submit generation request
            response = self._submit_generation_request(prompt, image_url, **kwargs)
            
            # Extract job set ID
            job_set_id = response.get("id")
            if not job_set_id:
                raise VideoGenerationError("No job set ID in API response")
            
            # Poll for completion
            self.logger.info(f"Polling for completion of job set: {job_set_id}")
            result = self._poll_for_completion(job_set_id)
            
            # Extract video URL
            video_url = self._extract_video_url(result)
            
            # Download the generated video
            self.logger.info(f"Downloading video from: {video_url[:100]}...")
            download_file(video_url, output_path)
            
            # Verify the file was downloaded successfully
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise VideoGenerationError("Downloaded video file is empty or missing")
            
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            self.logger.info(f"Video generated successfully: {output_path} ({file_size_mb:.2f} MB)")
            return output_path
        
        try:
            return retry_handler.retry_with_backoff(_generate_attempt)
        except (APIError, GenerationTimeoutError, QuotaExceededError) as e:
            # Don't retry these specific errors
            raise e
        except Exception as e:
            raise VideoGenerationError(f"Video generation failed: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if Higgsfield API is available"""
        if not self.api_key or not self.api_secret:
            return False
        
        # Check GCS access
        if not self.gcs_bucket:
            return False
        
        try:
            # Test GCS connectivity
            bucket = self.storage_client.bucket(self.gcs_bucket)
            bucket.exists()
            return True
            
        except Exception as e:
            self.logger.debug(f"Higgsfield availability check failed: {str(e)}")
            return False
