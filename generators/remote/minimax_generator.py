"""
Minimax API video generator implementation using I2V-01-Director model
"""

import os
import time
import json
import base64
import mimetypes
import logging
import requests
from typing import Dict, Any, List, Optional
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

class MinimaxGenerator(VideoGeneratorInterface):
    """Remote video generator using Minimax API with I2V-01-Director model"""
    
    # Pricing per second (estimated based on typical I2V pricing)
    PRICING = {
        "I2V-01-Director": 0.02,  # $0.02 per second (estimated)
    }
    
    # Supported camera movements and effects
    CAMERA_MOVEMENTS = [
        "Truck left", "Truck right", "Pan left", "Pan right",
        "Tilt up", "Tilt down", "Zoom in", "Zoom out",
        "Dolly in", "Dolly out", "Static", "Orbital"
    ]
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.model = config.get("model", "I2V-01-Director")
        self.max_duration = config.get("max_duration", 6)  # Minimax typical max
        self.base_url = config.get("base_url", "https://api.minimaxi.chat/v1")
        self.max_retries = config.get("max_retries", 3)
        self.polling_interval = config.get("polling_interval", 30)  # Poll every 30 seconds
        self.timeout = config.get("timeout", 1800)  # 30 minutes default
        
        # Validate configuration
        if not self.api_key:
            # Check environment variable as fallback
            self.api_key = os.getenv("MINIMAX_API_KEY")
            if not self.api_key:
                raise VideoGenerationError(
                    "Minimax API key not found in config or MINIMAX_API_KEY environment variable"
                )
        
        self.headers = {
            'authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        self.logger.info(f"Initialized Minimax generator with model: {self.model}")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return Minimax backend capabilities"""
        return {
            "max_duration": self.max_duration,
            "supported_resolutions": ["720x1280", "1280x720", "1024x1024"],  # Common I2V resolutions
            "supported_formats": ["mp4"],
            "supports_image_to_video": True,
            "supports_text_to_video": False,
            "requires_gpu": False,
            "api_based": True,
            "supports_camera_movements": True,
            "camera_movements": self.CAMERA_MOVEMENTS,
            "max_prompt_length": 500,
            "cost_per_second": self.PRICING.get(self.model, 0.02)
        }
    
    def estimate_cost(self, duration: float, resolution: str = "1280x720") -> float:
        """Estimate cost for video generation"""
        cost_per_second = self.PRICING.get(self.model, 0.02)
        return duration * cost_per_second
    
    def validate_inputs(self, 
                       prompt: str, 
                       input_image_path: str,
                       duration: float) -> List[str]:
        """Validate inputs for Minimax generation"""
        errors = []
        
        # Validate prompt
        if not prompt or len(prompt.strip()) == 0:
            errors.append("Prompt cannot be empty")
        elif len(prompt) > 500:
            errors.append("Prompt too long (max 500 characters for Minimax)")
        
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
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 format required by Minimax API"""
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                # Determine image format from file extension
                if image_path.lower().endswith('.png'):
                    return f"data:image/png;base64,{image_data}"
                else:
                    return f"data:image/jpeg;base64,{image_data}"
        except Exception as e:
            raise VideoGenerationError(f"Failed to encode image: {str(e)}")
    
    def _submit_generation_request(self, prompt: str, image_path: str) -> Dict[str, Any]:
        """Submit video generation request to Minimax API"""
        url = f"{self.base_url}/video_generation"
        
        # Encode image to base64
        base64_image = self._encode_image_to_base64(image_path)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "first_frame_image": base64_image
        }
        
        self.logger.info(f"Submitting generation request to Minimax API...")
        self.logger.debug(f"Request URL: {url}")
        self.logger.debug(f"Prompt: {prompt}")
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            self.logger.info("Generation request submitted successfully")
            return result
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise APIError("Invalid API key or authentication failed")
            elif response.status_code == 403:
                raise QuotaExceededError("API quota exceeded or access denied")
            elif response.status_code == 400:
                error_msg = "Bad request"
                try:
                    error_detail = response.json().get("error", {}).get("message", "")
                    if error_detail:
                        error_msg = f"Bad request: {error_detail}"
                except:
                    pass
                raise InvalidInputError(error_msg)
            else:
                raise APIError(f"HTTP {response.status_code}: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise APIError(f"Request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise APIError(f"Invalid JSON response: {str(e)}")
    
    def _poll_for_completion(self, task_id: str) -> Dict[str, Any]:
        """Poll Minimax API for task completion status"""
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            try:
                # Check task status
                self.logger.info(f"Checking status for task: {task_id}")
                
                status_response = requests.get(
                    f"{self.base_url}/query/video_generation",
                    headers=self.headers,
                    params={"task_id": task_id},
                    timeout=30
                )
                
                if status_response.status_code == 200:
                    result = status_response.json()
                    status = result.get("status", "unknown")
                    
                    if status == "Success":
                        # Task completed successfully
                        return result
                    elif status == "Fail":
                        error_msg = result.get("message", "Generation failed")
                        raise APIError(f"Generation failed: {error_msg}")
                    elif status in ["Processing", "Queueing", "Preparing"]:
                        # Still processing, continue polling
                        self.logger.info(f"Task {task_id} status: {status}")
                        time.sleep(self.polling_interval)
                        continue
                    else:
                        self.logger.warning(f"Unknown status: {status}")
                        time.sleep(self.polling_interval)
                        continue
                else:
                    self.logger.warning(f"Status check failed with code {status_response.status_code}")
                    time.sleep(self.polling_interval)
                
            except Exception as e:
                self.logger.warning(f"Error checking status: {str(e)}")
                time.sleep(self.polling_interval)
        
        raise GenerationTimeoutError(f"Generation timed out after {self.timeout} seconds")
    
    def _get_download_url(self, file_id: str) -> str:
        """Retrieve download URL for a file using Minimax files API"""
        try:
            # Extract group_id from JWT token if available
            group_id = self._extract_group_id_from_token()
            
            url = f"{self.base_url}/files/retrieve"
            params = {
                "GroupId": group_id,
                "file_id": file_id
            }
            
            self.logger.info(f"Retrieving download URL for file_id: {file_id}")
            
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            if "file" in result and "download_url" in result["file"]:
                download_url = result["file"]["download_url"]
                self.logger.info(f"Retrieved download URL: {download_url}")
                return download_url
            else:
                self.logger.error(f"No download URL in file retrieval response: {json.dumps(result, indent=2)}")
                raise VideoGenerationError("No download URL found in file retrieval response")
                
        except requests.exceptions.RequestException as e:
            if response.status_code == 401:
                raise APIError("Authentication failed - check API key")
            elif response.status_code == 429:
                raise QuotaExceededError("Rate limit exceeded")
            else:
                raise APIError(f"File retrieval request failed: {str(e)}")
        except Exception as e:
            raise VideoGenerationError(f"Failed to retrieve download URL: {str(e)}")
    
    def _extract_group_id_from_token(self) -> str:
        """Extract group_id from JWT token"""
        try:
            # JWT tokens have 3 parts separated by dots
            parts = self.api_key.split('.')
            if len(parts) != 3:
                raise ValueError("Invalid JWT token format")
            
            # Decode the payload (second part)
            payload = parts[1]
            # Add padding if needed
            payload += '=' * (4 - len(payload) % 4)
            
            decoded = base64.b64decode(payload)
            token_data = json.loads(decoded)
            
            if "GroupID" in token_data:
                return token_data["GroupID"]
            else:
                raise ValueError("GroupID not found in token")
                
        except Exception as e:
            self.logger.error(f"Failed to extract group_id from token: {str(e)}")
            # Fallback: try to use a default or raise error
            raise VideoGenerationError("Could not extract GroupID from API token. Please check your API key format.")
    
    def _extract_video_url(self, response: Dict[str, Any]) -> str:
        """Extract video URL from Minimax API response"""
        # Check for file_id first (most common case for Minimax)
        if "file_id" in response:
            file_id = response["file_id"]
            self.logger.info(f"Found file_id in response: {file_id}")
            return self._get_download_url(file_id)
        
        # Check for direct file URL in the response data
        if "file_url" in response:
            return response["file_url"]
        elif "data" in response and "file_url" in response["data"]:
            return response["data"]["file_url"]
        elif "result" in response and "file_url" in response["result"]:
            return response["result"]["file_url"]
        
        # Also check for video_url as fallback
        if "video_url" in response:
            return response["video_url"]
        elif "data" in response and "video_url" in response["data"]:
            return response["data"]["video_url"]
        elif "result" in response and "video_url" in response["result"]:
            return response["result"]["video_url"]
        
        # Log the response structure for debugging
        self.logger.error(f"Could not find video URL in response: {json.dumps(response, indent=2)}")
        raise VideoGenerationError("No video URL found in API response")
    
    def generate_video(self, 
                      prompt: str, 
                      input_image_path: str,
                      output_path: str,
                      duration: float = 5.0,
                      **kwargs) -> str:
        """Generate video using Minimax API"""
        # Validate inputs
        validation_errors = self.validate_inputs(prompt, input_image_path, duration)
        if validation_errors:
            raise InvalidInputError(f"Input validation failed: {'; '.join(validation_errors)}")
        
        # Log cost estimate
        estimated_cost = self.estimate_cost(duration)
        self.logger.info(f"Estimated cost: ${estimated_cost:.2f}")
        
        # Add camera movement suggestions to prompt if not present
        enhanced_prompt = self._enhance_prompt_with_camera_movement(prompt)
        
        # Use RetryHandler's retry_with_backoff method
        retry_handler = RetryHandler(max_retries=self.max_retries)
        
        def _generate_attempt():
            # Submit generation request
            response = self._submit_generation_request(enhanced_prompt, input_image_path)
            
            # Handle response based on API behavior
            if "task_id" in response:
                # Async processing - poll for completion
                self.logger.info(f"Task submitted with ID: {response['task_id']}")
                result = self._poll_for_completion(response["task_id"])
                video_url = self._extract_video_url(result)
            else:
                # Synchronous response - extract video URL directly
                video_url = self._extract_video_url(response)
            
            # Download the generated video
            self.logger.info(f"Downloading video from: {video_url}")
            download_file(video_url, output_path)
            
            # Verify the file was downloaded successfully
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise VideoGenerationError("Downloaded video file is empty or missing")
            
            self.logger.info(f"Video generated successfully: {output_path}")
            return output_path
        
        try:
            return retry_handler.retry_with_backoff(_generate_attempt)
        except (APIError, GenerationTimeoutError, QuotaExceededError) as e:
            # Don't retry these specific errors
            raise e
        except Exception as e:
            raise VideoGenerationError(f"Video generation failed: {str(e)}")
    
    def _enhance_prompt_with_camera_movement(self, prompt: str) -> str:
        """Enhance prompt with camera movement if not already specified"""
        # Check if prompt already contains camera movement instructions
        prompt_lower = prompt.lower()
        has_camera_movement = any(
            movement.lower() in prompt_lower or f"[{movement.lower()}]" in prompt_lower
            for movement in self.CAMERA_MOVEMENTS
        )
        
        if not has_camera_movement:
            # Add a subtle camera movement if none specified
            enhanced = f"[Static]{prompt}"
            self.logger.debug(f"Enhanced prompt with camera movement: {enhanced}")
            return enhanced
        
        return prompt
    
    def is_available(self) -> bool:
        """Check if Minimax API is available"""
        if not self.api_key:
            return False
        
        try:
            # Test API connectivity with a minimal request
            url = f"{self.base_url}/models"  # Assuming a models endpoint exists
            headers = {'authorization': f'Bearer {self.api_key}'}
            
            response = requests.get(url, headers=headers, timeout=10)
            return response.status_code in [200, 404]  # 404 is OK if endpoint doesn't exist
            
        except Exception as e:
            self.logger.debug(f"Minimax availability check failed: {str(e)}")
            return False
