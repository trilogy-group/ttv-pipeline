"""
Wan2.1 local video generator implementation
"""

import os
import logging
import subprocess
import tempfile
from typing import Dict, Any, List, Optional
from video_generator_interface import (
    VideoGeneratorInterface, 
    VideoGenerationError,
    InvalidInputError
)
from generators.base import ImageValidator

class Wan21Generator(VideoGeneratorInterface):
    """Local video generator using Wan2.1 models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.wan2_dir = config.get("wan2_dir", "./Wan2.1")
        self.i2v_model_dir = config.get("i2v_model_dir")
        self.flf2v_model_dir = config.get("flf2v_model_dir")
        self.gpu_count = config.get("gpu_count", 1)
        self.size = config.get("size", "1280*720")
        self.guide_scale = config.get("guide_scale", 5.0)
        self.sample_steps = config.get("sample_steps", 40)
        self.sample_shift = config.get("sample_shift", 5.0)
        self.frame_num = config.get("frame_num", 81)
        self.max_retries = config.get("chaining_max_retries", 3)
        self.use_fsdp_flags = config.get("chaining_use_fsdp_flags", True)
        
        # Validate directories exist
        if not os.path.exists(self.wan2_dir):
            raise VideoGenerationError(f"Wan2.1 directory not found: {self.wan2_dir}")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return Wan2.1 capabilities"""
        return {
            "max_duration": 10.0,  # Approximately 5 seconds at 81 frames
            "supported_resolutions": ["1280*720", "1024*576", "720*480"],
            "supports_image_to_video": True,
            "supports_text_to_video": False,  # Pure text-to-video not used in pipeline
            "requires_gpu": True,
            "api_based": False,
            "models": {
                "i2v": self.i2v_model_dir is not None,
                "flf2v": self.flf2v_model_dir is not None
            }
        }
    
    def estimate_cost(self, duration: float, resolution: str = "1280x720") -> float:
        """Local generation has no API cost"""
        return 0.0
    
    def validate_inputs(self, 
                       prompt: str, 
                       input_image_path: str,
                       duration: float) -> List[str]:
        """Validate inputs for Wan2.1 generation"""
        errors = []
        
        # Validate prompt
        if not prompt or len(prompt.strip()) == 0:
            errors.append("Prompt cannot be empty")
        elif len(prompt) > 1000:
            errors.append("Prompt too long (max 1000 characters)")
        
        # Validate image
        image_validation = ImageValidator.validate_image(input_image_path)
        if not image_validation["valid"]:
            errors.extend(image_validation["errors"])
        
        # Validate duration
        capabilities = self.get_capabilities()
        if duration > capabilities["max_duration"]:
            errors.append(f"Duration {duration}s exceeds maximum {capabilities['max_duration']}s")
        
        # Check model availability
        if not self.i2v_model_dir or not os.path.exists(self.i2v_model_dir):
            errors.append(f"I2V model directory not found: {self.i2v_model_dir}")
        
        return errors
    
    def generate_video(self, 
                      prompt: str, 
                      input_image_path: str,
                      output_path: str,
                      duration: float = 5.0,
                      **kwargs) -> str:
        """
        Generate video using Wan2.1 I2V model
        
        This wraps the existing pipeline functionality for consistency
        with the new interface.
        """
        # Validate inputs
        validation_errors = self.validate_inputs(prompt, input_image_path, duration)
        if validation_errors:
            raise InvalidInputError(f"Input validation failed: {'; '.join(validation_errors)}")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Build the command
        cmd = self._build_command(prompt, input_image_path, output_path, **kwargs)
        
        # Execute with retries
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Generating video (attempt {attempt + 1}/{self.max_retries})")
                self._run_command(cmd)
                
                # Verify output exists
                if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
                    self.logger.info(f"Successfully generated video: {output_path}")
                    return output_path
                else:
                    raise VideoGenerationError("Generated video file is missing or too small")
                    
            except subprocess.CalledProcessError as e:
                if attempt < self.max_retries - 1:
                    self.logger.warning(f"Generation failed, retrying with reduced parameters...")
                    # Try with reduced parameters on retry
                    cmd = self._build_command(
                        prompt, input_image_path, output_path,
                        sample_steps=30, guide_scale=3.0, **kwargs
                    )
                else:
                    raise VideoGenerationError(f"Video generation failed after {self.max_retries} attempts: {e}")
            except Exception as e:
                if attempt < self.max_retries - 1:
                    self.logger.warning(f"Error on attempt {attempt + 1}: {e}")
                else:
                    raise
        
        raise VideoGenerationError("Failed to generate video after all retries")
    
    def _build_command(self, 
                      prompt: str, 
                      input_image_path: str,
                      output_path: str,
                      sample_steps: Optional[int] = None,
                      guide_scale: Optional[float] = None,
                      **kwargs) -> List[str]:
        """Build the command for video generation"""
        # Base command
        if self.gpu_count > 1:
            # Distributed execution with torchrun
            cmd = [
                "torchrun",
                f"--nproc_per_node={self.gpu_count}",
                f"--rdzv_endpoint=localhost:29500",
                "generate.py"
            ]
        else:
            cmd = ["python", "generate.py"]
        
        # Add arguments
        cmd.extend([
            "--save_file", output_path,
            "--task", "i2v-14B",
            "--size", self.size,
            "--ckpt_dir", self.i2v_model_dir,
            "--image", input_image_path,
            "--prompt", prompt,
            "--sample_guide_scale", str(guide_scale or self.guide_scale),
            "--sample_steps", str(sample_steps or self.sample_steps),
            "--sample_shift", str(self.sample_shift),
            "--frame_num", str(self.frame_num)
        ])
        
        # Add FSDP flags for distributed processing
        if self.gpu_count >= 2 and self.use_fsdp_flags:
            cmd.extend(["--ulysses_size", str(self.gpu_count)])
            cmd.extend(["--dit_fsdp", "--t5_fsdp"])
        
        return cmd
    
    def _run_command(self, cmd: List[str]) -> str:
        """Execute a command and return output"""
        self.logger.info(f"Executing: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd=self.wan2_dir,
            capture_output=True,
            text=True,
            check=True
        )
        
        return result.stdout
    
    def is_available(self) -> bool:
        """Check if Wan2.1 is properly set up"""
        try:
            # Check if directories exist
            if not os.path.exists(self.wan2_dir):
                self.logger.error(f"Wan2.1 directory not found: {self.wan2_dir}")
                return False
            
            # Check if generate.py exists
            generate_script = os.path.join(self.wan2_dir, "generate.py")
            if not os.path.exists(generate_script):
                self.logger.error(f"generate.py not found in {self.wan2_dir}")
                return False
            
            # Check if model directory exists
            if self.i2v_model_dir and not os.path.exists(self.i2v_model_dir):
                self.logger.error(f"I2V model directory not found: {self.i2v_model_dir}")
                return False
            
            # Check GPU availability
            try:
                import torch
                if not torch.cuda.is_available():
                    self.logger.warning("No CUDA GPUs available")
                    return False
                
                gpu_count = torch.cuda.device_count()
                if gpu_count < self.gpu_count:
                    self.logger.warning(
                        f"Requested {self.gpu_count} GPUs but only {gpu_count} available"
                    )
                    return False
                    
            except ImportError:
                self.logger.error("PyTorch not installed")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking Wan2.1 availability: {e}")
            return False
