"""
Local generator using HunyuanVideo framework
"""

import os
import subprocess
import logging
from typing import Dict, Any, List, Optional

from video_generator_interface import (
    VideoGeneratorInterface,
    VideoGenerationError,
    InvalidInputError,
)
from generators.base import ImageValidator


class HunyuanVideoGenerator(VideoGeneratorInterface):
    """Local video generator for Tencent's HunyuanVideo."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hunyuan_dir = config.get("hunyuan_dir", "./HunyuanVideo")
        self.config_file = config.get("config_file")
        self.ckpt_path = config.get("ckpt_path")
        self.max_duration = config.get("max_duration", 5)
        self.sample_steps = config.get("sample_steps", 50)
        self.seed = config.get("seed")

        if not os.path.isdir(self.hunyuan_dir):
            raise VideoGenerationError(
                f"HunyuanVideo directory not found: {self.hunyuan_dir}"
            )
        if not self.config_file or not os.path.exists(self.config_file):
            raise VideoGenerationError(
                f"Config file not found: {self.config_file}"
            )
        if not self.ckpt_path or not os.path.exists(self.ckpt_path):
            raise VideoGenerationError(
                f"Checkpoint not found: {self.ckpt_path}"
            )

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "max_duration": self.max_duration,
            "supported_resolutions": ["1280x720", "1024x576"],
            "supports_image_to_video": True,
            "supports_text_to_video": False,
            "requires_gpu": True,
            "api_based": False,
            "models": {"hunyuan": True},
        }

    def estimate_cost(self, duration: float, resolution: str = "1280x720") -> float:
        return 0.0

    def validate_inputs(self, prompt: str, input_image_path: str, duration: float) -> List[str]:
        errors = []
        if not prompt:
            errors.append("Prompt cannot be empty")
        image_validation = ImageValidator.validate_image(input_image_path)
        if not image_validation["valid"]:
            errors.extend(image_validation["errors"])
        if duration > self.max_duration:
            errors.append(
                f"Duration {duration}s exceeds maximum {self.max_duration}s"
            )
        return errors

    def generate_video(
        self,
        prompt: str,
        input_image_path: str,
        output_path: str,
        duration: float = 5.0,
        **kwargs,
    ) -> str:
        validation_errors = self.validate_inputs(prompt, input_image_path, duration)
        if validation_errors:
            raise InvalidInputError(
                f"Input validation failed: {'; '.join(validation_errors)}"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        seed = kwargs.get("seed", self.seed)
        cmd = [
            "python",
            "inference.py",
            "--cfg-path",
            self.config_file,
            "--ckpt-path",
            self.ckpt_path,
            "--source-image",
            input_image_path,
            "--prompt",
            prompt,
            "--save-path",
            output_path,
            "--steps",
            str(self.sample_steps),
            "--fps",
            "16",
        ]
        if seed is not None:
            cmd.extend(["--seed", str(int(seed))])

        try:
            subprocess.run(cmd, cwd=self.hunyuan_dir, check=True)
            if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
                raise VideoGenerationError(
                    "Generated video file is missing or too small"
                )
            return output_path
        except subprocess.CalledProcessError as e:
            raise VideoGenerationError(f"HunyuanVideo generation failed: {e}")

    def is_available(self) -> bool:
        try:
            return os.path.isdir(self.hunyuan_dir) and os.path.exists(self.config_file)
        except Exception:
            return False
