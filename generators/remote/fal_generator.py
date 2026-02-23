"""
fal.ai video generator implementation.

This backend treats fal.ai as the provider and the model identifier as a
runtime parameter, so the same generator can target any fal-supported model.
"""

import base64
import json
import os
from typing import Any

import requests

from generators.base import ImageValidator, RetryHandler, download_file
from video_generator_interface import (
    APIError,
    InvalidInputError,
    VideoGenerationError,
    VideoGeneratorInterface,
)


class FalGenerator(VideoGeneratorInterface):
    """Remote video generator using fal.ai model endpoints."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("FAL_API_KEY")
        self.model = config.get("model")
        self.base_url = config.get("base_url", "https://fal.run").rstrip("/")
        self.max_duration = config.get("max_duration", 10)
        self.timeout = config.get("timeout", 600)
        self.max_retries = config.get("max_retries", 3)
        self.default_input = config.get("default_input", {})
        self.last_request_metrics: dict[str, Any] = {}

        if not self.api_key:
            raise VideoGenerationError("fal.ai API key is required (set config.fal.api_key or FAL_API_KEY)")
        if not self.model:
            raise VideoGenerationError("fal.ai model is required (set config.fal.model)")

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "provider": "fal.ai",
            "model": self.model,
            "max_duration": self.max_duration,
            "supports_image_to_video": True,
            "supports_text_to_video": True,
            "requires_gpu": False,
            "api_based": True,
            "provider_model_separation": True,
            "header_metrics": [
                "x-fal-request-id",
                "x-compute-time",
                "x-queue-time",
                "x-total-cost",
                "x-request-cost",
            ],
        }

    def estimate_cost(self, duration: float, resolution: str = "1280x720") -> float:
        # fal cost depends on model and provider-side pricing; rely on header metrics when available.
        return 0.0

    def validate_inputs(self, prompt: str, input_image_path: str, duration: float) -> list[str]:
        errors: list[str] = []

        if not prompt or not prompt.strip():
            errors.append("Prompt cannot be empty")

        image_validation = ImageValidator.validate_image(input_image_path, max_size_mb=10.0)
        if not image_validation["valid"]:
            errors.extend(image_validation["errors"])

        if duration <= 0:
            errors.append("Duration must be positive")
        elif duration > self.max_duration:
            errors.append(f"Duration {duration}s exceeds maximum configured duration {self.max_duration}s")

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
            raise InvalidInputError(f"Input validation failed: {'; '.join(validation_errors)}")

        model = kwargs.get("fal_model") or self.model
        endpoint = f"{self.base_url}/{model.lstrip('/')}"

        payload: dict[str, Any] = dict(self.default_input)
        payload.update(kwargs.get("fal_input", {}))
        payload.update({
            "prompt": prompt,
            "image_url": self._image_to_data_uri(input_image_path),
            "duration": duration,
        })

        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        retry_handler = RetryHandler(max_retries=self.max_retries)

        def _generate_attempt() -> str:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout)

            if response.status_code == 401:
                raise APIError("fal.ai authentication failed", status_code=401)
            if response.status_code == 429:
                raise APIError("fal.ai rate limit exceeded", status_code=429)
            if response.status_code >= 400:
                raise APIError(
                    f"fal.ai request failed with status {response.status_code}",
                    status_code=response.status_code,
                    response_body=response.text,
                )

            try:
                response_json = response.json()
            except json.JSONDecodeError as exc:
                raise APIError("fal.ai returned non-JSON response") from exc

            self.last_request_metrics = self._extract_response_metrics(response.headers)
            if self.last_request_metrics:
                self.logger.info(f"fal.ai response metrics: {self.last_request_metrics}")

            video_url = self._extract_video_url(response_json)
            download_file(video_url, output_path)
            self._write_metrics_sidecar(output_path)
            return output_path

        return retry_handler.retry_with_backoff(_generate_attempt)

    def is_available(self) -> bool:
        return bool(self.api_key and self.model)

    def _image_to_data_uri(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            data = base64.b64encode(image_file.read()).decode("utf-8")
        mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
        return f"data:{mime_type};base64,{data}"

    def _extract_video_url(self, payload: Any) -> str:
        url = self._find_first_url(payload)
        if not url:
            raise VideoGenerationError("No downloadable URL found in fal.ai response")
        return url

    def _find_first_url(self, value: Any) -> str | None:
        if isinstance(value, dict):
            for key in ("video", "video_url", "url", "file", "output"):
                if key in value:
                    result = self._find_first_url(value[key])
                    if result:
                        return result
            for nested in value.values():
                result = self._find_first_url(nested)
                if result:
                    return result
            return None

        if isinstance(value, list):
            for item in value:
                result = self._find_first_url(item)
                if result:
                    return result
            return None

        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
        return None

    def _extract_response_metrics(self, headers: dict[str, Any]) -> dict[str, Any]:
        metrics: dict[str, Any] = {}

        request_id = headers.get("x-fal-request-id")
        if request_id:
            metrics["request_id"] = request_id

        for header_key, metric_key in {
            "x-compute-time": "compute_time",
            "x-queue-time": "queue_time",
            "x-total-cost": "total_cost",
            "x-request-cost": "request_cost",
            "x-generation-time": "generation_time",
        }.items():
            value = headers.get(header_key)
            if value is not None:
                metrics[metric_key] = self._parse_numeric(value)

        return metrics

    @staticmethod
    def _parse_numeric(value: Any) -> Any:
        if isinstance(value, (int, float)):
            return value
        if not isinstance(value, str):
            return value
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def _write_metrics_sidecar(self, output_path: str) -> None:
        if not self.last_request_metrics:
            return

        sidecar_path = f"{output_path}.metrics.json"
        try:
            with open(sidecar_path, "w") as sidecar_file:
                json.dump(self.last_request_metrics, sidecar_file, indent=2)
        except Exception as exc:  # pragma: no cover
            self.logger.warning(f"Failed to write fal.ai metrics sidecar: {exc}")
