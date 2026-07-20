"""fal.ai video generation through explicit model profiles and the queue API."""

import base64
import os
import random
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

import requests

from generators.base import ImageValidator, download_file
from video_generator_interface import (
    APIError,
    GenerationTimeoutError,
    InvalidInputError,
    VideoGenerationError,
    VideoGeneratorInterface,
)


@dataclass(frozen=True)
class FalModelProfile:
    endpoint: str
    durations: tuple[int, ...]
    duration_encoding: str | None
    first_frame_field: str
    last_frame_field: str | None = None
    max_image_mb: float = 10.0
    option_values: tuple[tuple[str, tuple[Any, ...] | None], ...] = ()
    image_endpoint: str | None = None
    first_last_endpoint: str | None = None

    @property
    def options(self) -> dict[str, tuple[Any, ...] | None]:
        return dict(self.option_values)


SEEDANCE_DURATIONS = tuple(range(4, 16))
VEO_DURATIONS = (4, 6, 8)

_SEEDANCE_OPTIONS = (
    ("resolution", ("480p", "720p", "1080p", "4k")),
    ("aspect_ratio", ("auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16")),
    ("generate_audio", (True, False)),
    ("bitrate_mode", ("standard", "high")),
    ("end_user_id", None),
)
_SEEDANCE_FAST_OPTIONS = (
    ("resolution", ("480p", "720p")),
    *_SEEDANCE_OPTIONS[1:],
)
_VEO_OPTIONS = (
    ("aspect_ratio", ("auto", "16:9", "9:16")),
    ("generate_audio", (True, False)),
    ("negative_prompt", None),
    ("resolution", ("720p", "1080p", "4k")),
    ("seed", None),
    ("auto_fix", (True, False)),
    ("safety_tolerance", (1, 2, 3, 4, 5, 6, "1", "2", "3", "4", "5", "6")),
)


def _veo_profile(endpoint: str, image_endpoint: str, first_last_endpoint: str) -> FalModelProfile:
    is_first_last = endpoint == first_last_endpoint
    return FalModelProfile(
        endpoint=endpoint,
        durations=VEO_DURATIONS,
        duration_encoding="seconds",
        first_frame_field="first_frame_url" if is_first_last else "image_url",
        last_frame_field="last_frame_url" if is_first_last else None,
        max_image_mb=8.0,
        option_values=_VEO_OPTIONS,
        image_endpoint=image_endpoint,
        first_last_endpoint=first_last_endpoint,
    )


FAL_MODEL_PROFILES = {
    "bytedance/seedance-2.0/image-to-video": FalModelProfile(
        endpoint="bytedance/seedance-2.0/image-to-video",
        durations=SEEDANCE_DURATIONS,
        duration_encoding="integer_or_auto",
        first_frame_field="image_url",
        last_frame_field="end_image_url",
        max_image_mb=30.0,
        option_values=_SEEDANCE_OPTIONS,
    ),
    "bytedance/seedance-2.0/fast/image-to-video": FalModelProfile(
        endpoint="bytedance/seedance-2.0/fast/image-to-video",
        durations=SEEDANCE_DURATIONS,
        duration_encoding="integer_or_auto",
        first_frame_field="image_url",
        last_frame_field="end_image_url",
        max_image_mb=30.0,
        option_values=_SEEDANCE_FAST_OPTIONS,
    ),
    "fal-ai/minimax/hailuo-02/standard/image-to-video": FalModelProfile(
        endpoint="fal-ai/minimax/hailuo-02/standard/image-to-video",
        durations=(6, 10),
        duration_encoding="string",
        first_frame_field="image_url",
        last_frame_field="end_image_url",
        option_values=(
            ("prompt_optimizer", (True, False)),
            ("resolution", ("512P", "768P")),
        ),
    ),
    "fal-ai/minimax/video-01/image-to-video": FalModelProfile(
        endpoint="fal-ai/minimax/video-01/image-to-video",
        durations=(6,),
        duration_encoding=None,
        first_frame_field="image_url",
        option_values=(("prompt_optimizer", (True, False)),),
    ),
}

for _prefix in ("fal-ai/veo3.1", "fal-ai/veo3.1/fast"):
    _image_endpoint = f"{_prefix}/image-to-video"
    _first_last_endpoint = f"{_prefix}/first-last-frame-to-video"
    FAL_MODEL_PROFILES[_image_endpoint] = _veo_profile(
        _image_endpoint, _image_endpoint, _first_last_endpoint
    )
    FAL_MODEL_PROFILES[_first_last_endpoint] = _veo_profile(
        _first_last_endpoint, _image_endpoint, _first_last_endpoint
    )


def get_fal_clip_durations(model: str | None) -> tuple[int, ...]:
    """Return the allowed clip lengths for a supported fal endpoint."""
    if model not in FAL_MODEL_PROFILES:
        supported = ", ".join(sorted(FAL_MODEL_PROFILES))
        raise ValueError(f"Unsupported fal.ai model endpoint {model!r}. Supported: {supported}")
    return FAL_MODEL_PROFILES[model].durations


def fal_profile_supports_last_frame(model: str | None) -> bool:
    """Return whether the configured profile can condition on an ending frame."""
    get_fal_clip_durations(model)
    assert model is not None
    profile = FAL_MODEL_PROFILES[model]
    return bool(profile.last_frame_field or profile.first_last_endpoint)


class FalGenerator(VideoGeneratorInterface):
    """Remote video generator using profiled fal.ai queue endpoints."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
        model = config.get("model")
        if not isinstance(model, str) or not model:
            raise VideoGenerationError("fal.ai model is required (set config.fal.model)")
        self.model = model
        self.base_url = config.get("base_url", "https://queue.fal.run").rstrip("/")
        self.timeout = float(config.get("timeout", 600))
        self.http_timeout = float(config.get("http_timeout", min(30, self.timeout)))
        self.queue_start_timeout = float(config.get("queue_start_timeout", min(300, self.timeout)))
        self.polling_interval = float(config.get("polling_interval", 5))
        self.max_retries = max(1, int(config.get("max_retries", 3)))
        default_input = config.get("default_input", {})
        if not isinstance(default_input, dict):
            raise VideoGenerationError("fal.default_input must be an object")
        self.default_input = dict(default_input)
        self.last_request_metadata: dict[str, Any] = {}

        if not self.api_key:
            raise VideoGenerationError(
                "fal.ai API key is required (set config.fal.api_key or FAL_KEY)"
            )
        try:
            get_fal_clip_durations(self.model)
        except ValueError as exc:
            raise VideoGenerationError(str(exc)) from exc
        if self.timeout <= 0 or self.http_timeout <= 0 or self.queue_start_timeout <= 0:
            raise VideoGenerationError("fal.ai timeouts must be positive")
        if self.polling_interval < 0:
            raise VideoGenerationError("fal.ai polling_interval cannot be negative")

    def get_capabilities(self) -> dict[str, Any]:
        profile = FAL_MODEL_PROFILES[self.model]
        return {
            "provider": "fal.ai",
            "model": self.model,
            "allowed_durations": list(profile.durations),
            "max_duration": max(profile.durations),
            "supports_image_to_video": True,
            "supports_first_last_frame": bool(
                profile.last_frame_field or profile.first_last_endpoint
            ),
            "supports_text_to_video": False,
            "requires_gpu": False,
            "api_based": True,
            "provider_model_separation": True,
        }

    def estimate_cost(self, duration: float, resolution: str = "1280x720") -> float:
        return 0.0

    def validate_inputs(
        self, prompt: str, input_image_path: str, duration: float | None
    ) -> list[str]:
        profile = FAL_MODEL_PROFILES[self.model]
        errors: list[str] = []

        if not isinstance(prompt, str) or not prompt.strip():
            errors.append("Prompt cannot be empty")

        if not isinstance(input_image_path, str) or not input_image_path:
            errors.append("Input image path is required")
        else:
            image_validation = ImageValidator.validate_image(
                input_image_path, max_size_mb=profile.max_image_mb
            )
            if not image_validation["valid"]:
                errors.extend(image_validation["errors"])

        if duration is not None:
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or duration not in profile.durations
            ):
                errors.append(
                    f"Duration {duration!r}s is unsupported for {self.model}; "
                    f"use one of {list(profile.durations)}"
                )

        return errors

    def generate_video(
        self,
        prompt: str,
        input_image_path: str,
        output_path: str,
        duration: float | None = None,
        **kwargs: Any,
    ) -> str:
        self.last_request_metadata = {}
        validation_errors = self.validate_inputs(prompt, input_image_path, duration)
        if validation_errors:
            raise InvalidInputError(f"Input validation failed: {'; '.join(validation_errors)}")

        last_frame_path = kwargs.get("last_frame_path")
        if last_frame_path is not None and not isinstance(last_frame_path, str):
            raise InvalidInputError("last_frame_path must be a path string")
        profile = self._profile_for_request(bool(last_frame_path))
        if last_frame_path and profile.last_frame_field:
            last_frame_validation = ImageValidator.validate_image(
                last_frame_path, max_size_mb=profile.max_image_mb
            )
            if not last_frame_validation["valid"]:
                raise InvalidInputError(
                    "Last frame validation failed: " + "; ".join(last_frame_validation["errors"])
                )

        payload = self._build_payload(
            profile,
            prompt,
            input_image_path,
            last_frame_path,
            None if duration is None else int(duration),
            kwargs.get("fal_input", {}),
        )
        headers = self._headers()
        submit_url = f"{self.base_url}/{profile.endpoint}"
        deadline = time.monotonic() + self.timeout
        submission, _ = self._request_json(
            "POST",
            submit_url,
            headers=headers,
            payload=payload,
            safe_to_retry=False,
            deadline=deadline,
        )
        lifecycle = self._parse_submission(submission, profile.endpoint)
        self.last_request_metadata = lifecycle.copy()
        cancellation_check = kwargs.get("cancellation_check")

        try:
            status = self._wait_for_completion(
                lifecycle, headers, deadline, cancellation_check
            )
            result, response = self._request_json(
                "GET",
                lifecycle["response_url"],
                headers=headers,
                safe_to_retry=True,
                deadline=deadline,
            )
        except (GenerationTimeoutError, InterruptedError, KeyboardInterrupt):
            self._cancel(lifecycle["cancel_url"], headers)
            raise

        video = result.get("video")
        if not isinstance(video, dict) or not isinstance(video.get("url"), str):
            raise VideoGenerationError("fal.ai response did not contain video.url")

        self.last_request_metadata.update(
            {
                "status": "COMPLETED",
                "metrics": status.get("metrics", {}),
                "video": {
                    key: video[key]
                    for key in ("content_type", "file_name", "file_size")
                    if key in video
                },
            }
        )
        if "seed" in result:
            self.last_request_metadata["seed"] = result["seed"]
        billable_units = self._header(response.headers, "X-Fal-Billable-Units")
        if billable_units is not None:
            self.last_request_metadata["billable_units"] = billable_units

        download_file(video["url"], output_path)
        return output_path

    def is_available(self) -> bool:
        return bool(self.api_key and self.model in FAL_MODEL_PROFILES)

    def _profile_for_request(self, has_last_frame: bool) -> FalModelProfile:
        profile = FAL_MODEL_PROFILES[self.model]
        endpoint = (
            profile.first_last_endpoint
            if has_last_frame and profile.first_last_endpoint
            else (
                profile.image_endpoint
                if not has_last_frame and profile.image_endpoint
                else profile.endpoint
            )
        )
        return FAL_MODEL_PROFILES[endpoint]

    def _build_payload(
        self,
        profile: FalModelProfile,
        prompt: str,
        input_image_path: str,
        last_frame_path: str | None,
        duration: int | None,
        request_input: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(request_input, dict):
            raise InvalidInputError("fal_input must be an object")
        options = {**self.default_input, **request_input}
        unsupported = sorted(set(options) - set(profile.options))
        if unsupported:
            raise InvalidInputError(
                f"Unsupported fal.ai input fields for {profile.endpoint}: {unsupported}"
            )
        for name, allowed in profile.options.items():
            if name in options and allowed is not None and options[name] not in allowed:
                raise InvalidInputError(
                    f"Unsupported {name}={options[name]!r} for {profile.endpoint}; "
                    f"use one of {list(allowed)}"
                )

        payload = {
            **options,
            "prompt": prompt,
            profile.first_frame_field: self._image_to_data_uri(input_image_path),
        }
        if last_frame_path and profile.last_frame_field:
            payload[profile.last_frame_field] = self._image_to_data_uri(last_frame_path)
        if profile.duration_encoding == "integer_or_auto":
            payload["duration"] = "auto" if duration is None else duration
        elif duration is not None and profile.duration_encoding == "seconds":
            payload["duration"] = f"{duration}s"
        elif duration is not None and profile.duration_encoding == "string":
            payload["duration"] = str(duration)
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Fal-Request-Timeout": str(self.queue_start_timeout),
            "X-Fal-Store-IO": "0",
            "x-app-fal-disable-fallback": "true",
        }

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any] | None = None,
        safe_to_retry: bool,
        deadline: float | None = None,
    ) -> tuple[dict[str, Any], requests.Response]:
        for attempt in range(self.max_retries):
            try:
                request_timeout = self.http_timeout
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise GenerationTimeoutError(
                            f"fal.ai generation timed out after {self.timeout:g} seconds"
                        )
                    request_timeout = min(request_timeout, remaining)
                response = requests.request(
                    method,
                    url,
                    headers=headers,
                    json=payload,
                    timeout=request_timeout,
                )
            except requests.RequestException as exc:
                if not safe_to_retry or attempt == self.max_retries - 1:
                    message = (
                        "fal.ai submission outcome is unknown and was not retried"
                        if method == "POST"
                        else "fal.ai queue request failed"
                    )
                    raise APIError(message) from exc
                self._sleep_before_retry(attempt, None, deadline)
                continue

            if response.status_code < 400:
                try:
                    data = response.json()
                except ValueError as exc:
                    raise APIError("fal.ai returned a non-JSON response") from exc
                if not isinstance(data, dict):
                    raise APIError("fal.ai returned an invalid JSON response")
                return data, response

            retryable = self._is_retryable(response)
            if retryable and attempt < self.max_retries - 1:
                self._sleep_before_retry(attempt, response, deadline)
                continue
            raise self._api_error(response)

        raise APIError("fal.ai queue request failed")  # pragma: no cover

    def _wait_for_completion(
        self,
        lifecycle: dict[str, Any],
        headers: dict[str, str],
        deadline: float,
        cancellation_check=None,
    ) -> dict[str, Any]:
        while True:
            if cancellation_check and cancellation_check():
                raise InterruptedError("fal.ai generation cancelled")
            status, _ = self._request_json(
                "GET",
                lifecycle["status_url"],
                headers=headers,
                safe_to_retry=True,
                deadline=deadline,
            )
            state = status.get("status")
            self.last_request_metadata["status"] = state
            if state == "COMPLETED":
                if status.get("error") or status.get("error_type"):
                    raise APIError(
                        f"fal.ai generation failed: {status.get('error') or status.get('error_type')}",
                        error_type=status.get("error_type"),
                    )
                return status
            if state not in {"IN_QUEUE", "IN_PROGRESS"}:
                raise APIError(f"fal.ai returned unknown queue status {state!r}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GenerationTimeoutError(
                    f"fal.ai generation timed out after {self.timeout:g} seconds"
                )
            time.sleep(min(self.polling_interval, remaining))

    def _parse_submission(self, submission: dict[str, Any], endpoint: str) -> dict[str, Any]:
        required = ("request_id", "status_url", "response_url", "cancel_url")
        missing = [name for name in required if not submission.get(name)]
        if missing:
            raise APIError(f"fal.ai queue submission omitted fields: {missing}")
        for name in required[1:]:
            self._validate_lifecycle_url(submission[name])
        return {
            "request_id": submission["request_id"],
            "endpoint_id": endpoint,
            "status_url": submission["status_url"],
            "response_url": submission["response_url"],
            "cancel_url": submission["cancel_url"],
        }

    def _validate_lifecycle_url(self, url: str) -> None:
        expected = urlparse(self.base_url)
        actual = urlparse(url)
        if (actual.scheme, actual.netloc) != (expected.scheme, expected.netloc):
            raise APIError("fal.ai returned an untrusted queue lifecycle URL")

    def _cancel(self, cancel_url: str, headers: dict[str, str]) -> None:
        try:
            requests.put(cancel_url, headers=headers, timeout=self.http_timeout)
            self.last_request_metadata["cancellation_requested"] = True
        except requests.RequestException:
            self.last_request_metadata["cancellation_requested"] = False

    def _sleep_before_retry(
        self,
        attempt: int,
        response: requests.Response | None,
        deadline: float | None = None,
    ) -> None:
        retry_after = self._retry_after(response) if response is not None else None
        delay = retry_after if retry_after is not None else min(2**attempt, 30)
        delay += random.uniform(0, min(1.0, delay * 0.25))
        if deadline is not None and delay >= deadline - time.monotonic():
            raise GenerationTimeoutError(
                f"fal.ai generation timed out after {self.timeout:g} seconds"
            )
        time.sleep(delay)

    def _retry_after(self, response: requests.Response) -> float | None:
        value = self._header(response.headers, "Retry-After")
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            try:
                return max(0.0, parsedate_to_datetime(str(value)).timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                return None

    def _is_retryable(self, response: requests.Response) -> bool:
        if response.status_code in {400, 401, 403, 404, 422}:
            return False
        needs_retry = str(self._header(response.headers, "X-Fal-Needs-Retry") or "").lower() in {
            "1",
            "true",
            "yes",
        }
        return response.status_code in {429, 500, 502, 503, 504} or needs_retry

    def _api_error(self, response: requests.Response) -> APIError:
        try:
            body = response.json()
        except ValueError:
            body = {}
        error_type = self._header(response.headers, "X-Fal-Error-Type")
        if isinstance(body, dict):
            error_type = body.get("error_type") or error_type
            detail = body.get("detail") or body.get("error") or body.get("message")
        else:
            detail = None
        message = f"fal.ai request failed with status {response.status_code}"
        if detail:
            message += f": {str(detail)[:1000]}"
        return APIError(
            message,
            status_code=response.status_code,
            response_body=str(detail)[:1000] if detail else None,
            error_type=error_type,
        )

    @staticmethod
    def _header(headers: Any, name: str) -> Any:
        lowered = name.lower()
        return next((value for key, value in headers.items() if str(key).lower() == lowered), None)

    @staticmethod
    def _image_to_data_uri(image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            data = base64.b64encode(image_file.read()).decode("utf-8")
        extension = os.path.splitext(image_path)[1].lower()
        mime_type = {
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(extension, "image/jpeg")
        return f"data:{mime_type};base64,{data}"
