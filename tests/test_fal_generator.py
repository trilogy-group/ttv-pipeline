from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
from PIL import Image

from generators.factory import create_video_generator
from generators.remote.fal_generator import FalGenerator
from pipeline import (
    build_prompt_enhancement_instructions,
    get_duration_tradeoff,
    get_requested_segment_plan,
    plan_provider_segment_durations,
    validate_prompt_enhancement,
)
from video_generator_interface import APIError, GenerationTimeoutError, InvalidInputError


def _response(status_code, payload, headers=None):
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.headers = headers or {}
    response.json.return_value = payload
    return response


def _images(tmp_path):
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    Image.new("RGB", (128, 72)).save(first)
    Image.new("RGB", (128, 72)).save(last)
    return first, last


def _install_completed_queue(monkeypatch, endpoint, result=None, statuses=None):
    request_id = "req_123"
    request_base = f"https://queue.fal.run/{endpoint}/requests/{request_id}"
    calls = []
    status_responses = iter(
        statuses or [_response(200, {"status": "COMPLETED", "metrics": {"inference_time": 2.5}})]
    )

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if method == "POST":
            return _response(
                200,
                {
                    "request_id": request_id,
                    "status_url": f"{request_base}/status",
                    "response_url": f"{request_base}/response",
                    "cancel_url": f"{request_base}/cancel",
                },
            )
        if url.endswith("/status"):
            return next(status_responses)
        return _response(
            200,
            result
            or {
                "video": {
                    "url": "https://cdn.example.com/video.mp4",
                    "content_type": "video/mp4",
                    "file_size": 42,
                },
                "seed": 7,
            },
            {"X-Fal-Billable-Units": "6"},
        )

    monkeypatch.setattr("generators.remote.fal_generator.requests.request", request)

    def download(url, destination):
        assert url == "https://cdn.example.com/video.mp4"
        Path(destination).write_bytes(b"video")

    monkeypatch.setattr("generators.remote.fal_generator.download_file", download)
    return calls


@pytest.mark.parametrize(
    ("configured_model", "actual_model", "duration", "default_input", "expected_fields"),
    [
        (
            "bytedance/seedance-2.0/image-to-video",
            "bytedance/seedance-2.0/image-to-video",
            12,
            {"resolution": "1080p", "aspect_ratio": "16:9"},
            {"duration": 12, "image_url": True, "end_image_url": True},
        ),
        (
            "fal-ai/veo3.1/image-to-video",
            "fal-ai/veo3.1/first-last-frame-to-video",
            6,
            {"resolution": "720p"},
            {"duration": "6s", "first_frame_url": True, "last_frame_url": True},
        ),
        (
            "fal-ai/veo3.1/fast/image-to-video",
            "fal-ai/veo3.1/fast/first-last-frame-to-video",
            8,
            {},
            {"duration": "8s", "first_frame_url": True, "last_frame_url": True},
        ),
        (
            "fal-ai/minimax/hailuo-02/standard/image-to-video",
            "fal-ai/minimax/hailuo-02/standard/image-to-video",
            10,
            {"resolution": "768P"},
            {"duration": "10", "image_url": True, "end_image_url": True},
        ),
        (
            "fal-ai/minimax/video-01/image-to-video",
            "fal-ai/minimax/video-01/image-to-video",
            6,
            {"prompt_optimizer": False},
            {"image_url": True},
        ),
    ],
)
def test_profiled_payloads_use_queue_and_documented_fields(
    monkeypatch,
    tmp_path,
    configured_model,
    actual_model,
    duration,
    default_input,
    expected_fields,
):
    first, last = _images(tmp_path)
    output = tmp_path / "output.mp4"
    calls = _install_completed_queue(monkeypatch, actual_model)
    generator = FalGenerator(
        {
            "api_key": "test-key",
            "model": configured_model,
            "default_input": default_input,
            "polling_interval": 0,
        }
    )

    result = generator.generate_video(
        prompt="move",
        input_image_path=str(first),
        last_frame_path=str(last),
        output_path=str(output),
        duration=duration,
    )

    assert result == str(output)
    submit = calls[0]
    assert submit[0] == "POST"
    assert submit[1] == f"https://queue.fal.run/{actual_model}"
    payload = submit[2]["json"]
    assert payload["prompt"] == "move"
    assert all(
        payload[name] == value if value is not True else name in payload
        for name, value in expected_fields.items()
    )
    assert "generate_audio" not in payload
    if configured_model.endswith("video-01/image-to-video"):
        assert "duration" not in payload
        assert "end_image_url" not in payload

    headers = submit[2]["headers"]
    assert headers["Authorization"] == "Key test-key"
    assert headers["X-Fal-Store-IO"] == "0"
    assert headers["x-app-fal-disable-fallback"] == "true"
    assert generator.last_request_metadata == {
        "request_id": "req_123",
        "endpoint_id": actual_model,
        "status_url": f"https://queue.fal.run/{actual_model}/requests/req_123/status",
        "response_url": f"https://queue.fal.run/{actual_model}/requests/req_123/response",
        "cancel_url": f"https://queue.fal.run/{actual_model}/requests/req_123/cancel",
        "status": "COMPLETED",
        "metrics": {"inference_time": 2.5},
        "video": {"content_type": "video/mp4", "file_size": 42},
        "seed": 7,
        "billable_units": "6",
    }


@pytest.mark.parametrize(
    ("model", "expected_duration"),
    [
        ("bytedance/seedance-2.0/image-to-video", "auto"),
        ("fal-ai/veo3.1/image-to-video", None),
        ("fal-ai/minimax/hailuo-02/standard/image-to-video", None),
        ("fal-ai/minimax/video-01/image-to-video", None),
    ],
)
def test_omitted_duration_preserves_profile_default(
    monkeypatch, tmp_path, model, expected_duration
):
    first, _ = _images(tmp_path)
    calls = _install_completed_queue(monkeypatch, model)
    generator = FalGenerator({"api_key": "test-key", "model": model})

    generator.generate_video("move", str(first), str(tmp_path / "out.mp4"), duration=None)

    payload = calls[0][2]["json"]
    if expected_duration is None:
        assert "duration" not in payload
    else:
        assert payload["duration"] == expected_duration


def test_submit_is_not_retried_after_ambiguous_network_failure(monkeypatch, tmp_path):
    first, _ = _images(tmp_path)
    request = Mock(side_effect=requests.ConnectionError("connection lost"))
    monkeypatch.setattr("generators.remote.fal_generator.requests.request", request)
    generator = FalGenerator(
        {
            "api_key": "test-key",
            "model": "bytedance/seedance-2.0/image-to-video",
            "max_retries": 3,
        }
    )

    with pytest.raises(APIError, match="outcome is unknown"):
        generator.generate_video("move", str(first), str(tmp_path / "out.mp4"), 6)

    assert request.call_count == 1


def test_explicitly_rejected_submission_is_retried(monkeypatch, tmp_path):
    first, _ = _images(tmp_path)
    endpoint = "bytedance/seedance-2.0/image-to-video"
    request_id = "req_retry"
    request_base = f"https://queue.fal.run/{endpoint}/requests/{request_id}"
    responses = iter(
        [
            _response(429, {"error": "busy"}, {"X-Fal-Needs-Retry": "true"}),
            _response(
                200,
                {
                    "request_id": request_id,
                    "status_url": f"{request_base}/status",
                    "response_url": f"{request_base}/response",
                    "cancel_url": f"{request_base}/cancel",
                },
            ),
            _response(200, {"status": "COMPLETED"}),
            _response(200, {"video": {"url": "https://cdn.example.com/video.mp4"}}),
        ]
    )
    request = Mock(side_effect=lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr("generators.remote.fal_generator.requests.request", request)
    monkeypatch.setattr("generators.remote.fal_generator.time.sleep", lambda *_: None)
    monkeypatch.setattr("generators.remote.fal_generator.random.uniform", lambda *_: 0)
    monkeypatch.setattr(
        "generators.remote.fal_generator.download_file",
        lambda _url, destination: Path(destination).write_bytes(b"video"),
    )
    generator = FalGenerator({"api_key": "test-key", "model": endpoint, "max_retries": 2})

    generator.generate_video("move", str(first), str(tmp_path / "out.mp4"), 6)

    assert request.call_count == 4


def test_veo_first_last_profile_falls_back_to_image_endpoint_without_last_frame(
    monkeypatch, tmp_path
):
    first, _ = _images(tmp_path)
    actual = "fal-ai/veo3.1/image-to-video"
    calls = _install_completed_queue(monkeypatch, actual)
    generator = FalGenerator(
        {
            "api_key": "test-key",
            "model": "fal-ai/veo3.1/first-last-frame-to-video",
        }
    )

    generator.generate_video("move", str(first), str(tmp_path / "out.mp4"), 4)

    assert calls[0][1] == f"https://queue.fal.run/{actual}"
    assert "image_url" in calls[0][2]["json"]


def test_validation_error_is_not_retried_and_exposes_error_type(monkeypatch, tmp_path):
    first, _ = _images(tmp_path)
    request = Mock(
        return_value=_response(
            422,
            {"detail": "invalid duration", "error_type": "model_validation"},
            {"X-Fal-Needs-Retry": "true"},
        )
    )
    monkeypatch.setattr("generators.remote.fal_generator.requests.request", request)
    generator = FalGenerator(
        {
            "api_key": "test-key",
            "model": "bytedance/seedance-2.0/image-to-video",
            "max_retries": 3,
        }
    )

    with pytest.raises(APIError, match="invalid duration") as caught:
        generator.generate_video("move", str(first), str(tmp_path / "out.mp4"), 6)

    assert request.call_count == 1
    assert caught.value.status_code == 422
    assert caught.value.error_type == "model_validation"


def test_retryable_status_read_honors_retry_after_without_resubmitting(monkeypatch, tmp_path):
    first, _ = _images(tmp_path)
    endpoint = "bytedance/seedance-2.0/image-to-video"
    calls = _install_completed_queue(
        monkeypatch,
        endpoint,
        statuses=[
            _response(429, {"error": "busy"}, {"Retry-After": "0"}),
            _response(200, {"status": "COMPLETED"}),
        ],
    )
    sleeps = []
    monkeypatch.setattr("generators.remote.fal_generator.time.sleep", sleeps.append)
    monkeypatch.setattr("generators.remote.fal_generator.random.uniform", lambda *_: 0)
    generator = FalGenerator({"api_key": "test-key", "model": endpoint, "max_retries": 2})

    generator.generate_video("move", str(first), str(tmp_path / "out.mp4"), 6)

    assert sum(method == "POST" for method, *_ in calls) == 1
    assert sleeps == [0]


def test_timeout_attempts_best_effort_cancellation(monkeypatch, tmp_path):
    first, _ = _images(tmp_path)
    endpoint = "bytedance/seedance-2.0/image-to-video"
    _install_completed_queue(
        monkeypatch,
        endpoint,
        statuses=[_response(200, {"status": "IN_QUEUE"})],
    )
    cancel = Mock(return_value=_response(202, {"status": "CANCELLATION_REQUESTED"}))
    monkeypatch.setattr("generators.remote.fal_generator.requests.put", cancel)
    monkeypatch.setattr(
        "generators.remote.fal_generator.time.monotonic", Mock(side_effect=[0, 0, 1])
    )
    generator = FalGenerator({"api_key": "test-key", "model": endpoint, "timeout": 0.1})

    with pytest.raises(GenerationTimeoutError):
        generator.generate_video("move", str(first), str(tmp_path / "out.mp4"), 6)

    cancel.assert_called_once()
    assert generator.last_request_metadata["cancellation_requested"] is True


def test_cancellation_check_attempts_best_effort_cancellation(monkeypatch, tmp_path):
    first, _ = _images(tmp_path)
    endpoint = "bytedance/seedance-2.0/image-to-video"
    _install_completed_queue(
        monkeypatch,
        endpoint,
        statuses=[_response(200, {"status": "IN_QUEUE"})],
    )
    cancel = Mock(return_value=_response(202, {"status": "CANCELLATION_REQUESTED"}))
    monkeypatch.setattr("generators.remote.fal_generator.requests.put", cancel)
    monkeypatch.setattr("generators.remote.fal_generator.time.sleep", lambda *_: None)
    cancellation_check = Mock(side_effect=[False, True])
    generator = FalGenerator({"api_key": "test-key", "model": endpoint})

    with pytest.raises(InterruptedError, match="cancelled"):
        generator.generate_video(
            "move",
            str(first),
            str(tmp_path / "out.mp4"),
            6,
            cancellation_check=cancellation_check,
        )

    cancel.assert_called_once()
    assert generator.last_request_metadata["cancellation_requested"] is True


def test_keyboard_interrupt_attempts_best_effort_cancellation(monkeypatch, tmp_path):
    first, _ = _images(tmp_path)
    endpoint = "bytedance/seedance-2.0/image-to-video"
    _install_completed_queue(monkeypatch, endpoint)
    cancel = Mock(return_value=_response(202, {}))
    monkeypatch.setattr("generators.remote.fal_generator.requests.put", cancel)
    generator = FalGenerator({"api_key": "test-key", "model": endpoint})
    monkeypatch.setattr(
        generator, "_wait_for_completion", Mock(side_effect=KeyboardInterrupt)
    )

    with pytest.raises(KeyboardInterrupt):
        generator.generate_video("move", str(first), str(tmp_path / "out.mp4"), 6)

    cancel.assert_called_once()


def test_retry_after_parsing_and_sleep_respect_deadline(monkeypatch):
    generator = FalGenerator(
        {"api_key": "test-key", "model": "bytedance/seedance-2.0/image-to-video"}
    )
    invalid = _response(429, {}, {"Retry-After": "not-a-date"})
    delayed = _response(429, {}, {"Retry-After": "3600"})
    sleep = Mock()
    monkeypatch.setattr("generators.remote.fal_generator.time.sleep", sleep)
    monkeypatch.setattr("generators.remote.fal_generator.time.monotonic", lambda: 10)
    monkeypatch.setattr("generators.remote.fal_generator.random.uniform", lambda *_: 0)

    assert generator._retry_after(invalid) is None
    with pytest.raises(GenerationTimeoutError):
        generator._sleep_before_retry(0, delayed, deadline=11)
    sleep.assert_not_called()


def test_strict_output_and_profile_input_validation(monkeypatch, tmp_path):
    first, _ = _images(tmp_path)
    endpoint = "bytedance/seedance-2.0/image-to-video"
    _install_completed_queue(
        monkeypatch,
        endpoint,
        result={"output": {"url": "https://cdn.example.com/wrong.mp4"}},
    )
    generator = FalGenerator({"api_key": "test-key", "model": endpoint})

    with pytest.raises(Exception, match="video.url"):
        generator.generate_video("move", str(first), str(tmp_path / "out.mp4"), 6)

    with pytest.raises(InvalidInputError, match="unknown_option"):
        generator.generate_video(
            "move",
            str(first),
            str(tmp_path / "out.mp4"),
            6,
            fal_input={"unknown_option": False},
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "bytedance/seedance-2.0/image-to-video",
        "bytedance/seedance-2.0/fast/image-to-video",
    ],
)
def test_seedance_profiles_forward_generate_audio(monkeypatch, tmp_path, endpoint):
    first, _ = _images(tmp_path)
    calls = _install_completed_queue(monkeypatch, endpoint)
    generator = FalGenerator(
        {
            "api_key": "test-key",
            "model": endpoint,
            "default_input": {"generate_audio": False},
        }
    )

    generator.generate_video("move", str(first), str(tmp_path / "out.mp4"), 4)

    assert calls[0][2]["json"]["generate_audio"] is False


def test_veo_profile_forwards_generate_audio(monkeypatch, tmp_path):
    first, _ = _images(tmp_path)
    endpoint = "fal-ai/veo3.1/image-to-video"
    calls = _install_completed_queue(monkeypatch, endpoint)
    generator = FalGenerator(
        {
            "api_key": "test-key",
            "model": endpoint,
            "default_input": {"generate_audio": False},
        }
    )

    generator.generate_video("move", str(first), str(tmp_path / "out.mp4"), 4)

    assert calls[0][2]["json"]["generate_audio"] is False


def test_unprofiled_endpoint_fails_configuration_and_fal_key_is_preferred(monkeypatch):
    with pytest.raises(Exception, match="Unsupported fal.ai model endpoint"):
        FalGenerator({"api_key": "test-key", "model": "fal-ai/unknown"})

    monkeypatch.setenv("FAL_KEY", "documented-key")
    monkeypatch.setenv("FAL_API_KEY", "legacy-key")
    generator = FalGenerator({"model": "bytedance/seedance-2.0/image-to-video"})
    assert generator.api_key == "documented-key"
    assert generator.get_capabilities()["supports_text_to_video"] is False


def test_fal_duration_planning_and_llm_validation():
    config = {
        "default_backend": "fal",
        "generation_mode": "keyframe",
        "single_keyframe_mode": True,
        "duration_seconds": 17,
        "fal": {"model": "bytedance/seedance-2.0/image-to-video"},
    }
    assert plan_provider_segment_durations(17, tuple(range(4, 16))) == [13, 4]
    assert plan_provider_segment_durations(17, (6, 10)) == [6, 6, 6]
    assert plan_provider_segment_durations(17, (6,)) == [6, 6, 6]
    maximum_plan = plan_provider_segment_durations(14_440, tuple(range(4, 16)))
    assert sum(maximum_plan) == 14_440
    assert get_requested_segment_plan(config) == [13, 4]

    config["fal"]["model"] = "fal-ai/minimax/hailuo-02/standard/image-to-video"
    assert get_requested_segment_plan(config) == [6, 6, 6]
    config["fal"]["model"] = "fal-ai/minimax/video-01/image-to-video"
    assert "final generated endpoint" in get_duration_tradeoff(config)
    config["fal"]["model"] = "fal-ai/veo3.1/image-to-video"
    assert get_requested_segment_plan(config) == [8, 4, 6]
    config["fal"]["model"] = "bytedance/seedance-2.0/image-to-video"

    result = {
        "segmentation_logic": {
            "total_duration_seconds": 17,
            "number_of_segments": 2,
            "reasoning": "requested runtime",
        },
        "keyframe_prompts": [
            {"segment": 1, "prompt": "first"},
            {"segment": 2, "prompt": "second"},
        ],
        "video_prompts": [
            {
                "segment": 1,
                "prompt": "first",
                "first_frame": "provided_start_image.png",
                "last_frame": "segment_01.png",
                "duration_seconds": 13,
            },
            {
                "segment": 2,
                "prompt": "second",
                "first_frame": "segment_01.png",
                "last_frame": "segment_02.png",
                "duration_seconds": 4,
            },
        ],
    }
    validate_prompt_enhancement(result, config)
    assert "duration_seconds values [13, 4]" in build_prompt_enhancement_instructions(config)
    assert get_duration_tradeoff(config) is None


def test_factory_creates_profiled_fal_generator():
    model = "fal-ai/minimax/hailuo-02/standard/image-to-video"
    generator = create_video_generator(
        "fal.ai",
        {
            "fal": {"api_key": "test-key", "model": model},
            "remote_api_settings": {"polling_interval": 2, "timeout": 10},
        },
    )

    assert isinstance(generator, FalGenerator)
    assert generator.model == model
    assert generator.base_url == "https://queue.fal.run"
    assert generator.polling_interval == 2
