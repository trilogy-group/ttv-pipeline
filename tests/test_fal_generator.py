import json
from pathlib import Path
from unittest.mock import Mock

from generators.factory import create_video_generator
from generators.remote.fal_generator import FalGenerator


def test_extract_response_metrics_parses_cost_and_time_headers():
    generator = FalGenerator({"api_key": "test-key", "model": "fal-ai/test-model"})

    metrics = generator._extract_response_metrics(
        {
            "x-fal-request-id": "req_123",
            "x-compute-time": "2.31",
            "x-queue-time": "0.19",
            "x-total-cost": "0.048",
            "x-request-cost": "0.048",
        }
    )

    assert metrics["request_id"] == "req_123"
    assert metrics["compute_time"] == 2.31
    assert metrics["queue_time"] == 0.19
    assert metrics["total_cost"] == 0.048
    assert metrics["request_cost"] == 0.048


def test_generate_video_extracts_url_and_writes_metrics_sidecar(monkeypatch, tmp_path):
    generator = FalGenerator(
        {
            "api_key": "test-key",
            "model": "fal-ai/test-model",
            "timeout": 10,
            "max_retries": 1,
        }
    )

    input_image = tmp_path / "input.png"
    input_image.write_bytes(b"image")
    output_path = tmp_path / "output.mp4"

    monkeypatch.setattr(
        "generators.remote.fal_generator.ImageValidator.validate_image",
        lambda *_args, **_kwargs: {"valid": True, "errors": []},
    )

    response = Mock()
    response.status_code = 200
    response.headers = {
        "x-fal-request-id": "req_abc",
        "x-compute-time": "1.2",
        "x-total-cost": "0.01",
    }
    response.json.return_value = {
        "output": {
            "video": {
                "url": "https://cdn.example.com/video.mp4",
            }
        }
    }

    monkeypatch.setattr("generators.remote.fal_generator.requests.post", lambda *args, **kwargs: response)

    def _fake_download(url: str, destination: str) -> None:
        assert url == "https://cdn.example.com/video.mp4"
        Path(destination).write_bytes(b"video")

    monkeypatch.setattr("generators.remote.fal_generator.download_file", _fake_download)

    result_path = generator.generate_video(
        prompt="test prompt",
        input_image_path=str(input_image),
        output_path=str(output_path),
        duration=5,
    )

    assert result_path == str(output_path)
    assert output_path.exists()
    assert generator.last_request_metrics["request_id"] == "req_abc"
    metrics_sidecar = tmp_path / "output.mp4.metrics.json"
    assert metrics_sidecar.exists()
    sidecar = json.loads(metrics_sidecar.read_text())
    assert sidecar["total_cost"] == 0.01


def test_factory_creates_fal_generator_from_provider_backend_name():
    config = {
        "fal": {
            "api_key": "test-key",
            "model": "fal-ai/test-model",
        },
        "remote_api_settings": {
            "max_retries": 1,
            "timeout": 10,
        },
    }

    generator = create_video_generator("fal.ai", config)

    assert isinstance(generator, FalGenerator)
    assert generator.model == "fal-ai/test-model"
