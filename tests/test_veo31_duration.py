import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from PIL import Image
from pydantic import ValidationError

from api.models import JobCreateRequest
from generators.factory import create_video_generator
from generators.remote.veo3_generator import Veo3Generator
from pipeline import (
    MAX_REQUESTED_DURATION_SECONDS,
    build_prompt_enhancement_instructions,
    generate_video_segments_single_keyframe,
    get_duration_tradeoff,
    get_trim_duration_seconds,
    main,
    plan_veo_segment_durations,
    run_pipeline,
    stitch_video_segments,
    validate_prompt_enhancement,
)
from video_generator_interface import VideoGenerationError


def test_requested_duration_plan_and_llm_validation():
    assert plan_veo_segment_durations(4) == [4]
    assert plan_veo_segment_durations(10) == [4, 6]
    assert plan_veo_segment_durations(17) == [8, 4, 6]

    config = {
        "default_backend": "wan2.1",
        "default_video_generation_backend": "veo3",
        "duration_seconds": 9,
    }
    result = {
        "segmentation_logic": {
            "total_duration_seconds": 9,
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
                "duration_seconds": 4,
            },
            {
                "segment": 2,
                "prompt": "second",
                "first_frame": "segment_01.png",
                "last_frame": "segment_02.png",
                "duration_seconds": 6,
            },
        ],
    }

    validate_prompt_enhancement(result, config)
    assert "duration_seconds values [4, 6]" in build_prompt_enhancement_instructions(config)
    assert "ending keyframe will not appear" in get_duration_tradeoff(config)
    assert get_trim_duration_seconds(config) == 9
    assert get_trim_duration_seconds({"default_backend": "veo3", "duration_seconds": 10}) is None

    result["video_prompts"][1]["last_frame"] = None
    with pytest.raises(ValueError, match="first_frame and last_frame"):
        validate_prompt_enhancement(result, config)
    result["video_prompts"][1]["last_frame"] = "segment_02.png"

    result["video_prompts"][1]["duration_seconds"] = 8
    with pytest.raises(ValueError, match="requested segment plan"):
        validate_prompt_enhancement(result, config)


def test_omitted_duration_keeps_veo_planning_ai_inferred():
    config = {"default_backend": "veo3"}
    result = {
        "segmentation_logic": {
            "total_duration_seconds": 14,
            "number_of_segments": 2,
            "reasoning": "inferred runtime",
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
                "duration_seconds": 6,
            },
            {
                "segment": 2,
                "prompt": "second",
                "first_frame": "segment_01.png",
                "last_frame": "segment_02.png",
                "duration_seconds": 8,
            },
        ],
    }

    validate_prompt_enhancement(result, config)
    assert "Infer the final runtime" in build_prompt_enhancement_instructions(config)
    assert get_trim_duration_seconds(config) is None


def test_pipeline_forwards_segment_duration_and_both_frames(tmp_path):
    first_frame = tmp_path / "first.png"
    last_frame = tmp_path / "last.png"
    first_frame.touch()
    last_frame.touch()
    generator = Mock()

    with patch("generators.factory.create_video_generator", return_value=generator):
        generate_video_segments_single_keyframe(
            {
                "default_backend": "veo3",
                "i2i_mode": {"keyframe_position": "last"},
            },
            [{
                "segment": 1,
                "prompt": "move",
                "first_frame": str(first_frame),
                "last_frame": str(last_frame),
                "duration_seconds": 6,
            }],
            str(tmp_path),
        )

    assert generator.generate_video.call_args.kwargs["duration"] == 6
    assert generator.generate_video.call_args.kwargs["input_image_path"] == str(first_frame)
    assert generator.generate_video.call_args.kwargs["last_frame_path"] == str(last_frame)


def test_requested_duration_disables_incompatible_fallback(tmp_path):
    frame = tmp_path / "frame.png"
    frame.touch()
    generator = Mock()
    generator.generate_video.side_effect = VideoGenerationError("Veo failed")

    with patch("generators.factory.create_video_generator", return_value=generator), \
         patch("generators.factory.get_fallback_generator") as get_fallback:
        with pytest.raises(VideoGenerationError, match="Veo failed"):
            generate_video_segments_single_keyframe(
                {"default_backend": "veo3", "duration_seconds": 6},
                [{
                    "segment": 1,
                    "prompt": "move",
                    "first_frame": str(frame),
                    "last_frame": str(frame),
                    "duration_seconds": 6,
                }],
                str(tmp_path),
            )

    get_fallback.assert_not_called()


def test_veo_factory_forwards_fast_model():
    with patch.object(Veo3Generator, "_init_clients"):
        generator = create_video_generator(
            "veo3",
            {
                "google_veo": {
                    "project_id": "test-project",
                    "veo_model": "veo-3.1-fast-generate-001",
                }
            },
        )

    assert generator.model_name == "veo-3.1-fast-generate-001"


def test_veo_request_contains_model_duration_and_both_frames(tmp_path):
    first_frame = tmp_path / "first.png"
    last_frame = tmp_path / "last.png"
    Image.new("RGB", (128, 72)).save(first_frame)
    Image.new("RGB", (128, 72)).save(last_frame)
    output_path = tmp_path / "video.mp4"

    operation = SimpleNamespace(
        done=True,
        response=object(),
        result=SimpleNamespace(
            generated_videos=[SimpleNamespace(video=SimpleNamespace(uri="gs://outputs/video.mp4"))]
        ),
    )
    generate_videos = Mock(return_value=operation)

    with patch.object(Veo3Generator, "_init_clients"):
        generator = Veo3Generator({"project_id": "test-project", "max_retries": 1})
    generator.genai_client = SimpleNamespace(
        models=SimpleNamespace(generate_videos=generate_videos),
        operations=Mock(),
    )
    generator.storage_client = Mock()
    generator._ensure_bucket_exists = Mock()
    generator._upload_to_gcs = Mock(side_effect=["gs://inputs/first.png", "gs://inputs/last.png"])
    generator._download_from_gcs = Mock(return_value=str(output_path))

    generator.generate_video(
        prompt="A smooth transition",
        input_image_path=str(first_frame),
        last_frame_path=str(last_frame),
        output_path=str(output_path),
        duration=6,
    )

    request = generate_videos.call_args.kwargs
    assert request["model"] == "veo-3.1-generate-001"
    assert request["image"].gcs_uri == "gs://inputs/first.png"
    assert request["config"].duration_seconds == 6
    assert request["config"].last_frame.gcs_uri == "gs://inputs/last.png"


def test_trim_and_cli_duration_interfaces(tmp_path):
    with patch("pipeline.run_command") as run_command:
        stitch_video_segments(["segment.mp4"], str(tmp_path / "final.mp4"), 9)
    command = run_command.call_args.args[0]
    assert command[command.index("-t") + 1] == "9"
    assert command[command.index("-c:v") + 1] == "libx264"

    with patch.object(sys, "argv", ["pipeline.py", "--config", "config.yaml", "--duration-seconds", "15"]), \
         patch("pipeline.run_pipeline") as run_pipeline:
        main()
    run_pipeline.assert_called_once_with("config.yaml", None, 15)


def test_duration_limit_and_api_validation():
    assert sum(plan_veo_segment_durations(MAX_REQUESTED_DURATION_SECONDS)) == 14_440
    with pytest.raises(ValueError, match="at most 14440"):
        plan_veo_segment_durations(14_441)

    with pytest.raises(ValidationError):
        JobCreateRequest(prompt="move", duration_seconds=True)
    with pytest.raises(ValidationError):
        JobCreateRequest(prompt="move", duration_seconds=14_441)


def test_cli_rejects_unsupported_duration_before_generation():
    with patch(
        "pipeline.load_config",
        return_value={"prompt": "move", "default_backend": "wan2.1"},
    ), patch("pipeline.enhance_prompt") as enhance_prompt:
        with pytest.raises(ValueError, match="supported only for the veo3 backend"):
            run_pipeline("config.yaml", duration_seconds=6)

    enhance_prompt.assert_not_called()
