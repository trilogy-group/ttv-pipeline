import json
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
    VideoPrompt,
    build_prompt_enhancement_instructions,
    enhance_prompt_data,
    generate_video_chaining_mode,
    generate_video_segments_single_keyframe,
    get_backend_clip_durations,
    get_duration_tradeoff,
    get_provider_compatible_duration,
    get_requested_job_timeout,
    get_requested_segment_plan,
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
        "generation_mode": "keyframe",
        "single_keyframe_mode": True,
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
    assert get_trim_duration_seconds({
        "default_backend": "veo3",
        "generation_mode": "keyframe",
        "single_keyframe_mode": True,
        "duration_seconds": 10,
    }) is None

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
    instructions = build_prompt_enhancement_instructions(config)
    assert "Infer the final runtime" in instructions
    assert '"total_duration_seconds": 8' in instructions
    assert instructions.count('"duration_seconds": 4') >= 2
    assert '"duration_seconds": 5' not in instructions
    assert get_trim_duration_seconds(config) is None


@pytest.mark.parametrize("resolution", ["1080p", "4k"])
def test_high_resolution_veo_plans_and_validates_only_eight_second_clips(
    tmp_path, resolution
):
    config = {
        "default_backend": "veo3",
        "generation_mode": "keyframe",
        "single_keyframe_mode": True,
        "duration_seconds": 10,
        "google_veo": {"resolution": resolution},
    }

    assert get_backend_clip_durations(config) == (8,)
    assert get_requested_segment_plan(config) == [8, 8]

    frame = tmp_path / "frame.png"
    Image.new("RGB", (160, 90)).save(frame)
    with patch.object(Veo3Generator, "_init_clients"):
        generator = Veo3Generator(
            {"api_key": "test-key", "resolution": resolution}
        )

    assert generator.validate_inputs("move", str(frame), 8) == []
    assert any(
        "requires an 8-second duration" in error
        for error in generator.validate_inputs("move", str(frame), 6)
    )


def test_fallback_to_veo_uses_supported_duration():
    config = {"segment_duration_seconds": 5}

    assert get_provider_compatible_duration(config, "minimax", "veo3", 5) == 6
    assert get_provider_compatible_duration(config, "fal", "veo3", 9) == 8
    config["google_veo"] = {"resolution": "1080p"}
    assert get_provider_compatible_duration(config, "minimax", "veo3", 5) == 8


def test_pipeline_forwards_segment_duration_and_both_frames(tmp_path):
    first_frame = tmp_path / "first.png"
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    last_frame = frames_dir / "last.png"
    Image.new("RGB", (4, 4)).save(first_frame)
    Image.new("RGB", (4, 4)).save(last_frame)
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
                "last_frame": "last.png",
                "duration_seconds": 6,
            }],
            str(tmp_path),
        )

    assert generator.generate_video.call_args.kwargs["duration"] == 6
    assert generator.generate_video.call_args.kwargs["input_image_path"] == str(first_frame)
    assert generator.generate_video.call_args.kwargs["last_frame_path"] == str(last_frame)


def test_inferred_veo_fallback_uses_configured_duration(tmp_path):
    frame = tmp_path / "frame.png"
    Image.new("RGB", (4, 4)).save(frame)
    generator = Mock()
    generator.generate_video.side_effect = VideoGenerationError("Veo failed")
    fallback = Mock()
    config = {
        "default_backend": "veo3",
        "segment_duration_seconds": 5,
        "remote_api_settings": {"fallback_backend": "minimax"},
    }

    with patch("generators.factory.create_video_generator", return_value=generator), \
         patch("generators.factory.get_fallback_generator", return_value=fallback):
        generate_video_segments_single_keyframe(
            config,
            [{
                "segment": 1,
                "prompt": "move",
                "first_frame": str(frame),
                "last_frame": str(frame),
                "duration_seconds": 8,
            }],
            str(tmp_path),
        )

    assert fallback.generate_video.call_args.kwargs["duration"] == 5


def test_chaining_veo_fallback_uses_configured_duration(tmp_path):
    frame = tmp_path / "frame.png"
    Image.new("RGB", (4, 4)).save(frame)
    primary = Mock()
    primary.get_backend_name.return_value = "veo3"
    primary.validate_inputs.return_value = []
    primary.generate_video.side_effect = VideoGenerationError("Veo failed")
    fallback = Mock()
    fallback.get_backend_name.return_value = "minimax"
    fallback.validate_inputs.return_value = []

    def generate_fallback(**kwargs):
        with open(kwargs["output_path"], "wb") as file:
            file.write(b"video")
        return kwargs["output_path"]

    fallback.generate_video.side_effect = generate_fallback
    config = {
        "default_backend": "veo3",
        "initial_image": str(frame),
        "segment_duration_seconds": 5,
        "remote_api_settings": {"fallback_backend": "minimax"},
    }

    with patch("pipeline.create_video_generator", return_value=primary), \
         patch("pipeline.get_fallback_generator", return_value=fallback):
        generate_video_chaining_mode(
            config,
            [{"segment": 1, "prompt": "move", "duration_seconds": 8}],
            str(tmp_path),
            segment_duration=5,
        )

    assert primary.validate_inputs.call_args.kwargs["duration"] == 8
    assert fallback.validate_inputs.call_args.kwargs["duration"] == 5
    assert fallback.generate_video.call_args.kwargs["duration"] == 5


def test_requested_duration_disables_incompatible_fallback(tmp_path):
    frame = tmp_path / "frame.png"
    Image.new("RGB", (4, 4)).save(frame)
    generator = Mock()
    generator.generate_video.side_effect = VideoGenerationError("Veo failed")

    with patch("generators.factory.create_video_generator", return_value=generator), \
         patch("generators.factory.get_fallback_generator") as get_fallback:
        with pytest.raises(VideoGenerationError, match="Veo failed"):
            generate_video_segments_single_keyframe(
                {
                    "default_backend": "veo3",
                    "generation_mode": "keyframe",
                    "single_keyframe_mode": True,
                    "duration_seconds": 6,
                },
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


def test_veo_factory_uses_google_api_key_and_preview_model():
    with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
        with patch.object(Veo3Generator, "_init_clients"):
            generator = create_video_generator("veo3", {"google_veo": {}})

    assert generator.api_key == "test-key"
    assert generator.model_name == "veo-3.1-generate-preview"
    assert generator.estimate_cost(8) == 3.2


def test_veo_preview_model_requires_api_key():
    with patch.dict(
        "os.environ", {"GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""}
    ), patch.object(Veo3Generator, "_init_clients"):
        with pytest.raises(VideoGenerationError, match="Gemini API mode requires"):
            create_video_generator(
                "veo3",
                {
                    "google_veo": {
                        "project_id": "test-project",
                        "veo_model": "veo-3.1-generate-preview",
                    }
                },
            )


def test_veo_api_key_normalizes_vertex_model_name():
    with patch.object(Veo3Generator, "_init_clients"):
        generator = Veo3Generator(
            {"api_key": "test-key", "veo_model": "veo-3.1-generate-001"}
        )

    assert generator.model_name == "veo-3.1-generate-preview"


def test_veo_factory_does_not_force_developer_api_for_vertex_model():
    with patch.object(Veo3Generator, "_init_clients"):
        generator = create_video_generator(
            "veo3",
            {
                "google_veo": {
                    "project_id": "test-project",
                    "api_key": "developer-key",
                    "veo_model": "veo-3.1-generate-001",
                }
            },
        )

    assert generator.api_key is None
    assert generator.model_name == "veo-3.1-generate-001"


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


def test_veo_google_api_key_request_uses_local_frames_and_downloads(tmp_path):
    first_frame = tmp_path / "first.png"
    last_frame = tmp_path / "last.png"
    Image.new("RGB", (128, 72)).save(first_frame)
    Image.new("RGB", (128, 72)).save(last_frame)
    output_path = tmp_path / "video.mp4"
    video = Mock()
    operation = SimpleNamespace(
        done=True,
        response=SimpleNamespace(generated_videos=[SimpleNamespace(video=video)]),
    )
    generate_videos = Mock(return_value=operation)

    with patch.object(Veo3Generator, "_init_clients"):
        generator = Veo3Generator({"api_key": "test-key", "max_retries": 1})
    generator.genai_client = SimpleNamespace(
        models=SimpleNamespace(generate_videos=generate_videos),
        operations=Mock(),
        files=SimpleNamespace(download=Mock()),
    )

    generator.generate_video(
        prompt="A smooth transition",
        input_image_path=str(first_frame),
        last_frame_path=str(last_frame),
        output_path=str(output_path),
        duration=8,
    )

    request = generate_videos.call_args.kwargs
    assert request["model"] == "veo-3.1-generate-preview"
    assert request["image"].image_bytes
    assert request["config"].last_frame.image_bytes
    assert request["config"].duration_seconds == 8
    generator.genai_client.files.download.assert_called_once_with(file=video)
    video.save.assert_called_once_with(str(output_path))


def test_veo_google_api_key_does_not_retry_rejected_request(tmp_path):
    first_frame = tmp_path / "first.png"
    Image.new("RGB", (128, 72)).save(first_frame)
    rejected_request = Mock(side_effect=ValueError("invalid request"))

    with patch.object(Veo3Generator, "_init_clients"):
        generator = Veo3Generator({"api_key": "test-key", "max_retries": 3})
    generator.genai_client = SimpleNamespace(
        models=SimpleNamespace(generate_videos=rejected_request),
    )

    with pytest.raises(VideoGenerationError, match="invalid request"):
        generator.generate_video(
            prompt="A smooth transition",
            input_image_path=str(first_frame),
            output_path=str(tmp_path / "video.mp4"),
            duration=8,
        )

    rejected_request.assert_called_once()


def test_trim_and_cli_duration_interfaces(tmp_path):
    with patch("pipeline.run_command") as run_command:
        stitch_video_segments(["segment.mp4"], str(tmp_path / "final.mp4"), 9)
    command = run_command.call_args.args[0]
    assert command[command.index("-t") + 1] == "9"
    assert command[command.index("-c:v") + 1] == "libx264"

    with patch.object(sys, "argv", ["pipeline.py", "--config", "config.yaml", "--duration-seconds", "15"]), \
         patch("pipeline.run_pipeline") as run_pipeline:
        main()
    run_pipeline.assert_called_once_with(
        "config.yaml", None, 15, False, None, False, False
    )

    with patch.object(
        sys,
        "argv",
        [
            "pipeline.py",
            "--config",
            "config.yaml",
            "--plan-only",
            "--enhanced-prompt-file",
            "reviewed.json",
        ],
    ), patch("pipeline.run_pipeline") as run_pipeline:
        main()
    run_pipeline.assert_called_once_with(
        "config.yaml", None, None, True, "reviewed.json", False, False
    )

    with patch.object(
        sys,
        "argv",
        [
            "pipeline.py",
            "--config",
            "config.yaml",
            "--enhanced-prompt-file",
            "reviewed.json",
            "--keyframes-only",
        ],
    ), patch("pipeline.run_pipeline") as run_pipeline:
        main()
    run_pipeline.assert_called_once_with(
        "config.yaml", None, None, False, "reviewed.json", True, False
    )


def test_reviewed_prompt_plan_skips_llm_and_is_validated():
    plan = {
        "segmentation_logic": {
            "total_duration_seconds": 4,
            "number_of_segments": 1,
            "reasoning": "reviewed",
        },
        "keyframe_prompts": [{"segment": 1, "prompt": "frame"}],
        "video_prompts": [{
            "segment": 1,
            "prompt": "move",
            "first_frame": "provided_start_image.png",
            "last_frame": "segment_01.png",
            "duration_seconds": 4,
        }],
    }
    config = {
        "default_backend": "veo3",
        "generation_mode": "keyframe",
        "single_keyframe_mode": True,
        "duration_seconds": 4,
        "enhanced_prompt": plan,
    }

    with patch("pipeline.PromptEnhancer") as prompt_enhancer:
        assert enhance_prompt_data("unused", config) == plan
    prompt_enhancer.assert_not_called()

    config["duration_seconds"] = 6
    with pytest.raises(ValueError, match="requested segment plan"):
        enhance_prompt_data("unused", config)


def test_plan_only_preserves_existing_media(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("prompt: test\n")
    frame = tmp_path / "output" / "frames" / "existing.png"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"existing")
    plan = {
        "segmentation_logic": {
            "total_duration_seconds": 5,
            "number_of_segments": 1,
            "reasoning": "test",
        },
        "keyframe_prompts": [{"segment": 1, "prompt": "frame"}],
        "video_prompts": [{
            "segment": 1,
            "prompt": "move",
            "duration_seconds": 5,
        }],
    }

    def write_plan(*_args):
        plan_path = tmp_path / "output" / "enhanced_prompt.json"
        plan_path.write_text(json.dumps(plan))
        return plan

    with patch("pipeline.enhance_prompt", side_effect=write_plan):
        result = run_pipeline(str(config_path), plan_only=True)

    assert result == str(tmp_path / "output" / "enhanced_prompt.json")
    assert frame.read_bytes() == b"existing"


def test_duration_limit_and_api_validation():
    assert sum(plan_veo_segment_durations(MAX_REQUESTED_DURATION_SECONDS)) == 14_440
    with pytest.raises(ValueError, match="at most 14440"):
        plan_veo_segment_durations(14_441)

    with pytest.raises(ValidationError):
        JobCreateRequest(prompt="move", duration_seconds=True)
    with pytest.raises(ValidationError):
        JobCreateRequest(prompt="move", duration_seconds=14_441)
    with pytest.raises(ValidationError):
        VideoPrompt(segment=1, prompt="move")


def test_long_requested_duration_batches_prompt_enhancement():
    def batch_result(durations):
        return {
            "segmentation_logic": {
                "total_duration_seconds": sum(durations),
                "number_of_segments": len(durations),
                "reasoning": "continue the narrative",
            },
            "keyframe_prompts": [
                {"segment": index, "prompt": f"frame {index}"}
                for index in range(1, len(durations) + 1)
            ],
            "video_prompts": [
                {
                    "segment": index,
                    "prompt": f"video {index}",
                    "first_frame": (
                        "provided_start_image.png"
                        if index == 1
                        else f"segment_{index - 1:02d}.png"
                    ),
                    "last_frame": f"segment_{index:02d}.png",
                    "duration_seconds": duration,
                }
                for index, duration in enumerate(durations, start=1)
            ],
        }

    final_batch = batch_result([8])
    final_batch["video_prompts"][0]["first_frame"] = "segment_20.png"
    final_batch["video_prompts"][0]["last_frame"] = "segment_21.png"

    with patch("pipeline.PromptEnhancer") as prompt_enhancer:
        prompt_enhancer.return_value.enhance.side_effect = [
            batch_result([8] * 20),
            final_batch,
        ]
        result = enhance_prompt_data(
            "A long journey",
            {
                "default_backend": "veo3",
                "generation_mode": "keyframe",
                "single_keyframe_mode": True,
                "duration_seconds": 168,
            },
        )

    assert prompt_enhancer.return_value.enhance.call_count == 2
    assert result["segmentation_logic"]["number_of_segments"] == 21
    assert result["video_prompts"][20]["segment"] == 21
    assert result["video_prompts"][20]["first_frame"] == "segment_20.png"
    assert result["video_prompts"][20]["last_frame"] == "segment_21.png"


def test_chaining_mode_validates_its_actual_backend():
    config = {
        "generation_mode": "chaining",
        "default_backend": "veo3",
        "duration_seconds": 6,
    }
    with pytest.raises(ValueError, match="requires generation_mode=keyframe"):
        get_requested_segment_plan(config)


def test_requested_duration_scales_queue_timeout():
    config = {
        "default_backend": "veo3",
        "generation_mode": "keyframe",
        "single_keyframe_mode": True,
        "duration_seconds": MAX_REQUESTED_DURATION_SECONDS,
        "remote_api_settings": {"timeout": 600},
    }
    assert get_requested_job_timeout(config) == 3600 + 1805 * 600

    config.pop("duration_seconds")
    config["enhanced_prompt"] = {"video_prompts": [{}, {}, {}]}
    assert get_requested_job_timeout(config) == 3600 + 3 * 600


def test_cli_rejects_unsupported_duration_before_generation():
    with patch(
        "pipeline.load_config",
        return_value={
            "prompt": "move",
            "default_backend": "wan2.1",
            "generation_mode": "keyframe",
            "single_keyframe_mode": True,
        },
    ), patch("pipeline.enhance_prompt") as enhance_prompt:
        with pytest.raises(ValueError, match="supported only for veo3 and fal backends"):
            run_pipeline("config.yaml", duration_seconds=6)

    enhance_prompt.assert_not_called()
