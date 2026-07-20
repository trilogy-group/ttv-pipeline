import json
import zipfile
from unittest.mock import Mock, patch

import pytest
import yaml
from PIL import Image
from pydantic import ValidationError

from api.models import KeyframePrompt
from keyframe_generator import generate_keyframes_from_json
from pipeline import (
    generate_single_video_segment,
    generate_video_segments_sequential,
    run_pipeline,
    validate_prompt_enhancement,
    validate_existing_keyframes,
)
from workers.video_worker import CancellationToken, execute_pipeline_with_config
from workers.gcs_uploader import create_keyframe_storyboard_archive


def one_shot_plan():
    return {
        "segmentation_logic": {
            "total_duration_seconds": 4,
            "number_of_segments": 1,
            "reasoning": "test",
        },
        "keyframe_prompts": [
            {
                "segment": 1,
                "transition": "cut",
                "start_prompt": "independent start",
                "prompt": "same-scene end",
            }
        ],
        "video_prompts": [
            {
                "segment": 1,
                "prompt": "move within the scene",
                "first_frame": "segment_01_start.png",
                "last_frame": "segment_01.png",
                "duration_seconds": 4,
            }
        ],
    }


def test_cut_transition_requires_an_independent_start_prompt():
    with pytest.raises(ValidationError, match="cut keyframes require start_prompt"):
        KeyframePrompt(segment=1, transition="cut", prompt="end")


def test_local_plan_validates_cut_frame_reference():
    plan = one_shot_plan()
    plan["video_prompts"][0]["first_frame"] = "provided_start_image.png"

    with pytest.raises(ValueError, match="cut transition requires first_frame"):
        validate_prompt_enhancement(plan, {"default_backend": "wan2.1"})


def test_cut_resets_conditioning_and_continue_reuses_previous_end(tmp_path):
    plan = {
        "keyframe_prompts": [
            {
                "segment": 1,
                "transition": "cut",
                "start_prompt": "ancient scene start",
                "prompt": "ancient scene end",
            },
            {
                "segment": 2,
                "transition": "continue",
                "prompt": "same ancient scene continuation",
            },
        ]
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    frames_dir = tmp_path / "frames"

    def generate(**kwargs):
        output_path = kwargs["output_path"]
        with open(output_path, "wb") as file:
            file.write(b"frame")
        return output_path

    with patch("keyframe_generator.generate_keyframe", side_effect=generate) as image_call:
        generated = generate_keyframes_from_json(
            str(plan_path), str(frames_dir), model_name="test"
        )

    start_path = str(frames_dir / "segment_01_start.png")
    first_end_path = str(frames_dir / "segment_01.png")
    second_end_path = str(frames_dir / "segment_02.png")
    assert generated == [first_end_path, second_end_path]
    assert [call.kwargs["input_image_path"] for call in image_call.call_args_list] == [
        None,
        start_path,
        first_end_path,
    ]


def test_legacy_first_keyframe_uses_text_to_image_without_initial_frame(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"keyframe_prompts": [{"segment": 1, "prompt": "opening"}]})
    )
    frames_dir = tmp_path / "frames"

    def generate(**kwargs):
        output_path = kwargs["output_path"]
        with open(output_path, "wb") as file:
            file.write(b"frame")
        return output_path

    with patch("keyframe_generator.generate_keyframe", side_effect=generate) as image_call:
        generated = generate_keyframes_from_json(str(plan_path), str(frames_dir))

    assert generated == [str(frames_dir / "segment_01.png")]
    assert image_call.call_args.kwargs["input_image_path"] is None


def test_keyframes_only_stops_before_video_generation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "generation_mode": "keyframe",
                "image_generation_model": "test",
            }
        )
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(one_shot_plan()))

    with patch("pipeline.prepare_keyframes", return_value=["frame.png"]) as prepare, \
         patch("pipeline.generate_video_segments") as generate_video:
        result = run_pipeline(
            str(config_path),
            enhanced_prompt_file=str(plan_path),
            keyframes_only=True,
        )

    assert result == str(tmp_path / "output" / "frames")
    prepare.assert_called_once()
    generate_video.assert_not_called()


def test_plan_only_fails_when_prompt_enhancement_is_skipped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("prompt: test\n")
    stale_plan = tmp_path / "output" / "enhanced_prompt.json"
    stale_plan.parent.mkdir()
    stale_plan.write_text("{}")

    with pytest.raises(ValueError, match="requires an OpenAI API key"):
        run_pipeline(str(config_path), plan_only=True)


def test_storyboard_archive_contains_only_plan_and_frames(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "segment_01_start.png").write_bytes(b"start")
    (frames_dir / "segment_01.png").write_bytes(b"end")
    (tmp_path / "unrelated.mp4").write_bytes(b"video")

    archive_path = create_keyframe_storyboard_archive(str(tmp_path), one_shot_plan())

    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {
            "enhanced_prompt.json",
            "frames/segment_01.png",
            "frames/segment_01_start.png",
        }


def test_reviewed_keyframes_require_matching_dimensions(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    Image.new("RGB", (4, 4)).save(frames_dir / "segment_01_start.png")
    Image.new("RGB", (8, 4)).save(frames_dir / "segment_01.png")

    with pytest.raises(ValueError, match="keyframe dimensions do not match"):
        validate_existing_keyframes(one_shot_plan()["video_prompts"], str(tmp_path))


def test_local_generators_honor_explicit_cut_start_frame(tmp_path):
    output_dir = tmp_path / "output"
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True)
    start = frames_dir / "segment_02_start.png"
    end = frames_dir / "segment_02.png"
    start.write_bytes(b"start")
    end.write_bytes(b"end")
    prompt = {
        "segment": 2,
        "prompt": "new scene",
        "first_frame": start.name,
        "last_frame": end.name,
    }

    with patch("pipeline.run_command") as run_command:
        generate_single_video_segment(
            "wan", {"total_gpus": 1}, prompt, str(output_dir), "model"
        )
        parallel_command = run_command.call_args.args[0]
        assert parallel_command[parallel_command.index("--first_frame") + 1] == str(
            start
        )

        generate_video_segments_sequential(
            "wan", {"gpu_count": 1}, [prompt], str(output_dir), "model"
        )
        sequential_command = run_command.call_args.args[0]
        assert sequential_command[sequential_command.index("--first_frame") + 1] == str(
            start
        )


def test_api_worker_uploads_storyboard_and_skips_video_generation():
    plan = one_shot_plan()
    config = {
        "keyframes_only": True,
        "enhanced_prompt": plan,
        "gcs_bucket": "test-bucket",
    }
    token = CancellationToken("storyboard-job")
    queue = Mock()

    with patch("pipeline.enhance_prompt_data", return_value=plan), \
         patch(
             "workers.video_worker.generate_keyframes_with_progress",
             return_value=["frame.png"],
         ), \
         patch("workers.video_worker.generate_video_segments_with_progress") as video, \
         patch(
             "workers.gcs_uploader.create_keyframe_storyboard_archive",
             return_value="/tmp/keyframe_storyboard.zip",
         ), \
         patch(
             "workers.gcs_uploader.upload_named_job_artifact",
             return_value="gs://test-bucket/job/keyframe_storyboard.zip",
         ) as upload:
        result = execute_pipeline_with_config(
            job_id="storyboard-job",
            prompt="reviewed plan",
            config=config,
            cancellation_token=token,
            job_queue=queue,
            processes=[],
        )

    assert result == "gs://test-bucket/job/keyframe_storyboard.zip"
    video.assert_not_called()
    assert upload.call_args.kwargs["artifact_name"] == "keyframe_storyboard.zip"
