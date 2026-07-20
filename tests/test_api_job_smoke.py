import inspect
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from api.config import APIConfig, GCSConfig
from api.main import create_app
from api.models import JobStatus
from api.routes.jobs import create_plan
from tests.mocks.mock_redis import MockJobQueue, MockRedisManager
from workers.video_worker import process_video_job


def reviewed_plan():
    return {
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


def test_api_creates_and_resumes_reviewed_plan():
    pipeline_config = {
        "default_backend": "veo3",
        "generation_mode": "keyframe",
        "single_keyframe_mode": True,
        "openai_api_key": "test-secret",
    }
    api_config = APIConfig(
        gcs=GCSConfig(bucket="test-bucket"),
        pipeline_config=pipeline_config,
    )
    queue = MockJobQueue(MockRedisManager(api_config.redis))
    app = create_app()
    app.state.config = api_config
    app.state.job_queue = queue
    client = TestClient(app)
    plan = reviewed_plan()
    long_prompt = "# Brief\n\n" + "A" * 3000

    with patch("pipeline.enhance_prompt_data", return_value=plan) as enhance:
        plan_response = client.post(
            "/v1/plans",
            json={"prompt": long_prompt, "duration_seconds": 4},
        )

    assert plan_response.status_code == 200
    assert plan_response.json() == plan
    assert enhance.call_args.args[0] == long_prompt
    assert enhance.call_args.args[1]["duration_seconds"] == 4

    create_response = client.post(
        "/v1/jobs",
        json={"duration_seconds": 4, "enhanced_prompt": plan},
    )
    assert create_response.status_code == 202
    stored_job = queue.get_job(create_response.json()["id"])
    assert stored_job.prompt == "Resumed from reviewed prompt plan"
    assert stored_job.config["enhanced_prompt"] == plan

    storyboard_response = client.post(
        "/v1/jobs",
        json={
            "duration_seconds": 4,
            "enhanced_prompt": plan,
            "keyframes_only": True,
        },
    )
    assert storyboard_response.status_code == 202
    storyboard_job = queue.get_job(storyboard_response.json()["id"])
    assert storyboard_job.config["keyframes_only"] is True


def test_api_rejects_reviewed_plan_frame_paths():
    app = create_app()
    client = TestClient(app)
    plan = reviewed_plan()
    plan["video_prompts"][0]["first_frame"] = "/tmp/private.png"

    response = client.post("/v1/jobs", json={"enhanced_prompt": plan})

    assert response.status_code == 422
    assert "Frame references must be filenames" in response.text


def test_storyboard_job_exposes_generic_artifact_metadata():
    api_config = APIConfig(
        gcs=GCSConfig(bucket="test-bucket"),
        pipeline_config={
            "default_backend": "veo3",
            "generation_mode": "keyframe",
            "single_keyframe_mode": True,
        },
    )
    queue = MockJobQueue(MockRedisManager(api_config.redis))
    app = create_app()
    app.state.config = api_config
    app.state.job_queue = queue
    client = TestClient(app)

    assert not inspect.iscoroutinefunction(create_plan)

    response = client.post(
        "/v1/jobs",
        json={"enhanced_prompt": reviewed_plan(), "keyframes_only": True},
    )
    job_id = response.json()["id"]
    gcs_uri = f"gs://test-bucket/ttv-api/{job_id}/keyframe_storyboard.zip"
    queue.update_job_status(job_id, JobStatus.FINISHED, gcs_uri=gcs_uri)
    gcs_client = Mock()
    gcs_client.generate_signed_url.return_value = "https://example.com/storyboard"

    with patch("api.gcs_client.create_gcs_client", return_value=gcs_client):
        artifact = client.get(f"/v1/jobs/{job_id}/artifact-url")

    assert artifact.status_code == 200
    assert artifact.json()["artifact_name"] == "keyframe_storyboard.zip"
    assert artifact.json()["mime_type"] == "application/zip"
    assert artifact.json()["artifact_url"] == "https://example.com/storyboard"
    assert "video_url" not in artifact.json()

    queue.update_job_status(
        job_id,
        JobStatus.FINISHED,
        gcs_uri=f"gs://test-bucket/ttv-api/{job_id}/final_video.mp4",
    )
    with patch("api.gcs_client.create_gcs_client", return_value=gcs_client):
        video = client.get(f"/v1/jobs/{job_id}/video-url")
    assert video.json()["mime_type"] == "video/mp4"
    assert video.json()["video_url"] == "https://example.com/storyboard"


def test_mocked_api_job_reaches_worker_with_effective_config():
    pipeline_config = {
        "prompt": "default prompt",
        "default_backend": "veo3",
        "generation_mode": "keyframe",
        "single_keyframe_mode": True,
        "image_generation_model": "openai/gpt-image-1",
        "openai_api_key": "test-secret",
        "minimax": {"api_key": "nested-secret", "model": "I2V-01-Director"},
    }
    api_config = APIConfig(
        gcs=GCSConfig(bucket="test-bucket", credentials_path="test-credentials.json"),
        pipeline_config=pipeline_config,
    )
    queue = MockJobQueue(MockRedisManager(api_config.redis))
    app = create_app()
    app.state.config = api_config
    app.state.job_queue = queue
    client = TestClient(app)

    with patch.object(queue, "enqueue_job", wraps=queue.enqueue_job) as enqueue_job:
        create_response = client.post(
            "/v1/jobs",
            json={"prompt": "HTTP prompt", "duration_seconds": 9},
        )
    assert create_response.status_code == 202
    assert enqueue_job.call_args.kwargs["job_timeout"] == 4800
    assert "ending keyframe will not appear" in create_response.json()["warnings"][0]
    job_id = create_response.json()["id"]
    stored_config = queue.get_job(job_id).config.copy()
    assert stored_config == {
        "default_backend": "veo3",
        "generation_mode": "keyframe",
        "single_keyframe_mode": True,
        "image_generation_model": "openai/gpt-image-1",
        "openai_api_key": "[REDACTED]",
        "minimax": {"api_key": "[REDACTED]", "model": "I2V-01-Director"},
        "prompt": "HTTP prompt",
        "duration_seconds": 9,
        "gcs_bucket": "test-bucket",
        "gcs_prefix": "ttv-api",
        "credentials_path": "[REDACTED]",
        "signed_url_expiration": 3600,
    }
    assert "secret" not in str(stored_config)

    gcs_uri = f"gs://test-bucket/ttv-api/{job_id}/final_video.mp4"
    with patch("api.config.get_config_from_env", return_value=api_config), \
         patch("workers.video_worker.ensure_queue_initialized"), \
         patch("workers.video_worker.get_job_queue", return_value=queue), \
         patch("workers.video_worker.execute_pipeline_with_config", return_value=gcs_uri) as execute, \
         patch("workers.video_worker._record_job_metrics"):
        assert process_video_job(job_id, use_trio=False) == gcs_uri

    assert execute.call_args.kwargs["config"] == {
        **pipeline_config,
        "prompt": "HTTP prompt",
        "duration_seconds": 9,
        "gcs_bucket": "test-bucket",
        "gcs_prefix": "ttv-api",
        "credentials_path": "test-credentials.json",
        "signed_url_expiration": 3600,
    }
    status_response = client.get(f"/v1/jobs/{job_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == JobStatus.FINISHED
    assert status_response.json()["gcs_uri"] == gcs_uri
    assert client.get("/jobs").status_code == 404
