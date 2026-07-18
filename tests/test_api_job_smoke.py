from unittest.mock import patch

from fastapi.testclient import TestClient

from api.config import APIConfig, GCSConfig
from api.main import create_app
from api.models import JobStatus
from tests.mocks.mock_redis import MockJobQueue, MockRedisManager
from workers.video_worker import process_video_job


def test_mocked_api_job_reaches_worker_with_effective_config():
    pipeline_config = {
        "prompt": "default prompt",
        "default_backend": "fal",
        "image_generation_model": "openai/gpt-image-1",
        "openai_api_key": "test-secret",
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

    create_response = client.post("/v1/jobs", json={"prompt": "HTTP prompt"})
    assert create_response.status_code == 202
    job_id = create_response.json()["id"]
    stored_config = queue.get_job(job_id).config.copy()
    assert stored_config == {
        **pipeline_config,
        "prompt": "HTTP prompt",
        "gcs_bucket": "test-bucket",
        "gcs_prefix": "ttv-api",
        "credentials_path": "test-credentials.json",
        "signed_url_expiration": 3600,
    }

    gcs_uri = f"gs://test-bucket/ttv-api/{job_id}/final_video.mp4"
    with patch("workers.video_worker.ensure_queue_initialized"), \
         patch("workers.video_worker.get_job_queue", return_value=queue), \
         patch("workers.video_worker.execute_pipeline_with_config", return_value=gcs_uri) as execute, \
         patch("workers.video_worker._record_job_metrics"):
        assert process_video_job(job_id, use_trio=False) == gcs_uri

    assert execute.call_args.kwargs["config"] == stored_config
    status_response = client.get(f"/v1/jobs/{job_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == JobStatus.FINISHED
    assert status_response.json()["gcs_uri"] == gcs_uri
    assert client.get("/jobs").status_code == 404
