"""Version 2 generation routes; v1 routes and response shapes remain unchanged."""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from api.contracts.generation_v1 import canonical_bytes
from api.generation_service import Conflict, GenerationService

router = APIRouter(prefix="/v2", tags=["generation"])


def service(request):
    injected = getattr(request.app.state, "generation_service", None)
    if injected:
        return injected
    config = getattr(request.app.state, "config", None)
    if not config:
        raise HTTPException(503, "Pipeline configuration unavailable")
    pipeline = dict(config.pipeline_config)
    pipeline.update(
        gcs_bucket=(config.gcs.bucket if pipeline.get("integration_publish_gcs", True) else None),
        gcs_prefix=config.gcs.prefix,
        credentials_path=config.gcs.credentials_path,
    )
    root = pipeline.get("integration_root") or os.getenv(
        "TTV_INTEGRATION_ROOT", "./generation-data"
    )
    return GenerationService(root, pipeline)


def response(action):
    try:
        return Response(canonical_bytes(action()), media_type="application/json")
    except Conflict as error:
        raise HTTPException(409, str(error)) from error
    except KeyError as error:
        raise HTTPException(404, "Document or job not found") from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.get("/capabilities")
def capabilities(request: Request):
    return response(lambda: service(request).capabilities())


@router.get("/schemas/{name}")
def schema(name: str):
    root = Path(__file__).parents[1] / "contracts" / "v1_0"
    if name not in {"schema-hashes.json", *(p.name for p in root.glob("*.schema.json"))}:
        raise HTTPException(404, "Schema not found")
    return Response((root / name).read_bytes(), media_type="application/schema+json")


@router.post("/plans")
def plan(request: Request, body: dict):
    return response(lambda: service(request).plan(body))


@router.get("/plans/{plan_id}")
def get_plan(request: Request, plan_id: str):
    return response(lambda: service(request).get("GenerationPlan", plan_id))


@router.post("/jobs")
def approve(request: Request, body: dict):
    def submit():
        from api.models import JobCreateRequest

        svc = service(request)
        queue = getattr(request.app.state, "job_queue", None)
        if not queue:
            raise HTTPException(503, "Job queue unavailable")
        job = svc.approve(body)
        if job["status"] == "queued" and not queue.get_job(job["id"]):
            config = {**svc.config, "integration_root": str(svc.root), "generation_v2": True}
            queue.enqueue_job(
                JobCreateRequest(prompt="Approved generation plan"), config, job_id=job["id"]
            )
        return job

    return response(submit)


@router.get("/jobs/{job_id}")
def job(request: Request, job_id: str):
    return response(lambda: service(request).job(job_id))


@router.get("/jobs/{job_id}/result")
def result(request: Request, job_id: str):
    return response(lambda: service(request).result(job_id))


@router.post("/jobs/{job_id}/cancel")
def cancel(request: Request, job_id: str):
    def action():
        svc = service(request)
        svc.cancel(job_id)
        return svc.job(job_id)

    return response(action)
