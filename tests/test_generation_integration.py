"""Offline acceptance runs real orchestration, storage, probes and provider call accounting."""

import copy
import json
import subprocess
from datetime import UTC
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from api.contracts.generation_v1 import canonical_bytes, seal, validate
from api.generation_service import Conflict, GenerationService
from api.routes.generation import router
from video_generator_interface import (
    BillingObservation,
    VideoGenerationError,
    recorded_generation,
    set_generation_details,
)


class FakeProvider:
    calls = []
    hook = None

    def __init__(self, name):
        self.name = name

    def get_capabilities(self):
        return {
            "model": self.name,
            "allowed_durations": [3] if self.name == "long" else [1],
            "supports_image_to_video": True,
            "supports_audio": False,
        }

    def estimate_cost(self, duration):
        return 0.1

    @recorded_generation("offline-fixture")
    def generate_video(self, prompt, input_image_path, output_path, duration=1, **kwargs):
        self.calls.append(self.name)
        set_generation_details(
            model=self.name,
            parameters={"duration_s": duration, "fps": 10},
            seed=7,
            provider_request_id=f"fake-{len(self.calls)}",
            billing=BillingObservation(
                raw_unit_name="fixture_ticks", raw_units="12", estimated_usd=0.1, actual_usd=0.1
            ),
        )
        if FakeProvider.hook:
            FakeProvider.hook()
        if self.name == "failing":
            raise VideoGenerationError("Synthetic provider failure")
        color = "gray"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color={color}:s=320x180:r=10:d={duration}",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=48000:cl=stereo",
                "-t",
                str(duration),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-y",
                output_path,
            ],
            check=True,
        )
        return output_path


def keyframe(prompt, path, reference):
    Image.new("RGB", (320, 180), "white").save(path)
    return str(path)


def request():
    doc = json.loads(
        (
            Path(__file__).parents[1] / "api/contracts/v1_0/fixtures/GenerationRequest.valid.json"
        ).read_text()
    )
    template = doc["scenes"][0]
    doc.update(
        request_id="pilot-request",
        idempotency_key="pilot-request-key",
        project_id="offline-pilot",
        revision=1,
        supersedes_request_id=None,
        regeneration=None,
    )
    doc["scenes"] = []
    for i, provider in enumerate(["long", "short", "failing"]):
        scene = copy.deepcopy(template)
        scene.update(
            scene_id=f"scene-{i+1}",
            order=i + 1,
            requested_duration_s=3 if i < 2 else 1,
            reference_assets=[],
        )
        scene["continuity"] = {"mode": "independent", "required_reference_asset_ids": []}
        scene["generation_policy"].update(
            provider_preferences=[provider], max_attempts=8, max_estimated_cost_usd=10.0
        )
        scene["delivery"].update(target_width=320, target_height=180)
        doc["scenes"].append(scene)
    return seal(doc)


def approval(plan, mode="video"):
    variants = []
    for scene in plan["scenes"]:
        variants.append(scene["variants"][0]["variant_id"])
        if scene["scene_id"] == "scene-3":
            variants.append(
                next(v["variant_id"] for v in scene["variants"] if v["provider"] == "short")
            )
    return seal(
        {
            "contract_version": "1.0",
            "approval_id": "approval-" + mode,
            "request_id": plan["request_id"],
            "plan_id": plan["plan_id"],
            "plan_sha256": plan["document_sha256"],
            "created_at": "2026-09-22T00:00:00Z",
            "execution_mode": mode,
            "approved_variant_ids": variants,
            "approved_keyframe_result_id": None,
            "idempotency_key": "approval-key-" + mode,
            "allow_unknown_cost": True,
            "max_estimated_cost_usd": 10.0,
        }
    )


def service(tmp_path):
    FakeProvider.calls = []
    FakeProvider.hook = None
    return GenerationService(
        tmp_path,
        {"default_backend": "long", "integration_providers": ["short", "failing"]},
        lambda p, c: FakeProvider(p),
        keyframe,
    )


def test_pilot_and_regeneration(tmp_path):
    svc = service(tmp_path)
    req = request()
    plan = svc.plan(req)
    assert [len(s["variants"][0]["shots"]) for s in plan["scenes"]] == [1, 3, 1]
    assert svc.plan(req) == plan
    approved = approval(plan)
    job = svc.approve(approved)
    assert svc.approve(approved) == job
    result = svc.run(job["id"])
    validate("GenerationResult", result)
    assert result["terminal_status"] == "succeeded"
    assert FakeProvider.calls == ["long", "short", "short", "short", "failing", "short"]
    attempts = [
        a for a in result["scenes"][2]["attempts"] if "duration_s" in a["generation_parameters"]
    ]
    assert len(attempts) == 2
    assert attempts[1]["fallback_from_attempt_id"] == attempts[0]["attempt_id"]
    assert svc.run(job["id"]) == result
    assert canonical_bytes(svc.result(job["id"])) == canonical_bytes(result)
    assert len(FakeProvider.calls) == 6
    regen = copy.deepcopy(req)
    regen.update(
        request_id="regeneration",
        idempotency_key="regeneration",
        revision=2,
        supersedes_request_id=req["request_id"],
    )
    regen["scenes"] = [regen["scenes"][1]]
    regen["regeneration"] = {
        "base_result_ids": [result["result_id"]],
        "supersedes_take_ids": [result["scenes"][1]["takes"][0]["take_id"]],
        "reason": "New middle scene",
    }
    regenerated_plan = svc.plan(seal(regen))
    ap = approval(regenerated_plan)
    ap.update(approval_id="regen-approval", idempotency_key="regen-approval")
    replacement = svc.run(svc.approve(seal(ap))["id"])
    assert len(FakeProvider.calls) == 9
    assert (
        replacement["scenes"][0]["takes"][0]["supersedes_take_ids"]
        == regen["regeneration"]["supersedes_take_ids"]
    )
    assert svc.get("GenerationResult", result["result_id"]) == result


def test_approval_guards_and_conflicts(tmp_path):
    svc = service(tmp_path)
    req = request()
    plan = svc.plan(req)
    ap = approval(plan)
    for update in [{"plan_sha256": "sha256:" + "0" * 64}, {"approved_variant_ids": ["unknown"]}]:
        with pytest.raises(ValueError):
            svc.approve(seal({**ap, **update}))
    assert FakeProvider.calls == []
    svc.approve(ap)
    with pytest.raises(Conflict):
        svc.approve(seal({**ap, "max_estimated_cost_usd": 9.0}))
    with pytest.raises(Conflict):
        svc.plan(seal({**req, "revision": 2}))
    expired = seal({**plan, "plan_id": "expired", "valid_until": "2000-01-01T00:00:00Z"})
    with svc.db() as db:
        svc._put(db, "GenerationPlan", expired)
    with pytest.raises(ValueError):
        svc.approve(
            seal(
                {
                    **ap,
                    "plan_id": "expired",
                    "plan_sha256": expired["document_sha256"],
                    "idempotency_key": "expired",
                }
            )
        )
    assert FakeProvider.calls == []


def test_unapproved_fallback_and_cancellation(tmp_path):
    svc = service(tmp_path)
    req = request()
    req["scenes"] = req["scenes"][2:]
    plan = svc.plan(seal(req))
    ap = approval(plan)
    ap["approved_variant_ids"] = ap["approved_variant_ids"][:1]
    result = svc.run(svc.approve(seal(ap))["id"])
    assert result["terminal_status"] == "failed"
    assert FakeProvider.calls == ["failing"]
    svc = service(tmp_path / "cancel")
    plan = svc.plan(request())
    job = svc.approve(approval(plan))
    FakeProvider.hook = lambda: svc.cancel(job["id"])
    result = svc.run(job["id"])
    FakeProvider.hook = None
    assert result["terminal_status"] == "canceled"
    assert len(result["scenes"][0]["clips"]) == 1
    assert svc.result(job["id"]) == result
    assert len(FakeProvider.calls) == 1


def test_keyframe_approval_and_http_parity(tmp_path):
    svc = service(tmp_path)
    plan = svc.plan(request())
    ap = approval(plan, "keyframes")
    job = svc.approve(ap)
    result = svc.run(job["id"])
    assert result["terminal_status"] == "succeeded"
    assert not FakeProvider.calls
    video = approval(plan)
    video["approved_keyframe_result_id"] = result["result_id"]
    # Fallback variants lacking reviewed keyframes cannot start paid execution.
    result_video = svc.run(svc.approve(seal(video))["id"])
    assert result_video["terminal_status"] == "partial"
    app = FastAPI()
    app.state.generation_service = svc
    app.include_router(router)
    with TestClient(app) as client:
        response = client.get("/v2/plans/" + plan["plan_id"])
        assert response.content == canonical_bytes(plan)
        assert client.get("/v2/jobs/" + job["id"] + "/result").content == canonical_bytes(result)
        assert client.post("/v2/plans", json=request()).content == canonical_bytes(plan)


def test_golden_contracts():
    directory = Path(__file__).parents[1] / "api/contracts/v1_0/fixtures"
    for path in directory.glob("*.json"):
        kind, outcome, _ = path.name.split(".")
        if outcome == "valid":
            validate(kind, json.loads(path.read_text()))
        else:
            with pytest.raises(ValueError):
                validate(kind, json.loads(path.read_text()))


def test_budget_unknown_cost_and_file_protocol(tmp_path):
    svc = service(tmp_path)
    plan = svc.plan(request())
    ap = approval(plan)
    ap["max_estimated_cost_usd"] = 0.05
    result = svc.run(svc.approve(seal(ap))["id"])
    assert result["terminal_status"] == "failed"
    assert FakeProvider.calls == []
    assert all("Estimated cost budget exhausted" in s["warnings"] for s in result["scenes"])
    import sys

    args = [sys.executable, "-m", "api.generation_handoff", "--root", str(tmp_path)]
    response = subprocess.check_output([*args, "result", result["job_id"]])
    assert response == canonical_bytes(result)
    request_file = tmp_path / "request.json"
    request_file.write_bytes(canonical_bytes(request()))
    assert subprocess.check_output([*args, "plan", str(request_file)]) == canonical_bytes(plan)
    unknown = service(tmp_path / "unknown")
    original_factory = unknown.generator_factory

    def factory(name, config):
        provider = original_factory(name, config)
        provider.estimate_cost = lambda duration: None
        return provider

    unknown.generator_factory = factory
    unknown_plan = unknown.plan(request())
    assert all(
        v["estimated_cost"]["amount"] is None for s in unknown_plan["scenes"] for v in s["variants"]
    )
    ap = approval(unknown_plan)
    ap["allow_unknown_cost"] = False
    with pytest.raises(ValueError):
        unknown.approve(seal(ap))
    assert FakeProvider.calls == []


def test_standalone_sidecar_retains_attempts_and_source_clips(tmp_path):
    from api.generation_ledger import LegacyLedger, active_ledger

    ledger = LegacyLedger(tmp_path, "standalone")
    token = active_ledger.set(ledger)
    frame = tmp_path / "first.png"
    keyframe("", frame, None)
    output = tmp_path / "output.mp4"
    try:
        with pytest.raises(VideoGenerationError):
            FakeProvider("failing").generate_video("prompt", str(frame), str(output), duration=1)
        FakeProvider("long").generate_video("prompt", str(frame), str(output), duration=1)
    finally:
        active_ledger.reset(token)
    ledger.finish(preview=output)
    result = json.loads((ledger.root / "result.json").read_text())
    validate("GenerationResult", result)
    assert len(result["scenes"][0]["attempts"]) == 2
    assert result["preview_assets"][0]["role"] == "scene_preview"
    assert (
        result["scenes"][0]["attempts"][1]["fallback_from_attempt_id"]
        == result["scenes"][0]["attempts"][0]["attempt_id"]
    )
    output.unlink()
    from urllib.parse import urlsplit

    assert Path(urlsplit(result["scenes"][0]["clips"][0]["asset"]["uri"]).path).is_file()


def test_changed_capabilities_and_image_charge_stop_before_video(tmp_path):
    from datetime import datetime

    from video_generator_interface import GenerationOutput

    svc = service(tmp_path)
    plan = svc.plan(request())
    ap = approval(plan)
    factory = svc.generator_factory

    def changed(name, config):
        generator = factory(name, config)
        original = generator.get_capabilities
        generator.get_capabilities = lambda: {**original(), "allowed_durations": [7]}
        return generator

    svc.generator_factory = changed
    with pytest.raises(ValueError, match="capabilities"):
        svc.approve(ap)
    assert FakeProvider.calls == []
    svc.generator_factory = factory

    def charged_frame(prompt, path, reference):
        keyframe(prompt, path, reference)
        return GenerationOutput(
            str(path),
            "fixture",
            "image",
            None,
            None,
            None,
            {},
            prompt,
            datetime.now(UTC),
            datetime.now(UTC),
            BillingObservation(actual_usd=0.2),
        )

    svc.keyframe_generator = charged_frame
    ap["max_estimated_cost_usd"] = 0.15
    result = svc.run(svc.approve(seal(ap))["id"])
    assert FakeProvider.calls == []
    assert result["scenes"][0]["attempts"][0]["billing"]["actual_usd"] == 0.2
    assert not any(s["clips"] for s in result["scenes"])
    external = tmp_path.parent / "outside.png"
    keyframe("", external, None)
    from api.generation_service import file_hash

    asset = {"uri": external.as_uri(), "sha256": file_hash(external)}
    with pytest.raises(ValueError, match="outside configured"):
        svc._materialize(asset, tmp_path)
    svc.config["integration_asset_roots"] = [str(external.parent)]
    snapshot = Path(svc._materialize(asset, tmp_path))
    assert snapshot.is_relative_to(tmp_path) and snapshot != external
    external.unlink()
    assert snapshot.is_file() and file_hash(snapshot) == asset["sha256"]


def test_terminal_publication_recovers_without_provider_replay(tmp_path, monkeypatch):
    svc = service(tmp_path)
    job = svc.approve(approval(svc.plan(request())))
    publish = svc._publish_result

    def fail_publication(result):
        raise OSError("Simulated crash after terminal commit")

    monkeypatch.setattr(svc, "_publish_result", fail_publication)
    with pytest.raises(OSError):
        svc.run(job["id"])
    saved = svc.result(job["id"])
    assert saved["terminal_status"] == "succeeded"
    assert len(FakeProvider.calls) == 6
    monkeypatch.setattr(svc, "_publish_result", publish)
    assert svc.run(job["id"]) == saved
    assert len(FakeProvider.calls) == 6
    assert not svc.job(job["id"])["warnings"]


def test_provider_prepared_inputs_are_retained_by_content_hash(tmp_path):
    from urllib.parse import urlsplit

    from api.generation_service import file_hash
    from video_generator_interface import replace_generation_reference

    original = tmp_path / "input.png"
    prepared = tmp_path / "prepared.png"
    Image.new("RGB", (320, 180), "white").save(original)
    Image.new("RGB", (160, 90), "gray").save(prepared)

    class PreparedProvider:
        @recorded_generation("prepared-fixture")
        def generate_video(self, prompt, input_image_path, output_path, duration=1, **kwargs):
            replace_generation_reference(input_image_path, prepared)
            return output_path

    observed = PreparedProvider().generate_video("prompt", str(original), "unused")
    assert observed.reference_paths == (str(prepared),)
    svc = service(tmp_path / "service")
    result = {"reference_assets": [], "scenes": []}
    ids = svc._provider_references(observed, [], result, "fixture")
    assert ids == [result["reference_assets"][0]["asset_id"]]
    asset = result["reference_assets"][0]
    assert asset["sha256"] == file_hash(prepared) != file_hash(original)
    retained = Path(urlsplit(asset["uri"]).path)
    prepared.unlink()
    assert retained.is_file() and file_hash(retained) == asset["sha256"]


def test_provider_receipt_is_durable_before_completion_and_deduplicates(tmp_path):
    svc = service(tmp_path)
    frame = tmp_path / "frame.png"
    keyframe("", frame, None)
    captured = []

    def crash_after_submission():
        with svc.db() as db:
            captured.append(json.loads(db.execute("SELECT body FROM provider_calls").fetchone()[0]))
        raise KeyboardInterrupt("Synthetic process interruption")

    FakeProvider.hook = crash_after_submission
    try:
        with pytest.raises(KeyboardInterrupt):
            with svc._provider_call("job", "shot", "video") as operation:
                FakeProvider("long").generate_video(
                    "prompt", str(frame), str(tmp_path / "video.mp4")
                )
        assert captured[0]["provider_request_id"] == "fake-1"
        assert captured[0]["attempt_id"] == operation
        assert captured[0]["model"] == "long"
        with pytest.raises(Conflict, match="provider request already exists"):
            with svc._provider_call("job", "shot", "video"):
                pytest.fail("A known provider operation must not be submitted again")
        assert FakeProvider.calls == ["long"]
    finally:
        FakeProvider.hook = None


def test_provider_image_preparation_preserves_true_jpeg_type(tmp_path):
    from generators.base import ImageValidator

    original = tmp_path / "rgba.png"
    Image.new("RGBA", (32, 32), "white").save(original)
    first = Path(ImageValidator.prepare_image_for_api(str(original)))
    second = Path(ImageValidator.prepare_image_for_api(str(original)))
    try:
        assert first != second and first.suffix == ".jpg"
        with Image.open(first) as prepared:
            assert prepared.format == "JPEG" and prepared.mode == "RGB"
    finally:
        first.unlink()
        second.unlink()


def test_http_queue_and_both_workers_use_the_same_durable_service(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import trio

    import api.config
    import api.generation_service
    import workers.trio_video_worker as trio_worker
    import workers.video_worker as standard_worker

    svc = service(tmp_path)
    ap = approval(svc.plan(request()))
    jobs = {}
    statuses = []

    class Queue:
        def get_job(self, job_id):
            return jobs.get(job_id)

        def enqueue_job(self, request, config, job_id):
            assert job_id not in jobs
            jobs[job_id] = SimpleNamespace(config=config)

        def update_job_status(self, job_id, status, **fields):
            statuses.append((job_id, status.value, fields))

    queue = Queue()
    app = FastAPI()
    app.state.generation_service = svc
    app.state.job_queue = queue
    app.include_router(router)
    with TestClient(app) as client:
        created = client.post("/v2/jobs", json=ap)
        assert created.status_code == 200
        job_id = created.json()["id"]
        assert client.post("/v2/jobs", json=ap).content == created.content
    assert len(jobs) == 1
    monkeypatch.setattr(api.generation_service, "GenerationService", lambda root, config: svc)
    monkeypatch.setattr(api.config, "restore_job_config_secrets", lambda config: config)
    monkeypatch.setattr(standard_worker, "ensure_queue_initialized", lambda: None)
    monkeypatch.setattr(standard_worker, "get_job_queue", lambda: queue)
    monkeypatch.setattr(trio_worker, "get_job_queue", lambda: queue)
    uri = standard_worker.process_video_job(job_id)
    assert svc.result(job_id)["terminal_status"] == "succeeded"
    assert len(FakeProvider.calls) == 6
    assert trio.run(trio_worker.process_video_job_trio, job_id) == uri
    assert len(FakeProvider.calls) == 6
    assert [s[1] for s in statuses] == ["finished", "finished"]


def test_file_execution_stdout_is_only_the_canonical_result(tmp_path, monkeypatch, capsys):
    import api.generation_handoff as handoff

    svc = service(tmp_path)
    job = svc.approve(approval(svc.plan(request())))

    def noisy_keyframe(prompt, path, reference):
        print("Provider keyframe progress")
        return keyframe(prompt, path, reference)

    svc.keyframe_generator = noisy_keyframe
    monkeypatch.setattr(handoff, "GenerationService", lambda root, config: svc)
    handoff.main(["--root", str(tmp_path), "run", job["id"]])
    streams = capsys.readouterr()
    assert streams.out.encode() == canonical_bytes(svc.result(job["id"]))
    assert "Provider keyframe progress" in streams.err
