"""Provenance sidecar for the independently usable standalone and v1 pipelines."""

from __future__ import annotations

import inspect
import json
import re
import shutil
from contextvars import ContextVar
from dataclasses import asdict
from datetime import UTC
from functools import wraps
from pathlib import Path

from api.contracts.generation_v1 import canonical_bytes, seal, validate
from api.generation_service import file_hash, now, probe, uid

active_ledger = ContextVar("active_generation_ledger", default=None)


class LegacyLedger:
    def __init__(self, root, job_id=None):
        self.job_id = job_id or uid()
        self.root = Path(root).resolve() / self.job_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.events = []
        self.expected_segments = set()

    def asset(self, source, role):
        source = Path(source)
        sha = file_hash(source)
        destination = self.root / "media" / (sha[7:] + source.suffix)
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        import mimetypes

        return {
            "asset_id": uid(),
            "uri": destination.as_uri(),
            "sha256": sha,
            "mime_type": mimetypes.guess_type(str(source))[0] or "application/octet-stream",
            "role": role,
            "rights_note": None,
        }

    def record(self, output, error, references=()):
        from video_generator_interface import cleanup_prepared_reference

        references = output.reference_paths if output.reference_paths is not None else references
        captured = []
        for path in references:
            if path and Path(path).is_file():
                captured.append(self.asset(path, "provider_reference"))
                cleanup_prepared_reference(path)
        references = captured
        event = {
            "id": uid(),
            "segment": (
                int(match.group(1))
                if (match := re.search(r"segment_(\d+)", Path(output.path).name))
                else None
            ),
            "output": asdict(output),
            "error": type(error).__name__ if error else None,
            "references": references,
            "asset": None,
            "media": None,
        }
        event["output"]["started_at"] = output.started_at.isoformat().replace("+00:00", "Z")
        event["output"]["finished_at"] = output.finished_at.isoformat().replace("+00:00", "Z")
        if not error and Path(output.path).is_file():
            event["asset"] = self.asset(
                output.path, "generated_clip" if Path(output.path).suffix == ".mp4" else "keyframe"
            )
            try:
                event["media"] = probe(output.path)
            except ValueError, StopIteration, OSError:
                pass
        (self.root / (event["id"] + ".event.json")).write_bytes(canonical_bytes(event))
        self.events.append(event)

    def finish(self, error=None, preview=None):
        self.events = [json.loads(p.read_text()) for p in self.root.glob("*.event.json")]
        self.events.sort(
            key=lambda e: (
                e["segment"] if e.get("segment") is not None else float("inf"),
                e["output"]["started_at"],
                e["id"],
            )
        )
        if not self.events:
            return
        attempts = []
        shots = []
        clips = []
        frames = []
        references = []
        previous_failure = {}
        shot_by_path = {}
        for event in self.events:
            output = event["output"]
            sid = shot_by_path.setdefault(output["path"], event["id"])
            references.extend(event["references"])
            parameters = output["parameters"]
            duration = parameters.get(
                "duration_s", parameters.get("duration_seconds", parameters.get("duration", 1))
            )
            try:
                duration = float(str(duration).removesuffix("s"))
            except ValueError:
                duration = 1
            duration = max(0.001, duration)
            if sid == event["id"]:
                shots.append(
                    {
                        "shot_id": sid,
                        "order": len(shots),
                        "nominal_duration_s": duration,
                        "continuity_mode": "cut",
                    }
                )
            billing = output["billing"] or {
                "raw_unit_name": None,
                "raw_units": None,
                "estimated_usd": None,
                "actual_usd": None,
                "currency": "USD",
            }
            attempt = {
                "attempt_id": event["id"],
                "shot_id": sid,
                "attempt_number": len(attempts) + 1,
                "provider": output["provider"],
                "model": output["model"],
                "model_version": output["model_version"],
                "provider_request_id": output["provider_request_id"],
                "prompt_snapshot": output["prompt"],
                "reference_asset_ids": [r["asset_id"] for r in event["references"]],
                "generation_parameters": parameters,
                "seed": output["seed"],
                "started_at": output["started_at"],
                "finished_at": output["finished_at"],
                "status": "failed" if event["error"] else "succeeded",
                "fallback_from_attempt_id": previous_failure.get(output["path"]),
                "error": (
                    {
                        "code": event["error"],
                        "message": "Provider call failed; see private worker diagnostics.",
                    }
                    if event["error"]
                    else None
                ),
                "billing": billing,
            }
            attempts.append(attempt)
            if event["error"]:
                previous_failure[output["path"]] = attempt["attempt_id"]
            if event["asset"] and event["media"]:
                if event["asset"]["mime_type"].startswith("video/"):
                    clips.append(
                        {
                            "clip_id": uid(),
                            "attempt_id": attempt["attempt_id"],
                            "asset": event["asset"],
                            "media": event["media"],
                        }
                    )
                else:
                    frames.append(
                        {
                            "shot_id": sid,
                            "order": len(frames),
                            "position": "last",
                            "attempt_id": attempt["attempt_id"],
                            "asset": event["asset"],
                            "media": event["media"],
                        }
                    )
        succeeded = {
            event["segment"] if event["segment"] is not None else event["output"]["path"]
            for event in self.events if event["asset"]
        }
        failed = {
            event["segment"] if event["segment"] is not None else event["output"]["path"]
            for event in self.events if event["error"]
        }
        video_segments = {
            event["segment"] for event in self.events
            if event["asset"] and event["asset"]["mime_type"].startswith("video/")
        }
        incomplete = bool((failed - succeeded) or (self.expected_segments - video_segments))
        canceled = isinstance(error, InterruptedError)
        status = (
            "canceled"
            if canceled
            else "partial" if (error or incomplete) and (clips or frames)
            else "failed" if error or incomplete else "succeeded"
        )
        takes = []
        previews = []
        if clips and all(c["media"]["measured_duration_s"] for c in clips) and status == "succeeded":
            if preview and Path(str(preview)).is_file():
                previews.append(self.asset(preview, "scene_preview"))
            total = sum(c["media"]["measured_duration_s"] for c in clips)
            takes.append(
                {
                    "take_id": uid(),
                    "clip_ids": [c["clip_id"] for c in clips],
                    "clip_order": {c["clip_id"]: i for i, c in enumerate(clips)},
                    "preview_asset_id": previews[0]["asset_id"] if previews else None,
                    "supersedes_take_ids": [],
                    "depends_on_take_ids": [],
                    "nominal_duration_s": total,
                    "measured_duration_s": total,
                }
            )
        result = seal(
            {
                "contract_version": "1.0",
                "result_id": uid(),
                "result_kind": "video" if clips else "keyframes",
                "request_id": self.job_id,
                "plan_id": self.job_id,
                "approval_id": self.job_id,
                "job_id": self.job_id,
                "created_at": now(),
                "terminal_status": status,
                "reference_assets": references,
                "preview_assets": previews,
                "scenes": [
                    {
                        "scene_id": self.job_id,
                        "order": 0,
                        "status": status,
                        "actual_shots": shots,
                        "attempts": attempts,
                        "clips": clips,
                        "keyframes": frames,
                        "takes": takes,
                        "warnings": [
                            "Standalone/v1 execution provenance; authorization is the original invocation."
                        ],
                    }
                ],
            }
        )
        validate("GenerationResult", result)
        (self.root / "result.json").write_bytes(canonical_bytes(result))


def legacy_provenance(function):
    """Use one collector across the standard/Trio wrapper; v2 has its own ledger."""
    signature = inspect.signature(function)

    def setup(args, kwargs):
        values = signature.bind(*args, **kwargs).arguments
        config = values.get("config", {})
        ledger = LegacyLedger(
            config.get("integration_root", "./generation-data") + "/legacy", values.get("job_id")
        )
        return ledger, active_ledger.set(ledger)

    if inspect.iscoroutinefunction(function):

        @wraps(function)
        async def async_wrapper(*args, **kwargs):
            if active_ledger.get():
                return await function(*args, **kwargs)
            ledger, token = setup(args, kwargs)
            error = None
            result = None
            try:
                result = await function(*args, **kwargs)
                return result
            except BaseException as caught:
                error = caught
                raise
            finally:
                try:
                    ledger.finish(error, result)
                finally:
                    active_ledger.reset(token)

        return async_wrapper

    @wraps(function)
    def wrapper(*args, **kwargs):
        if active_ledger.get():
            return function(*args, **kwargs)
        ledger, token = setup(args, kwargs)
        error = None
        result = None
        try:
            result = function(*args, **kwargs)
            return result
        except BaseException as caught:
            error = caught
            raise
        finally:
            try:
                ledger.finish(error, result)
            finally:
                active_ledger.reset(token)

    return wrapper


def recorded_local_command(function):
    @wraps(function)
    def wrapper(cmd, *args, **kwargs):
        ledger = active_ledger.get()
        if not ledger or "--task" not in cmd or "--save_file" not in cmd:
            return function(cmd, *args, **kwargs)
        from datetime import datetime

        from video_generator_interface import BillingObservation, GenerationOutput

        started = datetime.now(UTC)
        fields = {
            cmd[i][2:]: cmd[i + 1]
            for i in range(len(cmd) - 1)
            if cmd[i].startswith("--") and not cmd[i + 1].startswith("--")
        }
        error = None
        try:
            return function(cmd, *args, **kwargs)
        except BaseException as caught:
            error = caught
            raise
        finally:
            output = GenerationOutput(
                path=fields["save_file"],
                provider="wan2.1",
                model=fields.get("task"),
                model_version=None,
                provider_request_id=None,
                seed=None,
                parameters={
                    k: v
                    for k, v in fields.items()
                    if k
                    in {
                        "task",
                        "size",
                        "frame_num",
                        "sample_guide_scale",
                        "sample_steps",
                        "sample_shift",
                    }
                },
                prompt=fields.get("prompt", ""),
                started_at=started,
                finished_at=datetime.now(UTC),
                billing=BillingObservation(estimated_usd=0.0),
            )
            ledger.record(output, error, [fields.get("first_frame"), fields.get("last_frame")])

    return wrapper
