"""Durable generation lifecycle shared by HTTP, file handoff, and both workers."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import mimetypes
import os
import shutil
import sqlite3
import subprocess
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import unquote, urlsplit

from api.contracts.generation_v1 import canonical_bytes, seal, validate


def now():
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def uid():
    return str(uuid.uuid4())


class Conflict(ValueError):
    pass


def file_hash(path):
    with open(path, "rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def probe(path):
    data = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
        )
    )
    stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    fraction = stream.get("avg_frame_rate", "0/1").split("/")
    fps = float(fraction[0]) / float(fraction[1]) if float(fraction[1]) else None
    duration = data.get("format", {}).get("duration", stream.get("duration"))
    return {
        "measured_duration_s": float(duration) if duration else None,
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": fps or None,
        "has_audio": any(s["codec_type"] == "audio" for s in data["streams"]),
    }


class GenerationService:
    def __init__(self, root, config=None, generator_factory=None, keyframe_generator=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = copy.deepcopy(config or {})
        self.generator_factory = generator_factory
        self.keyframe_generator = keyframe_generator
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents(kind TEXT, id TEXT, body BLOB NOT NULL,
                    PRIMARY KEY(kind,id));
                CREATE TABLE IF NOT EXISTS bindings(kind TEXT, key TEXT, hash TEXT, target TEXT,
                    PRIMARY KEY(kind,key));
                CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, body BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS configurations(plan_id TEXT PRIMARY KEY, body BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY, job_id TEXT, body BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS provider_calls(id TEXT PRIMARY KEY, job_id TEXT, body BLOB NOT NULL);
            """)

    @contextmanager
    def db(self):
        connection = sqlite3.connect(self.root / "generation.sqlite3", timeout=30)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            with connection:
                yield connection
        finally:
            connection.close()

    def _put(self, db, kind, document):
        validate(kind, document)
        key = {
            "GenerationRequest": "request_id",
            "GenerationPlan": "plan_id",
            "PlanApproval": "approval_id",
            "GenerationResult": "result_id",
        }[kind]
        identifier, body = document[key], canonical_bytes(document)
        old = db.execute(
            "SELECT body FROM documents WHERE kind=? AND id=?", (kind, identifier)
        ).fetchone()
        if old and old[0] != body:
            raise Conflict("Immutable document ID already has different content")
        db.execute("INSERT OR IGNORE INTO documents VALUES(?,?,?)", (kind, identifier, body))
        return document

    def get(self, kind, identifier):
        with self.db() as db:
            row = db.execute(
                "SELECT body FROM documents WHERE kind=? AND id=?", (kind, identifier)
            ).fetchone()
        if not row:
            raise KeyError(identifier)
        return validate(kind, json.loads(row[0]))

    def _existing(self, db, kind, key, content_hash):
        old = db.execute(
            "SELECT hash,target FROM bindings WHERE kind=? AND key=?", (kind, key)
        ).fetchone()
        if old and old[0] != content_hash:
            raise Conflict("Idempotency key already binds different content")
        return old[1] if old else None

    def _generator(self, provider, config=None):
        if self.generator_factory:
            return self.generator_factory(provider, config or self.config)
        from generators.factory import create_video_generator

        return create_video_generator(provider, config or self.config)

    @staticmethod
    def _estimate_cost(generator, provider, duration, aspect_ratio):
        if provider == "runway":
            return generator.estimate_cost(duration, aspect_ratio)
        return generator.estimate_cost(duration)

    def capabilities(self):
        from pipeline import get_video_generation_backend

        providers = [get_video_generation_backend(self.config)]
        providers += self.config.get("integration_providers", [])
        fallback = self.config.get("remote_api_settings", {}).get("fallback_backend")
        if fallback:
            providers.append(fallback)
        result = []
        for provider in dict.fromkeys(providers):
            try:
                generator = self._generator(provider)
                raw = generator.get_capabilities()
                durations = raw.get("allowed_durations")
                if durations is None:
                    from pipeline import get_backend_clip_durations

                    durations = get_backend_clip_durations(
                        {
                            **self.config,
                            "default_backend": provider,
                            "default_video_generation_backend": provider,
                        }
                    )
                durations = durations or list(range(1, math.floor(raw["max_duration"]) + 1))
                model = raw.get("model") or next(
                    (
                        getattr(generator, name, None)
                        for name in ("model", "model_name", "model_version")
                        if getattr(generator, name, None)
                    ),
                    provider,
                )
                result.append(
                    {
                        "provider": provider,
                        "model": model,
                        "capability_snapshot": {
                            "allowed_durations_s": sorted(durations),
                            "supports_first_frame": raw.get("supports_image_to_video", False),
                            "supports_last_frame": raw.get("supports_first_last_frame", False),
                            "supports_audio": raw.get(
                                "supports_audio", provider in {"veo3", "fal", "fal.ai"}
                            ),
                        },
                    }
                )
            except Exception:
                continue
        return {"contract_version": "1.0", "providers": result}

    def plan(self, request):
        validate("GenerationRequest", request)
        with self.db() as db:
            old = self._existing(
                db, "request", request["idempotency_key"], request["document_sha256"]
            )
            if old:
                return self.get("GenerationPlan", old)
        regeneration = request.get("regeneration")
        if regeneration:
            target_scenes = {scene["scene_id"] for scene in request["scenes"]}
            takes = set()
            for result_id in regeneration["base_result_ids"]:
                base = self.get("GenerationResult", result_id)
                source = self.get("GenerationRequest", base["request_id"])
                if source["project_id"] != request["project_id"]:
                    raise ValueError("Regeneration base belongs to another project")
                takes.update(
                    t["take_id"]
                    for s in base["scenes"]
                    if s["scene_id"] in target_scenes
                    for t in s["takes"]
                )
            if not set(regeneration["supersedes_take_ids"]) <= takes:
                raise ValueError("Regeneration take is missing or belongs to an unaffected scene")
        from pipeline import plan_provider_segment_durations

        capabilities = self.capabilities()["providers"]
        plan = {
            "contract_version": "1.0",
            "plan_id": uid(),
            "request_id": request["request_id"],
            "created_at": now(),
            "valid_until": (datetime.now(UTC) + timedelta(hours=24))
            .isoformat()
            .replace("+00:00", "Z"),
            "status": "ready",
            "scenes": [],
            "blocked_reasons": [],
        }
        previous_scene_ready = False
        for scene in sorted(request["scenes"], key=lambda s: s["order"]):
            policy = scene["generation_policy"]
            refs = {a["asset_id"] for a in scene["reference_assets"]}
            if not set(scene["continuity"]["required_reference_asset_ids"]) <= refs:
                raise ValueError("Required continuity reference is missing")
            variants, warnings = [], []
            has_first = any(a["role"] == "first_frame" for a in scene["reference_assets"])
            can_inherit = (
                scene["continuity"]["mode"] == "continue_from_previous"
                and previous_scene_ready
            )
            if not has_first and not can_inherit and not (
                self.config.get("image_generation_model") or self.keyframe_generator
            ):
                warnings.append("A first frame requires a configured image generation model.")
            if self.config.get("image_generation_model"):
                warnings.append(
                    f"Keyframe model: {self.config['image_generation_model']}; image cost estimate unavailable."
                )
            prefs = policy["provider_preferences"]
            available = sorted(
                capabilities,
                key=lambda cap: (
                    prefs.index(cap["provider"]) if cap["provider"] in prefs else len(prefs)
                ),
            )
            if prefs and available and available[0]["provider"] not in prefs:
                warnings.append(
                    "Preferred provider unavailable; review the selected provider override."
                )
            if not policy["allow_provider_fallback"]:
                available = available[:1]
            for cap in available:
                if not has_first and not can_inherit and not (
                    self.config.get("image_generation_model") or self.keyframe_generator
                ):
                    continue
                if cap["provider"] == "veo3" and scene["delivery"]["aspect_ratio"] not in {
                    "16:9", "9:16"
                }:
                    continue
                if (
                    scene["delivery"]["audio_policy"] == "required"
                    and not cap["capability_snapshot"]["supports_audio"]
                ):
                    continue
                if any(not a["mime_type"].startswith("image/") for a in scene["reference_assets"]):
                    warnings.append("This provider path accepts image references only.")
                    continue
                supplied_last = any(a["role"] == "last_frame" for a in scene["reference_assets"])
                if supplied_last and not cap["capability_snapshot"]["supports_last_frame"]:
                    warnings.append("Supplied ending frame requires a compatible provider variant.")
                    continue
                durations = cap["capability_snapshot"]["allowed_durations_s"]
                if any(int(d) != d for d in durations):
                    continue
                lengths = plan_provider_segment_durations(
                    math.ceil(scene["requested_duration_s"]), tuple(map(int, durations))
                )
                variant_id = uid()
                generator = self._generator(cap["provider"])
                estimates = [
                    self._estimate_cost(
                        generator, cap["provider"], d, scene["delivery"]["aspect_ratio"]
                    )
                    for d in lengths
                ]
                cost = (
                    None
                    if any(e is None for e in estimates)
                    or (cap["provider"] in {"fal", "fal.ai"} and not any(estimates))
                    else sum(estimates)
                )
                if cost is None:
                    warnings.append(
                        "Cost estimate unavailable; execution requires explicit unknown-cost approval."
                    )
                if (
                    cost is not None
                    and policy["max_estimated_cost_usd"] is not None
                    and cost > policy["max_estimated_cost_usd"]
                ):
                    continue
                excess = sum(lengths) - scene["requested_duration_s"]
                if excess:
                    warnings.append(
                        f"Provider duration exceeds target by {excess:g} seconds; trim in the edit."
                    )
                shots = []
                for i, length in enumerate(lengths):
                    prompt = "\n".join(
                        scene["intent"][field]
                        for field in ("summary", "entry_state", "exit_state", "continuity_notes")
                        if scene["intent"][field]
                    )
                    shots.append(
                        {
                            "shot_id": uid(),
                            "order": i,
                            "nominal_duration_s": length,
                            "target_trim_s": excess if i == len(lengths) - 1 else 0,
                            "continuity_mode": (
                                "continue"
                                if i or scene["continuity"]["mode"] == "continue_from_previous"
                                else "cut"
                            ),
                            "prompt_snapshot": prompt,
                            "first_frame_prompt": scene["intent"]["entry_state"] or prompt,
                            "last_frame_prompt": (
                                (scene["intent"]["exit_state"] or prompt)
                                if cap["capability_snapshot"]["supports_last_frame"]
                                and (
                                    self.config.get("integration_generate_last_frame", False)
                                    or (supplied_last and i == len(lengths) - 1)
                                )
                                else ""
                            ),
                            "reference_asset_ids": sorted(refs),
                        }
                    )
                variants.append(
                    dict(
                        **cap,
                        variant_id=variant_id,
                        shots=shots,
                        estimated_cost={
                            "amount": cost,
                            "currency": "USD",
                            "basis": (
                                "configured provider estimator"
                                if cost is not None
                                else "provider estimate unavailable"
                            ),
                        },
                    )
                )
            if not variants:
                plan["status"] = "blocked"
                plan["blocked_reasons"].append(
                    {
                        "scene_id": scene["scene_id"],
                        "code": "no_compatible_variant",
                        "message": "No available provider variant satisfies this scene budget and capabilities.",
                    }
                )
            previous_scene_ready = bool(variants)
            plan["scenes"].append(
                {
                    "scene_id": scene["scene_id"],
                    "order": scene["order"],
                    "requested_duration_s": scene["requested_duration_s"],
                    "variants": variants,
                    "warnings": list(dict.fromkeys(warnings)),
                }
            )
        plan = seal(plan)
        from api.config import redact_config_secrets

        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            old = self._existing(
                db, "request", request["idempotency_key"], request["document_sha256"]
            )
            if old:
                return self.get("GenerationPlan", old)
            self._put(db, "GenerationRequest", request)
            self._put(db, "GenerationPlan", plan)
            db.execute(
                "INSERT INTO configurations VALUES(?,?)",
                (plan["plan_id"], canonical_bytes(redact_config_secrets(self.config))),
            )
            db.execute(
                "INSERT INTO bindings VALUES(?,?,?,?)",
                (
                    "request",
                    request["idempotency_key"],
                    request["document_sha256"],
                    plan["plan_id"],
                ),
            )
        return plan

    def _check_approval(self, approval):
        validate("PlanApproval", approval)
        plan = self.get("GenerationPlan", approval["plan_id"])
        request = self.get("GenerationRequest", plan["request_id"])
        if (
            approval["plan_sha256"] != plan["document_sha256"]
            or approval["request_id"] != plan["request_id"]
        ):
            raise ValueError("Approval does not match the exact plan and request")
        if plan["status"] != "ready" or datetime.fromisoformat(
            plan["valid_until"].replace("Z", "+00:00")
        ) <= datetime.now(UTC):
            raise ValueError("Plan is blocked or expired; create and approve a new plan")
        allowed = approval["approved_variant_ids"]
        variants = {v["variant_id"]: v for s in plan["scenes"] for v in s["variants"]}
        if len(set(allowed)) != len(allowed) or not set(allowed) <= set(variants):
            raise ValueError("Unapproved or unknown variant")
        current = {cap["provider"]: cap for cap in self.capabilities()["providers"]}
        request_scenes = {s["scene_id"]: s for s in request["scenes"]}
        plan_scenes = {
            v["variant_id"]: s["scene_id"] for s in plan["scenes"] for v in s["variants"]
        }
        for identifier in allowed:
            variant = variants[identifier]
            cap = current.get(variant["provider"])
            if (
                cap is None
                or cap["model"] != variant["model"]
                or cap["capability_snapshot"] != variant["capability_snapshot"]
            ):
                raise ValueError(
                    "Provider capabilities or model changed; create and approve a new plan"
                )
            generator = self._generator(variant["provider"])
            ratio = request_scenes[plan_scenes[identifier]]["delivery"]["aspect_ratio"]
            estimates = [
                self._estimate_cost(generator, variant["provider"], shot["nominal_duration_s"], ratio)
                for shot in variant["shots"]
            ]
            current_cost = (
                None
                if any(cost is None for cost in estimates)
                or (variant["provider"] in {"fal", "fal.ai"} and not any(estimates))
                else sum(estimates)
            )
            if current_cost != variant["estimated_cost"]["amount"]:
                raise ValueError("Provider cost estimate changed; create and approve a new plan")
        for scene in plan["scenes"]:
            if not any(v["variant_id"] in allowed for v in scene["variants"]):
                raise ValueError("Each requested scene needs an approved variant")
        if (
            any(variants[v]["estimated_cost"]["amount"] is None for v in allowed)
            and not approval["allow_unknown_cost"]
        ):
            raise ValueError("Unknown cost requires explicit approval")
        if approval["execution_mode"] == "keyframes" and not approval["allow_unknown_cost"]:
            raise ValueError(
                "Keyframe cost is unavailable; explicit unknown-cost approval required"
            )
        if approval["approved_keyframe_result_id"]:
            result = self.get("GenerationResult", approval["approved_keyframe_result_id"])
            if (
                result["plan_id"] != plan["plan_id"]
                or result["request_id"] != plan["request_id"]
                or result["result_kind"] != "keyframes"
                or result["terminal_status"] != "succeeded"
            ):
                raise ValueError("Keyframe result must be a successful result for this exact plan")
        if approval["execution_mode"] == "video":
            for planned_scene in plan["scenes"]:
                scene = request_scenes[planned_scene["scene_id"]]
                needs_first_frame = (
                    not approval["approved_keyframe_result_id"]
                    and scene["continuity"]["mode"] != "continue_from_previous"
                    and not any(
                        asset["role"] == "first_frame"
                        for asset in scene["reference_assets"]
                    )
                )
                minimum_calls = min(
                    len(variant["shots"]) + needs_first_frame
                    for variant in planned_scene["variants"]
                    if variant["variant_id"] in allowed
                )
                if scene["generation_policy"]["max_attempts"] < minimum_calls:
                    raise ValueError(
                        "Scene attempt limit cannot cover required keyframe and video calls"
                    )
        return plan, request

    def approve(self, approval):
        validate("PlanApproval", approval)
        with self.db() as db:
            existing = self._existing(
                db, "approval", approval["idempotency_key"], approval["document_sha256"]
            )
            if existing:
                return self.job(existing)
        plan, request = self._check_approval(approval)
        job = {
            "id": uid(),
            "status": "queued",
            "progress": 0,
            "created_at": now(),
            "started_at": None,
            "finished_at": None,
            "request_id": request["request_id"],
            "plan_id": plan["plan_id"],
            "approval_id": approval["approval_id"],
            "current_scene_id": None,
            "current_shot_id": None,
            "warnings": [],
            "cancel_requested": False,
            "result_id": None,
            "result_uri": None,
            "result_sha256": None,
        }
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = self._existing(
                db, "approval", approval["idempotency_key"], approval["document_sha256"]
            )
            if existing:
                return self.job(existing)
            self._put(db, "PlanApproval", approval)
            db.execute("INSERT INTO jobs VALUES(?,?)", (job["id"], canonical_bytes(job)))
            db.execute(
                "INSERT INTO bindings VALUES(?,?,?,?)",
                ("approval", approval["idempotency_key"], approval["document_sha256"], job["id"]),
            )
        return job

    def job(self, job_id):
        with self.db() as db:
            row = db.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise KeyError(job_id)
        return json.loads(row[0])

    def _update(self, job_id, **fields):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                raise KeyError(job_id)
            job = json.loads(row[0])
            if job["result_id"] and not set(fields) <= {"result_uri", "warnings"}:
                return job
            job.update(fields)
            db.execute("UPDATE jobs SET body=? WHERE id=?", (canonical_bytes(job), job_id))
        return job

    def cancel(self, job_id):
        job = self._update(job_id, cancel_requested=True)
        if job["status"] == "queued":
            return self.run(job_id)
        return job

    def result(self, job_id):
        result_id = self.job(job_id)["result_id"]
        if not result_id:
            raise KeyError("Job has no terminal result yet")
        return self.get("GenerationResult", result_id)

    def _materialize(self, asset, directory):
        parsed = urlsplit(asset["uri"])
        if parsed.scheme == "file":
            if parsed.netloc not in {"", "localhost"}:
                raise ValueError("Remote file host is unsupported")
            path = Path(unquote(parsed.path)).resolve()
            roots = [
                self.root,
                *(Path(root).resolve() for root in self.config.get("integration_asset_roots", [])),
            ]
            if not any(path.is_relative_to(root) for root in roots):
                raise ValueError("Local asset is outside configured integration_asset_roots")
        elif parsed.scheme == "gs":
            prefixes = list(self.config.get("integration_asset_uri_prefixes", []))
            if self.config.get("gcs_bucket"):
                prefixes.append(
                    "gs://"
                    + self.config["gcs_bucket"]
                    + "/"
                    + self.config.get("gcs_prefix", "ttv-api").strip("/")
                    + "/"
                )
            if not any(asset["uri"].startswith(prefix.rstrip("/") + "/") for prefix in prefixes):
                raise ValueError(
                    "Object asset is outside configured integration_asset_uri_prefixes"
                )
            from google.cloud import storage

            path = directory / (uid() + Path(parsed.path).suffix)
            storage.Client().bucket(parsed.netloc).blob(
                parsed.path.lstrip("/")
            ).download_to_filename(path)
        else:
            raise ValueError("Input assets require durable file:// or gs:// URIs")
        if file_hash(path) != asset["sha256"]:
            raise ValueError("Reference asset hash mismatch")
        if not path.is_relative_to(self.root):
            destination = directory / (asset["sha256"][7:] + path.suffix)
            shutil.copy2(path, destination)
            if file_hash(destination) != asset["sha256"]:
                raise ValueError("Reference asset changed while snapshotting")
            path = destination
        return str(path)

    def _asset(self, path, role, job_id):
        path = Path(path).resolve()
        sha = file_hash(path)
        destination = self.root / "assets" / (sha[7:] + path.suffix)
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            shutil.copy2(path, destination)
        if file_hash(destination) != sha:
            raise Conflict("Stored asset content differs from its immutable hash")
        path = destination
        uri = path.as_uri()
        if self.config.get("gcs_bucket"):
            try:
                from api.config import GCSConfig
                from api.gcs_client import create_gcs_client

                client = create_gcs_client(
                    GCSConfig(
                        bucket=self.config["gcs_bucket"],
                        prefix=self.config.get("gcs_prefix", "ttv-api"),
                        credentials_path=self.config.get("credentials_path", "credentials.json"),
                    )
                )
                uri = client.upload_artifact(str(path), job_id, path.name)
            except Exception:
                try:
                    job = self.job(job_id)
                    self._update(
                        job_id,
                        warnings=list(dict.fromkeys([*job["warnings"], "Asset publication pending"])),
                    )
                except KeyError:
                    pass
        return {
            "asset_id": uid(),
            "uri": uri,
            "sha256": file_hash(path),
            "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            "role": role,
            "rights_note": None,
        }

    def _provider_references(self, output, fallback_ids, result, job_id):
        """Snapshot exact post-preparation inputs reported by the provider adapter."""
        if output is None or output.reference_paths is None:
            return fallback_ids
        from video_generator_interface import cleanup_prepared_reference

        known = list(result["reference_assets"])
        known.extend(k["asset"] for scene in result["scenes"] for k in scene["keyframes"])
        ids = []
        sources = {prepared: source for source, prepared in output.reference_sources}
        prepared_sources = {}
        for path in output.reference_paths:
            sha = file_hash(path)
            asset = next(
                (a for a in known if a["sha256"] == sha and a["asset_id"] in fallback_ids),
                None,
            ) or next((a for a in known if a["sha256"] == sha), None)
            if asset is None:
                asset = self._asset(path, "provider_reference", job_id)
                result["reference_assets"].append(asset)
                known.append(asset)
            ids.append(asset["asset_id"])
            if path in sources:
                source_sha = file_hash(sources[path])
                source_asset = next(
                    (a for a in known if a["sha256"] == source_sha and a["asset_id"] in fallback_ids),
                    None,
                ) or next((a for a in known if a["sha256"] == source_sha and a is not asset), None)
                if source_asset is None:
                    raise ValueError("Prepared reference has no recorded source asset")
                prepared_sources[asset["asset_id"]] = source_asset["asset_id"]
                cleanup_prepared_reference(path)
        if prepared_sources:
            output.parameters["prepared_reference_sources"] = canonical_bytes(prepared_sources).decode()
        return ids

    @contextmanager
    def _provider_call(self, job_id, shot_id, stage):
        """Claim a deterministic paid operation and persist its receipt during submission."""
        from video_generator_interface import generation_observer

        attempt_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, canonical_bytes([job_id, shot_id, stage]).decode())
        )
        receipt = {
            "attempt_id": attempt_id,
            "shot_id": shot_id,
            "stage": stage,
            "status": "started",
            "started_at": now(),
            "provider_request_id": None,
        }
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT body FROM provider_calls WHERE id=?", (attempt_id,)
            ).fetchone()
            if existing:
                # A known remote request or an uncertain submission must never become a new charge.
                previous = json.loads(existing[0])
                reason = (
                    "provider request already exists"
                    if previous.get("provider_request_id")
                    else "submission already recorded"
                )
                raise Conflict(
                    f"Paid operation deduplicated: {reason}; inspect its durable receipt"
                )
            db.execute(
                "INSERT INTO provider_calls VALUES(?,?,?)",
                (attempt_id, job_id, canonical_bytes(receipt)),
            )

        def observe(details):
            snapshot = dict(details)
            if snapshot.get("billing") is not None:
                snapshot["billing"] = asdict(snapshot["billing"])
            receipt.update(snapshot)
            with self.db() as db:
                db.execute(
                    "UPDATE provider_calls SET body=? WHERE id=?",
                    (canonical_bytes(receipt), attempt_id),
                )

        token = generation_observer.set(observe)
        try:
            yield attempt_id
        finally:
            generation_observer.reset(token)

    def _record_attempt(self, job_id, attempt):
        with self.db() as db:
            db.execute(
                "INSERT INTO attempts VALUES(?,?,?)",
                (attempt["attempt_id"], job_id, canonical_bytes(attempt)),
            )
            db.execute(
                "UPDATE provider_calls SET body=? WHERE id=?",
                (canonical_bytes(attempt), attempt["attempt_id"]),
            )

    def _keyframe(self, prompt, output, reference=None, reference_images_dir=None):
        if self.keyframe_generator:
            return self.keyframe_generator(prompt, output, reference)
        from keyframe_generator import generate_keyframe_output

        return generate_keyframe_output(
            prompt,
            str(output),
            model_name=self.config["image_generation_model"],
            imageRouter_api_key=self.config.get("image_router_api_key"),
            stability_api_key=self.config.get("stability_api_key"),
            openai_api_key=self.config.get("openai_api_key"),
            gemini_api_key=self.config.get("gemini_api_key"),
            input_image_path=reference,
            size=self.config.get("image_size"),
            reference_images_dir=reference_images_dir,
            max_retries=0,
            allow_provider_fallback=False,
        )

    def _publish_result(self, result):
        """Recover publication from the terminal database record without replaying providers."""
        path = self.root / "results" / f"{result['result_id']}.json"
        path.parent.mkdir(exist_ok=True)
        body = canonical_bytes(result)
        import tempfile

        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != body:
                    raise Conflict("Terminal result file has different content") from None
        finally:
            temporary.unlink()
        uri = path.as_uri()
        warnings = []
        try:
            uri = self._asset(path, "generation_result", result["job_id"])["uri"]
        except Exception:
            warnings.append("Result is durable locally; object-store publication failed.")
        warnings = [
            warning for warning in self.job(result["job_id"])["warnings"]
            if warning != "Result publication pending"
        ] + warnings
        if self.config.get("gcs_bucket") and urlsplit(uri).scheme == "file":
            warnings.append("Result publication pending")
        warnings = list(dict.fromkeys(warnings))
        self._update(result["job_id"], result_uri=uri, warnings=warnings)
        return result

    def run(self, job_id):
        existing = self.job(job_id)
        if existing["result_id"]:
            return self._publish_result(self.get("GenerationResult", existing["result_id"]))
        # A process crash after submission is uncertain. Never replay an active job automatically.
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                raise KeyError(job_id)
            job = json.loads(row[0])
            if job["result_id"]:
                return self.get("GenerationResult", job["result_id"])
            if job["status"] != "queued":
                raise Conflict("Job is already running; paid calls will not be replayed")
            job.update(status="started", started_at=now())
            db.execute("UPDATE jobs SET body=? WHERE id=?", (canonical_bytes(job), job_id))
        approval = self.get("PlanApproval", job["approval_id"])
        plan = self.get("GenerationPlan", job["plan_id"])
        request = self.get("GenerationRequest", job["request_id"])
        result = {
            "contract_version": "1.0",
            "result_id": uid(),
            "result_kind": approval["execution_mode"],
            "request_id": request["request_id"],
            "plan_id": plan["plan_id"],
            "approval_id": approval["approval_id"],
            "job_id": job_id,
            "created_at": now(),
            "terminal_status": "failed",
            "reference_assets": [
                copy.deepcopy(a) for s in request["scenes"] for a in s["reference_assets"]
            ],
            "preview_assets": [],
            "scenes": [],
        }
        run_error = None
        try:
            self._check_approval(approval)
            with self.db() as db:
                config = json.loads(
                    db.execute(
                        "SELECT body FROM configurations WHERE plan_id=?", (plan["plan_id"],)
                    ).fetchone()[0]
                )
            from api.config import _restore_config_secrets

            # Restore only credentials from deployment configuration; plan-time provider settings remain frozen.
            self.config = _restore_config_secrets(config, self.config)
            total_spent = 0.0
            previous_scene_boundary = None
            previous_scene_take = None
            keyframes = (
                self.get("GenerationResult", approval["approved_keyframe_result_id"])
                if approval["approved_keyframe_result_id"]
                else None
            )
            for scene in sorted(request["scenes"], key=lambda s: s["order"]):
                out = {
                    "scene_id": scene["scene_id"],
                    "order": scene["order"],
                    "status": "failed",
                    "actual_shots": [],
                    "attempts": [],
                    "clips": [],
                    "keyframes": [],
                    "takes": [],
                    "warnings": [],
                }
                result["scenes"].append(out)
                self._update(job_id, current_scene_id=scene["scene_id"])
                if self.job(job_id)["cancel_requested"]:
                    out["status"] = "canceled"
                    continue
                try:
                    ps = next(s for s in plan["scenes"] if s["scene_id"] == scene["scene_id"])
                    variants = [
                        v
                        for v in ps["variants"]
                        if v["variant_id"] in approval["approved_variant_ids"]
                    ]
                    directory = self.root / "media" / job_id / str(scene["order"])
                    directory.mkdir(parents=True, exist_ok=True)
                    scene_references = list(scene["reference_assets"])
                    continues = scene["continuity"]["mode"] == "continue_from_previous"
                    inherited_frame = previous_scene_boundary if continues else None
                    if inherited_frame:
                        boundary_asset = self._asset(inherited_frame, "first_frame", job_id)
                        if previous_scene_take:
                            boundary_asset["source_take_id"] = previous_scene_take
                        result["reference_assets"].append(boundary_asset)
                        scene_references.append(boundary_asset)
                    elif continues and not any(
                        a["role"] == "first_frame" for a in scene_references
                    ):
                        out["warnings"].append(
                            "Continuity needs a supplied or successfully generated previous boundary"
                        )
                        continue
                    refs = {
                        a["asset_id"]: self._materialize(a, directory) for a in scene_references
                    }
                    for asset in scene_references:
                        snapshot = self._asset(refs[asset["asset_id"]], asset["role"], job_id)
                        for recorded in result["reference_assets"]:
                            if recorded["asset_id"] == asset["asset_id"]:
                                recorded["uri"] = snapshot["uri"]
                    reference_directory = None
                    extra_references = [
                        a
                        for a in scene_references
                        if a["role"] not in {"first_frame", "last_frame"}
                    ]
                    if extra_references:
                        if (
                            not self.keyframe_generator
                            and "gemini"
                            not in str(self.config.get("image_generation_model", "")).lower()
                        ):
                            raise ValueError(
                                "Multiple visual references require a configured Gemini keyframe model"
                            )
                        import shutil

                        reference_directory = directory / "reference-images"
                        reference_directory.mkdir(exist_ok=True)
                        for asset in extra_references:
                            shutil.copy2(
                                refs[asset["asset_id"]],
                                reference_directory
                                / (uid() + Path(refs[asset["asset_id"]]).suffix),
                            )
                    scene_spent = 0.0
                    last_failed = None
                    call_count = 0
                    for variant in variants:
                        complete = []
                        previous_frame = inherited_frame
                        generator = self._generator(variant["provider"], self.config)
                        cap = variant["capability_snapshot"]
                        for shot in sorted(variant["shots"], key=lambda s: s["order"]):
                            self._update(job_id, current_shot_id=shot["shot_id"])
                            if self.job(job_id)["cancel_requested"]:
                                break
                            if call_count >= scene["generation_policy"]["max_attempts"]:
                                out["warnings"].append("Scene attempt limit reached")
                                break
                            estimate = self._estimate_cost(
                                generator, variant["provider"], shot["nominal_duration_s"],
                                scene["delivery"]["aspect_ratio"]
                            )
                            if variant["estimated_cost"]["amount"] is None:
                                estimate = None
                            scene_budget = scene["generation_policy"]["max_estimated_cost_usd"]
                            job_budget = approval["max_estimated_cost_usd"]
                            if estimate is not None and (
                                (scene_budget is not None and scene_spent + estimate > scene_budget)
                                or (job_budget is not None and total_spent + estimate > job_budget)
                            ):
                                out["warnings"].append("Estimated cost budget exhausted")
                                break
                            frames = {}
                            if keyframes:
                                ks = next(
                                    s
                                    for s in keyframes["scenes"]
                                    if s["scene_id"] == scene["scene_id"]
                                )
                                for k in ks["keyframes"]:
                                    if k["shot_id"] == shot["shot_id"]:
                                        frames[k["position"]] = self._materialize(
                                            k["asset"], directory
                                        )
                                        out["keyframes"].append({**k, "attempt_id": None})
                                if "first" not in frames:
                                    out["warnings"].append(
                                        "Approved keyframe result has no frame for this variant"
                                    )
                                    break
                            else:
                                supplied = next(
                                    (
                                        refs[r]
                                        for r in shot["reference_asset_ids"]
                                        if next(a for a in scene_references if a["asset_id"] == r)[
                                            "role"
                                        ]
                                        == "first_frame"
                                    ),
                                    None,
                                )
                                reference = previous_frame or supplied
                                supplied_last = next(
                                    (
                                        refs[a["asset_id"]]
                                        for a in scene_references
                                        if a["role"] == "last_frame"
                                    ),
                                    None,
                                )
                                for position in (
                                    ("first", "last")
                                    if cap["supports_last_frame"] and shot["last_frame_prompt"]
                                    else ("first",)
                                ):
                                    if self.job(job_id)["cancel_requested"]:
                                        break
                                    supplied_frame = (
                                        reference if position == "first" else supplied_last
                                    )
                                    if supplied_frame:
                                        path = supplied_frame
                                    else:
                                        if call_count >= scene["generation_policy"]["max_attempts"]:
                                            raise ValueError("Scene attempt limit reached")
                                        if not approval["allow_unknown_cost"]:
                                            raise ValueError(
                                                "Keyframe generation has unknown cost and needs explicit approval"
                                            )
                                        if (
                                            scene_budget is not None and scene_spent >= scene_budget
                                        ) or (job_budget is not None and total_spent >= job_budget):
                                            raise ValueError(
                                                "Known keyframe charges exhausted the execution budget"
                                            )
                                        self._check_approval(approval)
                                        path = directory / (uid() + ".png")
                                        image_reference_ids = [
                                            a["asset_id"] for a in extra_references
                                        ]
                                        if reference:
                                            input_asset = self._asset(
                                                reference, "provider_reference", job_id
                                            )
                                            if reference == inherited_frame and previous_scene_take:
                                                input_asset["source_take_id"] = previous_scene_take
                                            result["reference_assets"].append(input_asset)
                                            image_reference_ids.append(input_asset["asset_id"])
                                        started = now()
                                        generated = None
                                        failure = None
                                        try:
                                            with self._provider_call(
                                                job_id, shot["shot_id"], position
                                            ) as image_attempt_id:
                                                call_count += 1
                                                generated = self._keyframe(
                                                    shot[f"{position}_frame_prompt"],
                                                    path,
                                                    reference,
                                                    (
                                                        str(reference_directory)
                                                        if reference_directory
                                                        else None
                                                    ),
                                                )
                                        except Conflict:
                                            raise
                                        except Exception as image_error:
                                            generated = getattr(
                                                image_error, "generation_output", None
                                            )
                                            failure = type(image_error).__name__
                                        if not any(
                                            s["shot_id"] == shot["shot_id"]
                                            for s in out["actual_shots"]
                                        ):
                                            out["actual_shots"].append(
                                                {
                                                    **{
                                                        k: shot[k]
                                                        for k in (
                                                            "shot_id",
                                                            "nominal_duration_s",
                                                            "continuity_mode",
                                                        )
                                                    },
                                                    "order": len(out["actual_shots"]),
                                                }
                                            )
                                        from video_generator_interface import GenerationOutput

                                        observed = (
                                            generated
                                            if isinstance(generated, GenerationOutput)
                                            else None
                                        )
                                        image_attempt = {
                                            "attempt_id": image_attempt_id,
                                            "shot_id": shot["shot_id"],
                                            "attempt_number": len(out["attempts"]) + 1,
                                            "provider": (
                                                observed.provider
                                                if observed
                                                else (
                                                    "offline-fixture"
                                                    if self.keyframe_generator
                                                    else "unknown"
                                                )
                                            ),
                                            "model": observed.model if observed else None,
                                            "model_version": (
                                                observed.model_version if observed else None
                                            ),
                                            "provider_request_id": (
                                                observed.provider_request_id if observed else None
                                            ),
                                            "prompt_snapshot": (
                                                observed.prompt
                                                if observed
                                                else shot[f"{position}_frame_prompt"]
                                            ),
                                            "reference_asset_ids": self._provider_references(
                                                observed, image_reference_ids, result, job_id
                                            ),
                                            "generation_parameters": (
                                                observed.parameters if observed else {}
                                            ),
                                            "seed": observed.seed if observed else None,
                                            "started_at": (
                                                observed.started_at.isoformat().replace(
                                                    "+00:00", "Z"
                                                )
                                                if observed
                                                else started
                                            ),
                                            "finished_at": (
                                                observed.finished_at.isoformat().replace(
                                                    "+00:00", "Z"
                                                )
                                                if observed
                                                else now()
                                            ),
                                            "status": "failed" if failure else "succeeded",
                                            "fallback_from_attempt_id": None,
                                            "error": (
                                                {
                                                    "code": failure,
                                                    "message": "Keyframe provider call failed.",
                                                }
                                                if failure
                                                else None
                                            ),
                                            "billing": (
                                                asdict(observed.billing)
                                                if observed and observed.billing
                                                else {
                                                    "raw_unit_name": None,
                                                    "raw_units": None,
                                                    "estimated_usd": None,
                                                    "actual_usd": None,
                                                    "currency": "USD",
                                                }
                                            ),
                                        }
                                        self._record_attempt(job_id, image_attempt)
                                        out["attempts"].append(image_attempt)
                                        image_billing = image_attempt["billing"]
                                        image_charge = (
                                            image_billing["actual_usd"]
                                            if image_billing["actual_usd"] is not None
                                            else image_billing["estimated_usd"]
                                        )
                                        if image_charge is not None:
                                            scene_spent += image_charge
                                            total_spent += image_charge
                                        if failure:
                                            raise ValueError("Keyframe generation failed")
                                    frames[position] = str(path)
                                    asset = self._asset(path, "keyframe", job_id)
                                    out["keyframes"].append(
                                        {
                                            "shot_id": shot["shot_id"],
                                            "order": shot["order"],
                                            "position": position,
                                            "attempt_id": (
                                                None
                                                if supplied_frame
                                                else image_attempt["attempt_id"]
                                            ),
                                            "asset": asset,
                                            "media": probe(path),
                                        }
                                    )
                            if self.job(job_id)["cancel_requested"]:
                                break
                            if approval["execution_mode"] == "keyframes":
                                complete.append(shot["shot_id"])
                                previous_frame = frames.get("last", frames["first"])
                                continue
                            self._check_approval(approval)
                            if call_count >= scene["generation_policy"]["max_attempts"]:
                                out["warnings"].append("Scene attempt limit reached")
                                break
                            if estimate is not None and (
                                (scene_budget is not None and scene_spent + estimate > scene_budget)
                                or (job_budget is not None and total_spent + estimate > job_budget)
                            ):
                                out["warnings"].append(
                                    "Estimated cost budget exhausted after keyframe generation"
                                )
                                break
                            if not any(
                                s["shot_id"] == shot["shot_id"] for s in out["actual_shots"]
                            ):
                                out["actual_shots"].append(
                                    {
                                        **{
                                            k: shot[k]
                                            for k in (
                                                "shot_id",
                                                "nominal_duration_s",
                                                "continuity_mode",
                                            )
                                        },
                                        "order": len(out["actual_shots"]),
                                    }
                                )
                            started = now()
                            output = None
                            error = None
                            path = directory / (uid() + ".mp4")
                            try:
                                with self._provider_call(
                                    job_id, shot["shot_id"], "video"
                                ) as attempt_id:
                                    call_count += 1
                                    output = generator.generate_video(
                                        prompt=shot["prompt_snapshot"],
                                        input_image_path=frames["first"],
                                        output_path=str(path),
                                        duration=shot["nominal_duration_s"],
                                        last_frame_path=frames.get("last"),
                                        aspect_ratio=scene["delivery"]["aspect_ratio"],
                                        approved_prompt=True,
                                        cancellation_check=lambda: self.job(job_id)[
                                            "cancel_requested"
                                        ],
                                    )
                            except Conflict:
                                raise
                            except Exception as caught:
                                output = getattr(caught, "generation_output", None)
                                status = getattr(caught, "status_code", None)
                                status = (
                                    status
                                    if isinstance(status, int) and 100 <= status <= 599
                                    else None
                                )
                                error = {
                                    "code": type(caught).__name__
                                    + (f"_HTTP_{status}" if status else ""),
                                    "message": (
                                        f"Provider request failed with HTTP {status}."
                                        if status
                                        else "Provider call failed; see private worker diagnostics."
                                    ),
                                }
                            billing = (
                                asdict(output.billing)
                                if output and output.billing
                                else {
                                    "raw_unit_name": None,
                                    "raw_units": None,
                                    "estimated_usd": estimate,
                                    "actual_usd": None,
                                    "currency": "USD",
                                }
                            )
                            attempt = {
                                "attempt_id": attempt_id,
                                "shot_id": shot["shot_id"],
                                "attempt_number": len(out["attempts"]) + 1,
                                "provider": output.provider if output else variant["provider"],
                                "model": output.model if output else variant["model"],
                                "model_version": output.model_version if output else None,
                                "provider_request_id": (
                                    output.provider_request_id if output else None
                                ),
                                "prompt_snapshot": (
                                    output.prompt if output else shot["prompt_snapshot"]
                                ),
                                "reference_asset_ids": self._provider_references(
                                    output,
                                    [
                                        k["asset"]["asset_id"]
                                        for k in out["keyframes"]
                                        if k["shot_id"] == shot["shot_id"]
                                    ],
                                    result,
                                    job_id,
                                ),
                                "generation_parameters": output.parameters if output else {},
                                "seed": output.seed if output else None,
                                "started_at": (
                                    output.started_at.isoformat().replace("+00:00", "Z")
                                    if output
                                    else started
                                ),
                                "finished_at": (
                                    output.finished_at.isoformat().replace("+00:00", "Z")
                                    if output
                                    else now()
                                ),
                                "status": "failed" if error else "succeeded",
                                "fallback_from_attempt_id": last_failed,
                                "error": error,
                                "billing": billing,
                            }
                            if error and self.job(job_id)["cancel_requested"]:
                                attempt["status"] = "canceled"
                            self._record_attempt(job_id, attempt)
                            out["attempts"].append(attempt)
                            charge = (
                                billing["actual_usd"]
                                if billing["actual_usd"] is not None
                                else billing["estimated_usd"]
                            )
                            if charge is not None:
                                scene_spent += charge
                                total_spent += charge
                            if error:
                                last_failed = attempt["attempt_id"]
                                break
                            asset = self._asset(output.path, "generated_clip", job_id)
                            media = probe(output.path)
                            if not media["measured_duration_s"]:
                                raise ValueError("Generated clip has no measured duration")
                            clip = {
                                "clip_id": uid(),
                                "attempt_id": attempt["attempt_id"],
                                "asset": asset,
                                "media": media,
                            }
                            out["clips"].append(clip)
                            if scene["delivery"]["audio_policy"] == "required" and not media["has_audio"]:
                                raise ValueError("Generated clip has no required audio")
                            complete.append(clip)
                            # Continue from the actual generated boundary, rather than an imagined last frame.
                            boundary = directory / (uid() + ".png")
                            subprocess.run(
                                [
                                    "ffmpeg",
                                    "-v",
                                    "error",
                                    "-sseof",
                                    "-0.1",
                                    "-i",
                                    output.path,
                                    "-frames:v",
                                    "1",
                                    "-y",
                                    str(boundary),
                                ],
                                check=True,
                            )
                            previous_frame = str(boundary)
                        if len(complete) == len(variant["shots"]):
                            out["status"] = "succeeded"
                            if approval["execution_mode"] == "video":
                                regeneration = request["regeneration"] or {}
                                supersedes = []
                                for rid in regeneration.get("base_result_ids", []):
                                    base = self.get("GenerationResult", rid)
                                    for oldscene in base["scenes"]:
                                        if oldscene["scene_id"] == scene["scene_id"]:
                                            supersedes.extend(
                                                t["take_id"]
                                                for t in oldscene["takes"]
                                                if t["take_id"]
                                                in regeneration.get("supersedes_take_ids", [])
                                            )
                                out["takes"].append(
                                    {
                                        "take_id": uid(),
                                        "clip_ids": [c["clip_id"] for c in complete],
                                        "clip_order": {
                                            c["clip_id"]: i for i, c in enumerate(complete)
                                        },
                                        "preview_asset_id": None,
                                        "supersedes_take_ids": supersedes,
                                        "depends_on_take_ids": sorted(
                                            {
                                                a["source_take_id"]
                                                for a in scene_references
                                                if a.get("source_take_id")
                                            }
                                        ),
                                        "nominal_duration_s": sum(
                                            s["nominal_duration_s"] for s in variant["shots"]
                                        ),
                                        "measured_duration_s": sum(
                                            c["media"]["measured_duration_s"] for c in complete
                                        ),
                                    }
                                )
                                previous_scene_take = out["takes"][-1]["take_id"]
                            previous_scene_boundary = previous_frame
                            break
                    if out["status"] != "succeeded":
                        previous_scene_take = previous_scene_boundary = None
                    if self.job(job_id)["cancel_requested"]:
                        out["status"] = "canceled"
                    elif out["status"] != "succeeded" and out["clips"]:
                        out["status"] = "partial"
                    self._update(
                        job_id, progress=round(100 * len(result["scenes"]) / len(request["scenes"]))
                    )
                except sqlite3.Error, OSError:
                    raise
                except Exception as scene_error:
                    out["status"] = "partial" if out["clips"] else "failed"
                    out["warnings"].append(f"Scene stopped: {type(scene_error).__name__}")
                    previous_scene_take = previous_scene_boundary = None

        except Exception as error:
            run_error = type(error).__name__
        # Every requested scene appears, even when validation, an asset probe, or cancellation stopped work.
        included = {s["scene_id"] for s in result["scenes"]}
        canceled = self.job(job_id)["cancel_requested"]
        for scene in request["scenes"]:
            if scene["scene_id"] not in included:
                result["scenes"].append(
                    {
                        "scene_id": scene["scene_id"],
                        "order": scene["order"],
                        "status": "canceled" if canceled else "failed",
                        "actual_shots": [],
                        "attempts": [],
                        "clips": [],
                        "keyframes": [],
                        "takes": [],
                        "warnings": [],
                    }
                )
        if run_error:
            for scene in result["scenes"]:
                if scene["status"] != "succeeded":
                    scene["warnings"].append(f"Execution stopped: {run_error}")
        statuses = {s["status"] for s in result["scenes"]}
        result["terminal_status"] = (
            "canceled"
            if canceled
            else (
                "succeeded"
                if statuses == {"succeeded"}
                else (
                    "partial"
                    if any(
                        s["takes"] or s["clips"] or s["status"] == "succeeded"
                        for s in result["scenes"]
                    )
                    else "failed"
                )
            )
        )
        result["created_at"] = now()
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            job = json.loads(
                db.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()[0]
            )
            if job["cancel_requested"]:
                result["terminal_status"] = "canceled"
            result = seal(result)
            self._put(db, "GenerationResult", result)
            path = self.root / "results" / f"{result['result_id']}.json"
            job.update(
                status=(
                    "canceled"
                    if result["terminal_status"] == "canceled"
                    else (
                        "finished"
                        if result["terminal_status"] in {"succeeded", "partial"}
                        else "failed"
                    )
                ),
                progress=100,
                finished_at=now(),
                result_id=result["result_id"],
                result_uri=path.as_uri(),
                result_sha256=result["document_sha256"],
                warnings=list(dict.fromkeys([*job["warnings"], "Result publication pending"])),
            )
            db.execute("UPDATE jobs SET body=? WHERE id=?", (canonical_bytes(job), job_id))
        return self._publish_result(result)


def run_queued_generation(job_id, queue):
    from api.config import restore_job_config_secrets
    from api.models import JobStatus

    job = queue.get_job(job_id)
    config = restore_job_config_secrets(job.config)
    service = GenerationService(config["integration_root"], config)
    service.run(job_id)
    state = service.job(job_id)
    queue.update_job_status(
        job_id, JobStatus(state["status"]), progress=100, gcs_uri=state["result_uri"]
    )
    return state["result_uri"]
