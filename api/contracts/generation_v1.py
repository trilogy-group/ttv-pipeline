"""Transport-neutral generation contract 1.0. Published schemas are generated below."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ID = Annotated[str, Field(min_length=1, max_length=256)]
Hash = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
UTC = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")]
Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Order = Annotated[int, Field(ge=0)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Document(Strict):
    contract_version: Literal["1.0"]
    created_at: UTC
    document_sha256: Hash


class Media(Strict):
    measured_duration_s: Seconds | None
    width: Annotated[int, Field(gt=0)] | None
    height: Annotated[int, Field(gt=0)] | None
    fps: PositiveSeconds | None
    has_audio: bool | None


class AssetRef(Strict):
    asset_id: ID
    uri: str
    sha256: Hash
    mime_type: str
    role: str
    rights_note: str | None
    media: Media | None = None
    source_take_id: ID | None = None


class Transition(Strict):
    type: Literal["cut", "none"]
    editorial_note: str | None


class Intent(Strict):
    summary: Annotated[str, Field(min_length=1)]
    entry_state: str
    exit_state: str
    continuity_notes: str


class Continuity(Strict):
    mode: Literal["independent", "continue_from_previous"]
    required_reference_asset_ids: list[ID]


class Delivery(Strict):
    aspect_ratio: str
    target_width: Annotated[int, Field(gt=0)]
    target_height: Annotated[int, Field(gt=0)]
    audio_policy: Literal["preserve_if_present", "mute", "required"]


class Policy(Strict):
    provider_preferences: list[str]
    max_attempts: Annotated[int, Field(ge=1, le=100)]
    max_estimated_cost_usd: Seconds | None
    allow_provider_fallback: bool


class RequestScene(Strict):
    scene_id: ID
    order: Order
    intent: Intent
    requested_duration_s: Annotated[float, Field(gt=0, le=14440)]
    transition_out: Transition
    continuity: Continuity
    reference_assets: list[AssetRef]
    delivery: Delivery
    generation_policy: Policy


class Storyboard(Strict):
    revision_id: ID
    sha256: Hash


class Regeneration(Strict):
    base_result_ids: list[ID]
    supersedes_take_ids: list[ID]
    reason: str


class GenerationRequest(Document):
    request_id: ID
    project_id: ID
    revision: Annotated[int, Field(ge=1)]
    supersedes_request_id: ID | None
    idempotency_key: ID
    storyboard: Storyboard
    scenes: Annotated[list[RequestScene], Field(min_length=1)]
    regeneration: Regeneration | None


class Capability(Strict):
    allowed_durations_s: Annotated[list[PositiveSeconds], Field(min_length=1)]
    supports_first_frame: bool
    supports_last_frame: bool
    supports_audio: bool


class Cost(Strict):
    amount: Seconds | None
    currency: Literal["USD"]
    basis: str


class Shot(Strict):
    shot_id: ID
    order: Order
    nominal_duration_s: PositiveSeconds
    target_trim_s: Seconds
    continuity_mode: Literal["cut", "continue"]
    prompt_snapshot: str
    first_frame_prompt: str
    last_frame_prompt: str
    reference_asset_ids: list[ID]


class Variant(Strict):
    variant_id: ID
    provider: str
    model: str
    capability_snapshot: Capability
    shots: Annotated[list[Shot], Field(min_length=1)]
    estimated_cost: Cost


class PlanScene(Strict):
    scene_id: ID
    order: Order
    requested_duration_s: PositiveSeconds
    variants: list[Variant]
    warnings: list[str]


class BlockReason(Strict):
    scene_id: ID | None
    code: str
    message: str


class GenerationPlan(Document):
    plan_id: ID
    request_id: ID
    valid_until: UTC
    status: Literal["ready", "blocked"]
    scenes: list[PlanScene]
    blocked_reasons: list[BlockReason]


class PlanApproval(Document):
    approval_id: ID
    request_id: ID
    plan_id: ID
    plan_sha256: Hash
    execution_mode: Literal["keyframes", "video"]
    approved_variant_ids: Annotated[list[ID], Field(min_length=1)]
    approved_keyframe_result_id: ID | None
    idempotency_key: ID
    allow_unknown_cost: bool
    max_estimated_cost_usd: Seconds | None


class Billing(Strict):
    raw_unit_name: str | None
    raw_units: str | None
    estimated_usd: Seconds | None
    actual_usd: Seconds | None
    currency: Literal["USD"]


class AttemptError(Strict):
    code: str
    message: str


class Attempt(Strict):
    attempt_id: ID
    shot_id: ID
    attempt_number: Annotated[int, Field(ge=1)]
    provider: str
    model: str | None
    model_version: str | None
    provider_request_id: str | None
    prompt_snapshot: str
    reference_asset_ids: list[ID]
    generation_parameters: dict[str, str | int | float | bool | None]
    seed: int | None
    started_at: UTC
    finished_at: UTC
    status: Literal["succeeded", "failed", "canceled"]
    fallback_from_attempt_id: ID | None
    error: AttemptError | None
    billing: Billing


class ActualShot(Strict):
    shot_id: ID
    order: Order
    nominal_duration_s: PositiveSeconds
    continuity_mode: Literal["cut", "continue"]


class Clip(Strict):
    clip_id: ID
    attempt_id: ID
    asset: AssetRef
    media: Media


class Keyframe(Strict):
    shot_id: ID
    order: Order
    position: Literal["first", "last"]
    attempt_id: ID | None
    asset: AssetRef
    media: Media


class Take(Strict):
    take_id: ID
    clip_ids: Annotated[list[ID], Field(min_length=1)]
    clip_order: dict[str, Order]
    preview_asset_id: ID | None
    supersedes_take_ids: list[ID]
    depends_on_take_ids: list[ID]
    nominal_duration_s: PositiveSeconds
    measured_duration_s: PositiveSeconds


class ResultScene(Strict):
    scene_id: ID
    order: Order
    status: Literal["succeeded", "partial", "failed", "canceled"]
    actual_shots: list[ActualShot]
    attempts: list[Attempt]
    clips: list[Clip]
    keyframes: list[Keyframe]
    takes: list[Take]
    warnings: list[str]


class GenerationResult(Document):
    result_id: ID
    result_kind: Literal["video", "keyframes"]
    request_id: ID
    plan_id: ID
    approval_id: ID
    job_id: ID
    terminal_status: Literal["succeeded", "partial", "failed", "canceled"]
    reference_assets: list[AssetRef]
    preview_assets: list[AssetRef]
    scenes: list[ResultScene]


class ResultRef(Strict):
    result_id: ID
    sha256: Hash


class ClipEdit(Strict):
    clip_id: ID
    order: Order
    trim_in_s: Seconds
    trim_out_s: PositiveSeconds
    audio_use: Literal["preserve", "mute"]


class SelectedScene(Strict):
    scene_id: ID
    order: Order
    take_id: ID
    clip_edits: Annotated[list[ClipEdit], Field(min_length=1)]
    transition_out: Transition


class ContinuityDecision(Strict):
    scene_id: ID
    dependency_take_id: ID
    decision: Literal["accept_stale", "use_boundary", "regenerate"]
    replacement_asset: AssetRef | None


class EDLRef(Strict):
    path: str
    sha256: Hash


class EditorialSelection(Document):
    selection_version: Literal[1]
    edit_revision_id: ID
    project_id: ID
    storyboard_revision_id: ID
    source_results: Annotated[list[ResultRef], Field(min_length=1)]
    scenes: Annotated[list[SelectedScene], Field(min_length=1)]
    continuity_decisions: list[ContinuityDecision]
    edl: EDLRef | None


MODELS = {
    m.__name__: m
    for m in (GenerationRequest, GenerationPlan, PlanApproval, GenerationResult, EditorialSelection)
}


def canonical_bytes(value: object) -> bytes:
    """Canonical 1.0 JSON: sorted Unicode keys, integral numbers normalized, no whitespace."""

    def normalize(obj):
        if isinstance(obj, float):
            if not math.isfinite(obj):
                raise ValueError("Non-finite JSON number")
            return int(obj) if obj.is_integer() else obj
        if isinstance(obj, dict):
            return {k: normalize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [normalize(v) for v in obj]
        return obj

    return json.dumps(
        normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def seal(document: dict) -> dict:
    document = {k: v for k, v in document.items() if k != "document_sha256"}
    return {**document, "document_sha256": digest(document)}


def assert_safe(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if re.sub(r"[^a-z]", "", key.lower()) in {
                "authorization",
                "headers",
                "httpheaders",
                "apikey",
                "token",
                "accesstoken",
                "refreshtoken",
                "password",
                "secret",
                "credentials",
                "signedurl",
                "downloadurl",
                "privatekey",
                "secretkey",
            }:
                raise ValueError("Credentials and delivery URLs are forbidden in durable documents")
            assert_safe(child)
    elif isinstance(value, list):
        for child in value:
            assert_safe(child)
    elif isinstance(value, str):
        for candidate in re.findall(r"(?:https?|gs|s3|file)://[^\s\"<>]+", value, flags=re.I):
            uri = urlsplit(candidate)
            if uri.username or uri.password or uri.query or uri.fragment:
                raise ValueError("Durable URIs cannot contain credentials, queries, or fragments")
        if re.search(r"\bBearer\s+\S+", value, flags=re.I):
            raise ValueError("Authorization values are forbidden in durable documents")


def _unique(items: list[dict[str, object]], key: str) -> dict[object, dict[str, object]]:
    result = {item[key]: item for item in items}
    if len(result) != len(items):
        raise ValueError(f"Duplicate {key}")
    return result


def validate(kind: str, document: dict) -> dict:
    if document.get("contract_version") != "1.0":
        raise ValueError("Unsupported contract version; expected exactly 1.0")
    assert_safe(document)
    try:
        MODELS[kind].model_validate(json.loads(canonical_bytes(document)))
    except ValidationError as error:
        location = ".".join(map(str, error.errors(include_input=False)[0]["loc"]))
        raise ValueError(f"Invalid {kind} field: {location}") from None
    for field in ("created_at", "valid_until"):
        if field in document:
            datetime.fromisoformat(document[field].replace("Z", "+00:00"))
    if seal(document)["document_sha256"] != document["document_sha256"]:
        raise ValueError("Document hash mismatch")
    scenes = document.get("scenes", [])
    for key in ("scene_id", "order"):
        if len({s[key] for s in scenes}) != len(scenes):
            raise ValueError(f"Duplicate scene {key}")
    if kind == "GenerationRequest":
        _unique([asset for scene in scenes for asset in scene["reference_assets"]], "asset_id")
        for scene in scenes:
            for role in ("first_frame", "last_frame"):
                if sum(asset["role"] == role for asset in scene["reference_assets"]) > 1:
                    raise ValueError(f"Multiple {role} assets in one scene")
    if kind == "EditorialSelection":
        for scene in scenes:
            for edit in scene["clip_edits"]:
                if edit["trim_out_s"] <= edit["trim_in_s"]:
                    raise ValueError("Clip edit out-point must exceed in-point")
    if kind == "GenerationResult":
        for scene in scenes:
            shots = _unique(scene["actual_shots"], "shot_id")
            _unique(scene["actual_shots"], "order")
            attempts = _unique(scene["attempts"], "attempt_id")
            clips = _unique(scene["clips"], "clip_id")
            _unique(scene["takes"], "take_id")
            seen: set[str] = set()
            for attempt in scene["attempts"]:
                if attempt["shot_id"] not in shots:
                    raise ValueError("Attempt refers to missing shot")
                fallback = attempt.get("fallback_from_attempt_id")
                if fallback is not None and fallback not in seen:
                    raise ValueError("Fallback refers to missing or later attempt")
                seen.add(attempt["attempt_id"])
            for clip in clips.values():
                if (
                    clip["attempt_id"] not in attempts
                    or attempts[clip["attempt_id"]]["status"] != "succeeded"
                ):
                    raise ValueError("Clip must refer to a successful attempt")
            for take in scene["takes"]:
                if not take["clip_ids"] or len(set(take["clip_ids"])) != len(take["clip_ids"]):
                    raise ValueError("Take requires distinct ordered clips")
                if any(clip_id not in clips for clip_id in take["clip_ids"]):
                    raise ValueError("Take refers to missing clip")
                if set(take["clip_order"]) != set(take["clip_ids"]) or len(
                    set(take["clip_order"].values())
                ) != len(take["clip_ids"]):
                    raise ValueError("Take requires explicit unique clip order")

    return document


def publish():
    root = Path(__file__).with_name("v1_0")
    root.mkdir(exist_ok=True)
    manifest = {}
    for name, model in MODELS.items():
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            **model.model_json_schema(),
        }
        data = canonical_bytes(schema)
        (root / f"{name}.schema.json").write_bytes(data)
        manifest[name] = "sha256:" + hashlib.sha256(data).hexdigest()
    (root / "schema-hashes.json").write_bytes(canonical_bytes(manifest))


if __name__ == "__main__":
    publish()
