# Media Tooling integration validation

Validated locally on Python 3.14.3 with FFmpeg/FFprobe. The implementation branch is `feat/ttv-media-integration`, based on `4c20bfc774ebd96c60d9b1988bf810c5165467e1`. Paid provider generation, deployment, push and merge were not run.

## Checks

The focused producer suite passed **123 tests**:

```sh
python -m pytest tests/test_generation_integration.py tests/test_fal_generator.py tests/test_veo31_duration.py tests/test_keyframe_checkpoints.py tests/test_keyframe_generator_gemini.py tests/test_models.py tests/test_queue.py -q --tb=short
```

The 14 integration tests exercise immutable hashes and shared fixtures, exact approval/expiry/capability guards, approved fallback provenance, duplicate/conflicting requests, known/unknown costs, cancellation, targeted regeneration, keyframe review, HTTP/file canonical parity, both worker entry points, retained standalone source clips, prepared input snapshots, durable submission receipts and terminal publication recovery without provider replay.

Ruff and Black pass on the new integration modules, contract producer, pilot and acceptance tests. `git diff --check` and Python compilation pass. `uv build --wheel --out-dir /tmp/ttv-integration-wheel` succeeds; the wheel contains all five schemas, ten fixtures, hash manifest and contract README.

The existing worker/backend regression set has **90 passed and 21 failed**, with exactly the same failing test names on the immutable base commit and this implementation:

```sh
python -m pytest tests/test_models.py tests/test_queue.py tests/test_video_worker.py tests/test_trio_video_worker.py tests/test_trio_integration.py tests/test_integration_mocked_backends.py tests/test_keyframe_generator_gemini.py -q --tb=short
```

These failures include legacy mocks requiring deployment configuration/credentials, existing Trio expectations and authenticated API fixture mismatches. The artifact endpoint suite also has the same baseline result: **1 passed, 17 failed**, primarily authorization fixture mismatches. These comparisons establish baseline parity for the exercised paths; the full pre-existing suite is not green in this environment.

## Cross-repository pilot

`scripts/integration_pilot.py` uses the production generation service with deterministic providers and real synthetic media. It emits a single-clip scene, a multi-clip scene, a failed attempt followed by an approved fallback, a downstream take dependency, and a targeted middle-scene replacement.

The Media Tooling consumer at commit `8bce40e435d6454002e529193fb4367de9ebae67` imported those actual producer documents and assets, rendered both 7-second edits through its unchanged renderer, and passed its existing verifier. Replacing the middle scene preserved the outer take IDs and asset hashes. A stale dependency blocked compilation until an explicit editorial acceptance. The shared 22-case validator corpus matched: seven accepted and fifteen rejected.

Local evidence is under `/Users/magos/.codex/worktrees/ttv-media-integration/media-pilot-commitable`: `acceptance.json`, `contract-parity.json`, `pilot.log`, and `verify-cli.json`. Producer documents are under `/Users/magos/.codex/worktrees/ttv-media-integration/offline-pilot-commitable/handoff`.

## Operational limits

- API and workers require the same durable integration filesystem; live Redis/RQ dispatch and GCS transfers were not exercised against deployed services.
- An interrupted active provider operation requires operator investigation. Its durable submission receipt prevents a new charge under the same operation key. Terminal publication retries are automatic on repeated collection through the execution service.
- Keyframe review covers the variants represented in its immutable result. A later approved fallback with no reviewed frames stops safely and requires a suitable keyframe result or a fresh full-video approval.
- v2 planning deterministically derives prompts from editorial intent and the existing provider-duration planner. It does not call the optional LLM prompt enhancer.
- Standalone/v1 provenance sidecars retain the original invocation and media; they do not invent v2 request/plan/approval documents. Full Media Tooling import uses the v2 artifact chain.
- Provider billing that lacks a verified USD conversion remains unknown even when raw units are present. Live provider model access, output quality, remote cancellation and actual charges need a separately authorized operational pilot.
