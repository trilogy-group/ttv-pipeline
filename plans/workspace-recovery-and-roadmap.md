# TTV Pipeline Recovery and Product Roadmap

Status: Phase 3 complete; Phase 4 pending
Last updated: 2026-07-19
Workspace: `/Users/magos/dev/kumanday/Parlina/ttv-pipeline`

## Purpose

This is the durable handoff and execution plan for recovering the repository,
modernizing Veo support, adding requested-duration control, and building a
low-touch human-in-the-loop frontend. It is the source of truth after context
compaction or a new Codex session.

## Resume instructions

1. Read this file before changing the repository.
2. Run `git status --short --branch` and compare it with the status recorded
   below; live Git state wins.
3. Continue the first phase whose status is not complete.
4. Do not merge or cherry-pick the closed Higgsfield branch.
5. Do not expose, commit, log, or include credentials in Docker build context.
6. Keep work on semantic branch names; never use a `codex/` prefix.

## Recovered history and decisions

- The older `plans/analysis.md` is from 2025-09-01. It covers a Minimax-era
  cleanup, not the later Higgsfield decision.
- Higgsfield PR #13 was closed unmerged by `kumanday` on 2026-02-23 with no
  explanatory review or comment.
- fal.ai PR #14 merged about 4.5 hours later. This suggests, but does not prove,
  that direct Higgsfield support was intentionally superseded.
- Preserve PR #13 and its remote branch as historical reference. Do not merge
  or cherry-pick it. If proprietary Higgsfield controls become a concrete need,
  build a small new adapter against its current SDK.
- Direct Vertex Veo 3.1 is the preferred first implementation. The repository
  already targets Vertex and this is smaller than adapting fal's model-specific
  Veo request schema.
- Treat user-facing `duration_seconds` as the requested final runtime. Keep
  provider clip duration as a separate internal concept.
- Treat a narrative scene, its timed beats or shots, and a provider-generated
  take as separate concepts. A longer generation may contain several shots, and
  a longer scene may still require several generations.
- Treat provider reference limits as budgets, not targets. Build a curated,
  versioned reference pack for each scene from reusable project assets.
- Keep audio references, independent audio timelines, splicing, muxing, and the
  final mix in a separate phase from visual scene preparation and assembly.
- Start frontend validation with a same-origin, zero-build UI. Add a frontend
  framework only when the proven workflow needs it.

## State at recovery start

- Checked-out branch: `feat/add-higgsfield-provider` at `e40c546`, matching its
  remote.
- Live `origin/main`: `046cf36`, five commits ahead of local `main`.
- Higgsfield branch divergence: four commits ahead and five behind main, with
  merge conflicts.
- Modified tracked file: `api/gcs_client.py`, containing an uncommitted
  emergency hardcoded `/app/pipeline_config.yaml` fallback.
- Untracked root files included duplicate service-account JSON files, a TLS
  certificate/private key, `config/`, `rescue_video.py`, and `start-worker.sh`.
- Root credential JSON files were byte-identical to copies already under the
  ignored `credentials/` directory.
- `.dockerignore` did not exclude `.env`, `pipeline_config.yaml`, credentials,
  or PEM files while `Dockerfile.api` uses `COPY . .`.
- Full tests were not runnable in the global Python 3.14 environment because
  project dependencies were missing. A focused 58-test config/model suite
  passed before recovery work.

## Phase 0 - Recover and secure the workspace

Status: complete

Goal: establish a safe current-main baseline without carrying hackathon or
secret-bearing debris forward.

Actions:

- [x] Exclude local secrets from both Git and Docker build contexts.
- [x] Remove root credential duplicates after repointing runtime configuration
      and Compose mounts to the ignored `credentials/` directory.
- [x] Move local TLS material under the ignored `certs/` directory.
- [x] Preserve required nonsecret Compose configuration.
- [x] Move one-off `rescue_video.py` and unused `start-worker.sh` to a recoverable
      backup outside the repository.
- [x] Discard the uncommitted emergency GCS fallback.
- [x] Switch to local `main`, fast-forward to `origin/main`, and create
      `chore/workspace-recovery`.
- [x] Keep PR #13 and the remote Higgsfield branch as the archive.
- [x] Validate ignored-secret behavior, Compose syntax, diff hygiene, and a
      focused test suite.
- [x] Record final branch, backup location, and validation evidence here; the
      Phase 0 commit is discoverable from this branch's Git history.

Conditional action:

- Rotate the service-account key if evidence shows that an image was built and
  pushed/shared from the dirty tree. Local file presence alone does not prove
  exposure, so no external key rotation is authorized by this phase.

Phase 0 result:

- Recovery branch: `chore/workspace-recovery`, based on `origin/main` at
  `046cf36`.
- Recoverable backup: `/Users/magos/.Trash/ttv-pipeline-phase0-2026-07-18`.
  It contains the two duplicate root credential files and the two one-off
  scripts. Canonical credential copies remain under ignored `credentials/`.
- TLS certificate and private key moved under ignored `certs/`; the private key
  retained mode `0600`.
- Local ignored `pipeline_config.yaml`, base Compose, HTTP/3 Compose, sample
  config, and relevant docs now use `credentials/credentials.json`.
- `.gitignore` and `.dockerignore` now exclude local credential JSONs, PEMs,
  credential/certificate directories, `.env` files, and the live pipeline
  config. Docker images continue to receive required files only via runtime
  mounts.
- Required nonsecret `config/nginx.conf` and `config/redis.conf` were preserved;
  the Nginx upstream was corrected to the Compose service name `api`.
- The emergency `api/gcs_client.py` fallback was discarded; the checked-out
  source matches current main.
- External credential rotation was not performed. The ignored local pipeline
  config contains old live provider credentials; rotate them before reuse if
  their exposure or continued validity cannot be established.

Phase 0 validation:

- `docker compose config --quiet`: valid.
- `docker compose -f docker-compose.http3.yml config --quiet`: valid.
- Git/Docker ignore and credential-path assertions: passed.
- Recoverable backup presence and TLS key mode assertions: passed.
- `git diff --check`: clean.
- Dependency-light config/model tests: see the latest execution log below.
- `tests/test_fal_generator.py` cannot currently collect because the global
  Python environment lacks the declared `requests` dependency. Reproducible
  dependency setup remains Phase 1 work; Phase 0 changed no Python code.

## Phase 1 - Establish a trustworthy baseline

Status: complete

Goal: make CLI/API behavior testable and consistent before new product work.

Completed one-off tasks:

- [x] Standardize development on Python 3.14 and `uv sync --extra dev`; make
  `pyproject.toml` the dependency source of truth.
- [x] Make threading and Trio workers use the effective configuration stored with
  each job instead of rebuilding prompt-only configuration.
- [x] Resolve the `/jobs` versus `/v1/jobs` contract mismatch.
- [x] Share CLI frame preparation and frame-path resolution with API workers.
- [x] Fix remote fallback exception handling and reversed fallback arguments.
- [x] Stop persisting unredacted configuration/API keys in output directories.
- [x] Add one mocked API-to-worker smoke test.

Deferred cleanup:

- Decide whether to fix the legacy Angie/HTTP/3 deployment path or delete it.
  It is not required by the requested Phase 1 slice and its historical tests
  reference files that are not present in the repository.

Exit criteria:

- A documented local setup command creates a runnable environment.
- Focused CLI, API, worker, and config tests pass from that environment.
- The effective job configuration reaches the generation pipeline unchanged.

Phase 1 result:

- `.python-version` pins Python 3.14; `uv.lock` is committed alongside
  `pyproject.toml`, and root `requirements.txt` was removed.
- The supported development bootstrap is `uv sync --extra dev`.
- The Python 3.14 upgrade removes the unused `stability-sdk` and deprecated
  `google-generativeai` dependency trees. Gemini keyframes use the supported
  `google-genai` client with inline image parts.
- API jobs are canonical at `/v1/jobs`; middleware, OpenAPI, examples, and
  smoke tests use the same prefix. The unversioned `/jobs` route returns 404.
- Job creation stores the merged pipeline and GCS configuration with secret
  values replaced by redaction markers. Both worker modes restore only those
  exact fields from their locally mounted runtime configuration, preserving
  queued nonsecret settings without retaining credentials in Redis.
- `prepare_keyframes` now owns initial-frame generation/preservation and safe
  first/last-frame path resolution for CLI and API workers.
- Remote fallback catches the generator interface's real exception type and
  calls the fallback factory with `(primary_backend, config)`.
- CLI and worker output directories no longer receive full configuration YAML
  files containing API keys.
- The Trio job path no longer duplicates Redis initialization, constructs the
  private `trio.Cancelled` exception, or leaks successful-job temp directories.
- Post-review hardening added the advertised Nginx TLS listener, made optional
  Redis authentication consistent across the server, RQ workers, and health
  checks, propagated `GCS_CREDENTIALS_PATH` into the nested Veo configuration,
  preserved a supplied `segment_00.png` through keyframe cleanup, and baked a
  safe sample config into standalone API/worker images.

Phase 1 validation:

- `uv sync --extra dev`: succeeds with CPython 3.14.3.
- Focused Python 3.14 and Phase 1 suite: 117 passed. The later review-focused
  API/config/queue/worker suite passed 85 tests.
- `uv lock --check`, Python compilation, `bash -n setup.sh`, and
  `git diff --check`: pass.
- `uv build` produces a wheel containing the API, workers, generators, and
  top-level pipeline modules.
- Python 3.14 Linux wheel resolution passes for PyTorch 2.10.0,
  torchvision 0.25.0, and torchaudio 2.10.0 on CUDA 12.8; the updated Python
  and NVIDIA base-image tags resolve. The Python 3.14 API and worker images now
  build successfully, load their standalone sample configuration, and connect
  to an authenticated Redis instance through RQ's existing environment support.
- Container checks also prove Redis rejects unauthenticated clients when a
  password is configured and that the mounted Nginx TLS configuration is valid.
- Full historical suite baseline: 340 passed, 103 failed, 4 skipped. The
  failures cluster in missing legacy Angie assets, obsolete API fixtures and
  removed artifact/log/cancel routes, old monitoring assumptions, and separate
  Trio executor tests. They are recorded as legacy cleanup outside this focused
  gate rather than hidden by it.

## Phase 2 - Veo 3.1 and requested duration

Status: complete
Priority: highest product priority

Goal: deliver a correct direct-Vertex Veo 3.1 vertical slice with first/last
frames and requested final duration.

Required work:

- [x] Default to `veo-3.1-generate-001`, while allowing the fast GA model through
  configuration.
- [x] Forward the configured Veo model through the factory.
- [x] Represent Veo clip durations as allowed values `[4, 6, 8]`, not a scalar max.
- [x] Send `duration_seconds` to the Google SDK.
- [x] Pass the existing ending keyframe via `GenerateVideosConfig.last_frame`.
- [x] Add optional CLI `--duration-seconds` and API `duration_seconds`.
- [x] Omitted duration preserves AI-inferred behavior.
- [x] Provided duration constrains decomposition and final runtime.
- [x] Plan provider-compatible segment durations; trim only when an exact sum is
  impossible, and surface the first/last-frame tradeoff for a trimmed final clip.
- [x] Validate LLM output against the requested segment plan.
- [x] Add mocked tests asserting model, duration, first frame, and last frame in the
  actual Google request.

Sizing:

- Direct Veo 3.1 plus one-provider duration support is a bounded Codex task.
- A generalized multi-provider duration planner should become a separate spec if
  more than the initial providers require materially different behavior.

Phase 2 result:

- Direct Vertex generation now defaults to `veo-3.1-generate-001`; setting
  `google_veo.veo_model: veo-3.1-fast-generate-001` selects the fast GA model.
- Veo capabilities and validation use the provider's discrete 4, 6, and 8 second
  clip lengths. The Google SDK request receives the selected model, per-segment
  `duration_seconds`, the first image, and `GenerateVideosConfig.last_frame`.
- CLI `--duration-seconds` and API `duration_seconds` flow through the effective
  job configuration. Without a request, the LLM still infers runtime and segment
  count while choosing provider-compatible clip lengths.
- Requested runtimes are decomposed into the fewest Veo-compatible clips. Odd or
  sub-four-second requests use the smallest covering plan and trim only the final
  output. The API response and worker/CLI logs warn that trimming removes the
  final generated ending keyframe.
- Both prompt-enhancement paths now share one instruction and validation flow.
  LLM segment numbers, counts, total runtime, and per-segment durations must match
  the requested plan before keyframe or video generation starts.
- Exact-sum concatenation remains stream-copy. Only the impossible-sum trim path
  re-encodes, because stream-copy trimming was measurably inexact at packet
  boundaries.

Phase 2 validation:

- Focused Python 3.14 suite: 122 passed.
- Mocked `google-genai` request asserts the default model, duration, first-frame
  GCS image, and last-frame GCS image; a factory test covers the fast model.
- Mocked API-to-worker smoke test proves requested duration survives redacted job
  storage and secret restoration, and the API surfaces the trim warning.
- Real ffmpeg check concatenated 4- and 6-second clips, trimmed the covering plan,
  and produced an ffprobe duration of exactly `9.000000` seconds.
- Python compilation, YAML parsing, CLI help, and `git diff --check` pass.
- A broader legacy `tests/test_main.py` probe still reproduces the Phase 1
  baseline failures for obsolete routes/fixtures and live readiness dependencies;
  no Phase 2 code path is implicated.

## Phase 3 - Make fal.ai a first-class provider

Status: complete
Priority: next product phase
Execution: one bounded feature PR; split only if a model requires a materially
different asset workflow

Goal: make fal a reliable provider with explicit contracts for the video models
the product uses, especially Seedance, without pretending every fal endpoint
shares one input schema.

### Current assessment

- The current adapter sends one generic synchronous payload directly to
  `fal.run`. It does not use fal's recommended durable queue lifecycle.
- It retries every exception around the entire generation call. This can retry
  authentication and validation failures and can resubmit an ambiguously
  accepted paid job.
- It always requires an input image while advertising text-to-video support.
- It injects numeric `duration` and `image_url` fields into every endpoint even
  though model field names, encodings, and allowed values differ.
- The configured Hailuo-02 endpoint accepts 6 or 10 seconds, but the pipeline can
  send 5.
- Response parsing recursively accepts the first URL in any field instead of the
  documented `video.url` output.
- The current cost/timing header list is partly speculative. fal documents
  request identity, billable units, queue metrics, and separate Platform APIs
  that are better foundations for later usage analysis.

### Design decisions

- Keep one fal provider transport and a small table of concrete model profiles.
  A profile owns its endpoint, supported generation mode, required assets,
  duration contract, payload mapping, capability claims, and output mapping.
- Continue accepting an exact `fal.model` endpoint ID, but require it to match a
  tested profile. Fail configuration validation for unprofiled endpoints instead
  of guessing their capabilities or payload fields.
- Use fal's queue REST API through the already-installed `requests` dependency;
  do not add `fal-client` unless the REST surface proves insufficient. Submit
  once, retain the returned request and lifecycle URLs, poll status, retrieve the
  result, and cancel on a local deadline where possible.
- Keep this phase image-driven because that is the pipeline's current generation
  contract. Do not advertise text-to-video, reference-to-video, extension, or
  video editing until the pipeline exposes and tests those input roles. The
  Seedance reference-to-video vertical slice follows the scene/reference domain
  work in Phase 4 rather than expanding this provider-transport PR.
- Parse the documented `video.url` result only. Preserve `video.content_type`,
  `video.file_name`, `video.file_size`, and returned `seed` when present as
  ephemeral provider metadata.
- Make `FAL_KEY` the documented environment variable while retaining
  `FAL_API_KEY` as a compatibility fallback.
- Disable fal's implicit equivalent-model fallback for profiled requests so the
  selected model, price, and capabilities remain deterministic; send
  `x-app-fal-disable-fallback: true` and continue using the pipeline's explicit
  provider fallback.
- Disable remote input/output payload retention by default because requests can
  contain prompts and base64 keyframes. Send `X-Fal-Store-IO: 0`; this does not
  prevent later usage and billing reconciliation.
- Make the existing requested-duration planner consume the selected profile's
  allowed or fixed clip lengths instead of treating fal as one scalar maximum.

### Initial first-class model profiles

1. **Seedance 2.0 - priority vertical slice**
   - Support `bytedance/seedance-2.0/image-to-video` and
     `bytedance/seedance-2.0/fast/image-to-video`.
   - Map the starting keyframe to `image_url` and the optional ending keyframe to
     `end_image_url`.
   - Accept integer durations from 4 through 15 seconds or `auto` when the user
     did not request a runtime.
   - Validate endpoint-specific resolution limits, aspect ratio, and image size
     before submission.
2. **Veo 3.1 through fal**
   - Support `fal-ai/veo3.1/image-to-video` and
     `fal-ai/veo3.1/first-last-frame-to-video`, including their fast variants
     after confirming the live schemas are identical.
   - Select the endpoint from the available keyframes instead of sending a last
     frame to an image-only schema.
   - Encode durations as `4s`, `6s`, or `8s` and map frames to `image_url` or
     `first_frame_url` plus `last_frame_url` as required.
3. **Hailuo-02 through fal**
   - Correct the currently configured
     `fal-ai/minimax/hailuo-02/standard/image-to-video` path.
   - Encode duration as 6 or 10 seconds and map the optional ending keyframe to
     `end_image_url`.
   - Keep resolution constraints in the profile rather than a provider-wide
     scalar maximum.
4. **MiniMax Video-01 through fal**
   - Support `fal-ai/minimax/video-01/image-to-video` with its documented
     `prompt`, `image_url`, and `prompt_optimizer` schema.
   - Treat it as a fixed six-second profile; reject incompatible requested clip
     lengths rather than sending an unsupported duration field.
   - Do not copy Hailuo duration or ending-frame fields into this older endpoint.

Additional model variants should be one profile plus payload-contract tests, not
new provider classes. The Hailuo and MiniMax names overlap in fal's catalog, so
endpoint IDs are the authoritative identity.

### Audio boundary

- A profiled endpoint may return a video file that already contains model-native
  audio. Phase 3 transports that file as one opaque video artifact; it does not
  extract, replace, align, mix, or independently version its audio.
- Do not add user-facing audio generation switches or audio reference inputs in
  this phase. Record their presence in a released provider schema only so Phase
  7 can make an explicit product decision later.
- Narrative prompt text may still describe dialogue, ambience, or sound when a
  model understands it. That does not create an audio timeline or editing
  contract in the pipeline.

### Queue, rate-limit, and retry behavior

- Submit to `queue.fal.run` and retain `request_id`, `status_url`, `response_url`,
  and `cancel_url` before polling.
- Treat 400, 401, 403, 404, and model-validation 422 responses as non-retryable.
- Detect 429 responses plus `X-Fal-Needs-Retry` and `X-Fal-Error-Type`. Honor
  `Retry-After` when supplied; otherwise use capped exponential backoff with
  jitter.
- Retry only safe status/result reads and explicitly rejected submissions. Never
  blindly repeat a submission after an ambiguous network failure or after a
  `request_id` has been received.
- Rely on fal's queue to retry accepted jobs for 429 concurrency limits, 503/504
  failures, and runner connection errors. Do not wrap the entire queued job in
  the generic local retry handler.
- Bound queue-start time separately from the worker's total deadline. On local
  timeout or cancellation, call fal's cancel URL and report that in-progress
  cancellation is best-effort.
- Surface fal's machine-readable `error_type` and model validation details in the
  existing generator exception hierarchy without logging credentials or full
  base64 inputs.

### Seedance 2.5 release readiness

As of 2026-07-19, fal describes Seedance 2.5 as announced but not released and
says ByteDance has not published its full specification. Do not ship a guessed
fal endpoint or payload.

Release checklist:

1. Confirm the endpoint ID and schema from the live fal model API page.
2. Diff the released schema against the Seedance 2.0 profile.
3. Add the smallest new profile, exact capability data, and mocked payload tests.
4. Verify duration, reference limits, resolution, output format, cost units, and
   whether the output contains native audio against fal's published values.
5. Run one explicitly authorized paid smoke request before making 2.5 selectable
   by default.

The linked MuAPI-oriented community repository is a useful provisional watchlist,
not an API contract. Recheck candidate fields such as `duration`, `resolution` or
`ratio`, `output_format`, `bitrate_mode`, `camera_fixed`, `return_last_frame`,
role-tagged reference assets, and `seed` only after fal publishes its own schema.
Recheck `generate_audio` and audio-reference fields as inputs to the separate
Phase 7 design rather than adding them to visual preparation opportunistically.

### Observability and cost discovery, without metrics capture

- Retain the fal `request_id` and exact `endpoint_id` because the queue lifecycle
  needs them and a later metrics job can use them as reconciliation keys.
- Document the future event fields now: submitted, processing-started, completed,
  and downloaded timestamps; final status and `error_type`; queue
  `metrics.inference_time`; and `X-Fal-Billable-Units`. Do not add a metrics
  record, sidecar, or persistence path in this phase.
- Identify but do not yet ingest these Platform API surfaces:
  - `GET /v1/models/pricing` and `POST /v1/models/pricing/estimate`;
  - `GET /v1/models/billing-events`, keyed by `request_id`;
  - `GET /v1/models/usage` and `GET /v1/models/analytics`;
  - `GET /v1/models/requests/by-endpoint` for timing and failure reconciliation.
- Defer a database schema, scheduled ingestion, dashboards, alerts, cost
  attribution, and webhook processing to later phases. The durable staged API
  can add signed, idempotent webhooks when jobs no longer need a blocking worker.

### Required work

- [x] Replace the direct synchronous call with queue submit/status/result/cancel.
- [x] Add the four concrete model profiles above and correct capability claims.
- [x] Route first and last keyframes and duration through the selected profile.
- [x] Remove generic payload injection and recursive URL discovery for profiled
      endpoints.
- [x] Add rate-limit, safe-retry, deadline, cancellation, and typed-error handling.
- [x] Preserve only queue-required request identity for later reconciliation;
      stop treating speculative headers as cost records.
- [x] Remove the current speculative metrics sidecar; metrics persistence remains
      outside this phase.
- [x] Update sample configuration and provider documentation with supported
      endpoint IDs and their capabilities.
- [x] Add mocked request-contract and queue-lifecycle tests.

### Exit criteria

- Seedance 2.0, fal Veo 3.1, Hailuo-02, and MiniMax Video-01 each produce the
  documented payload for the keyframe inputs the profile claims to support.
- Omitted duration preserves each model's automatic/default behavior; provided
  duration is validated and encoded according to that model's contract.
- Tests prove submit happens once, polling reaches completion, output uses
  `video.url`, non-retryable errors fail immediately, retryable reads back off,
  and timeout attempts cancellation.
- Every generation exposes a fal request ID and endpoint ID for future billing
  reconciliation without persisting prompts, images, API keys, or a new metrics
  store.
- Seedance 2.5 remains unavailable until the release checklist passes.

Phase 3 validation:

- `uv run pytest tests/test_fal_generator.py tests/test_veo31_duration.py -q`:
  34 passed on Python 3.14.3.
- `uv run ruff check generators/remote/fal_generator.py
  tests/test_fal_generator.py`: passed.
- `uv run mypy --follow-imports=skip generators/remote/fal_generator.py`: passed.
- `uv run pytest -q`: 373 passed, 103 failed, 4 skipped. The failures reproduce
  the existing unrelated baseline around absent Angie assets, legacy API test
  fixtures without Redis/GCS readiness, and Trio compatibility; no fal or
  requested-duration test failed.

Sources reviewed 2026-07-19:

- [fal asynchronous queue and lifecycle](https://fal.ai/docs/documentation/model-apis/inference/queue)
- [fal reliability and automatic retries](https://fal.ai/docs/documentation/model-apis/inference/reliability)
- [fal platform headers](https://fal.ai/docs/documentation/model-apis/common-parameters)
- [fal pricing and Platform APIs](https://fal.ai/docs/documentation/model-apis/pricing)
- [Seedance 2.0 image-to-video schema](https://fal.ai/models/bytedance/seedance-2.0/image-to-video/api)
- [Seedance 2.0 reference-to-video schema](https://fal.ai/models/bytedance/seedance-2.0/reference-to-video)
- [Seedance 2.0 technical report](https://arxiv.org/abs/2604.14148)
- [Veo 3.1 first/last-frame schema](https://fal.ai/models/fal-ai/veo3.1/first-last-frame-to-video/api)
- [Hailuo-02 image-to-video schema](https://fal.ai/models/fal-ai/minimax/hailuo-02/standard/image-to-video/api)
- [MiniMax Video-01 image-to-video schema](https://fal.ai/models/fal-ai/minimax/video-01/image-to-video/api)
- [fal's Seedance 2.5 prerelease note](https://fal.ai/learn/tools/what-is-seedance-2-5)
- [community Seedance 2.5/MuAPI field watchlist](https://github.com/Anil-matcha/awesome-seedance-2.5-api-prompts)

## Phase 4 - Durable scene preparation and storyboard API

Status: pending
Execution: OpenSymphony/spec-driven

Goal: replace the monolithic ephemeral job with a resumable visual workflow that
can plan short clips or longer scene takes without equating a generated clip with
a single shot.

### Core vocabulary

- **Scene:** a narrative unit with a purpose, requested duration, entry state,
  exit state, and relationship to adjacent scenes.
- **Beat or shot:** a timed visual event inside a scene. Several beats may be
  generated together in one provider request.
- **Reference pack:** the curated, versioned set of project assets used for one
  scene take. Provider limits are maximums, not a reason to include every asset.
- **Take:** one provider-generated scene output plus its immutable prompt,
  reference-pack snapshot, model, request identity, and selected trim range.
- **Transition contract:** the intended relationship between adjacent selected
  takes, such as a hard cut, match cut, continuous action, or dissolve.

Initial architecture:

- Store a project/storyboard manifest in existing Redis.
- Store reusable visual assets, keyframes, and take versions in existing GCS.
- Add a database only when retention or query requirements exceed Redis.
- Keep visual render, trim, and stitch operations as ordinary jobs.
- Keep the current code's segment terminology until the domain specification
  decides whether a migration is worth the churn.

Stages/actions:

1. Create a project from the brief and requested final duration.
2. Decompose the story into scenes, then give each scene a timed beat or shot
   plan that covers its requested duration.
3. Create a project visual-asset library and automatically propose a small
   reference pack for each scene. Let the user change only the proposed roles or
   exceptions.
4. Generate optional entry, beat, or exit keyframes where the selected model
   benefits from anchors; do not require a keyframe at every edit boundary.
5. Approve the scene packet: scene intent, timed beats, reference pack, anchors,
   continuity state, and transition intent.
6. Render or regenerate scene takes and retain every version.
7. Select a take and trim range for each scene, then stitch the visual timeline
   according to the transition contracts.

Required semantics:

- Version scene plans, prompts, reference packs, keyframes, and takes; do not
  silently overwrite.
- A reference role belongs to its use in a scene, not permanently to the asset.
  The same image or video may serve different roles in different scenes.
- Preserve asset provenance and rights notes, and snapshot the exact asset
  versions used by each paid generation.
- Keep an explicit scene continuity ledger for character condition and position,
  wardrobe, props, location, time, weather, lighting, screen direction, camera
  grammar, and unresolved action.
- Do not force the prior take's last frame into every following scene. Use it for
  continuous action; use shared references or deliberate composition for hard
  and match cuts.
- Changing a shared boundary keyframe invalidates the takes on both sides that
  consume it.
- Expose provider/model capabilities so unsupported UI actions are hidden.
- Make each stage resumable and safe to retry without duplicating paid jobs.
- Allocate requested final duration to scenes first and provider-compatible
  takes second. Trimming remains an explicit selected-take operation.

### Seedance multimodal vertical slice

After the Phase 3 fal transport is complete, use the live
`bytedance/seedance-2.0/reference-to-video` endpoint to validate this domain with
one bounded image/video-reference flow before Seedance 2.5 is released:

- Support typed image and video references and their prompt aliases.
- Enforce the released Seedance 2.0 limits rather than adopting reported 2.5
  limits early.
- Store the immutable reference-pack snapshot with the take and expose the
  provider request ID for later cost reconciliation.
- Exclude audio references and independent audio controls; those belong to Phase
  7 even though the provider endpoint supports them.

### Explicit audio exclusion

Scene preparation may retain dialogue or sound intent as ordinary prompt text,
and previews may play audio already muxed into a provider result. This phase does
not define audio assets or reference roles, multiple audio timelines, track
selection, extraction, synchronization, splicing, crossfades, loudness handling,
or final muxing. Phase 7 owns that work against the selected visual timeline.

Exit criteria:

- The manifest distinguishes scenes, timed beats, reference packs, and takes.
- The workflow can approve a scene packet, generate multiple takes, select a trim,
  and re-stitch without rerendering unaffected scenes.
- A Seedance 2.0 request proves image/video reference mapping and immutable take
  provenance through the real staged API contract.
- No audio-timeline or muxing semantics leak into the scene-preparation domain.

## Phase 5 - Low-touch frontend

Status: pending
Execution: smoke UI as one-off; full workflow spec-driven

Goal: make the common path nearly automatic while allowing intervention at
high-value review points.

Suggested UX:

1. Brief: prompt, requested duration, provider/model, aspect ratio.
2. Storyboard: editable scene cards with timed beats and one accept-all path.
3. References: automatically propose each scene's visual reference pack and show
   only conflicts, missing roles, provenance warnings, or user-requested changes.
4. Anchors: offer one gallery review gate for optional entry, beat, and exit
   keyframes, with approve-all, regenerate, and edit-with-instructions controls.
5. Render: show scene takes with exception-based retry/edit controls, trim
   selection, and the intended transition to the next scene.
6. Final visual stitch and playback.

The primary review object is a scene packet, not an individual short segment.
Audio already embedded in a provider preview may play, but this frontend phase
does not expose track, splice, mix, or mux controls.

Implementation order:

- First, a same-origin HTML/CSS/JS smoke frontend served by FastAPI.
- Adopt React or another framework only after the staged workflow demonstrates
  that native UI state management is the constraint.

## Phase 6 - Scene revision, extension, and repair

Status: pending
Execution: provider spike first; OpenSymphony/spec-driven if retained

Goal: revise an individual selected scene take without rerendering unrelated
scenes.

Approach:

- Spike one concrete provider/model first; fal Kling O1 is the current candidate.
- Do not force video editing through the image-to-video generator interface.
- Support the smallest useful operation exposed by the chosen model: regenerate
  with instructions, replace the visual reference pack, extend the take, or edit
  a temporal region.
- Keep original and revised take versions plus their instruction deltas.
- Require review before a revision becomes the selected take.
- Re-stitch without rerendering unaffected scenes.
- Generalize the provider contract only after a second implementation proves
  that schemas share useful structure.
- Preserve provider-native audio opaquely with the video artifact; audio repair
  and reassembly remain Phase 7 work.

## Phase 7 - Audio timelines, splicing, and final mux

Status: pending
Execution: provider/audio spike first; OpenSymphony/spec-driven if retained

Goal: add explicit audio semantics after the selected visual scene timeline is
stable, without coupling audio architecture to scene-preparation work.

Inputs from earlier phases:

- Selected visual scene takes, trim ranges, and transition contracts.
- Provider-native muxed audio where present, treated as an optional source.
- Narrative dialogue, music, ambience, and sound-effect intent retained as text.

Required work:

- Decide when provider-native audio is retained, extracted, muted, or replaced.
- Define separate dialogue, music, ambience, and sound-effect tracks or timelines
  only where the product workflow demonstrates that they are needed.
- Add audio reference assets and provider audio-reference routing here, including
  provenance and rights metadata.
- Define trim, splice, crossfade, synchronization, sample-rate/channel handling,
  loudness policy, and deterministic final mux behavior.
- Preserve audio source and mix versions so revising one scene or track does not
  overwrite approved work.
- Rebuild the final audio/video output without regenerating unaffected video.

Non-goals:

- Do not redesign scene decomposition, visual reference packs, keyframe review,
  or video-provider transport in this phase.
- Do not add a full digital-audio-workstation interface; expose only the controls
  proven necessary by the selected workflow.

Exit criteria:

- The same selected visual timeline can be rendered with a deterministic,
  versioned audio mix.
- Scene trims and transitions produce synchronized audio edits without modifying
  the underlying selected video takes.
- Model-native audio and independently supplied tracks have explicit, testable
  precedence rather than implicit ffmpeg behavior.

## Deferred specifications

Create these only after the outline and relevant spike are approved:

- Durable project/storyboard domain and API spec.
- Scene, beat, reference-pack, take, continuity, and transition semantics.
- Visual reference asset provenance and versioning spec.
- Artifact versioning and invalidation spec.
- Full HITL frontend interaction spec.
- Scene revision/extension/editing spec, conditional on spike results.
- Audio timeline, splicing, and final-mux spec, kept independent from the visual
  preparation specifications above.

## Execution log

### 2026-07-18

- Analysis completed; no source changes were made during analysis.
- Remote refs refreshed; `origin/main` advanced to `046cf36`.
- Phase 0 authorized by the user and started.
- Wrote this durable plan and completed Phase 0 on
  `chore/workspace-recovery`.
- Moved duplicate credentials and one-off scripts to
  `/Users/magos/.Trash/ttv-pipeline-phase0-2026-07-18`.
- Moved TLS material under ignored `certs/`, secured Git/Docker exclusions,
  normalized runtime credential paths, and preserved required Compose config.

### 2026-07-19

- Re-scoped Phase 3 from a Hailuo payload fix into first-class fal provider work
  with explicit Seedance 2.0, Veo 3.1, Hailuo-02, and MiniMax Video-01 profiles.
- Selected fal's durable queue lifecycle and model-specific duration planning;
  documented safe retry, rate-limit, cancellation, and error semantics.
- Confirmed from fal's current prerelease note that Seedance 2.5 is not yet
  released and recorded a release checklist instead of inventing a fal schema.
- Identified request identity and Platform API reconciliation points for later
  timing and cost analysis while keeping metrics capture out of Phase 3.
- Reframed the future workflow around scenes, timed beats, curated visual
  reference packs, versioned takes, continuity state, and transition contracts
  so longer model outputs do not remain synonymous with shots.
- Added a bounded Seedance 2.0 image/video-reference vertical slice after the fal
  transport work, while keeping the unreleased Seedance 2.5 limits gated.
- Isolated audio references, audio timelines, splicing, mixing, and final muxing
  in Phase 7 so visual preparation and review phases remain modular.
- Implemented the first-class fal provider on `feat/fal-first-class-provider`:
  eight released endpoint IDs across four model families, queue lifecycle,
  safe retry and cancellation, exact payload/output contracts, profiled duration
  planning, ephemeral reconciliation metadata, updated configuration/docs, and
  mocked contract/lifecycle coverage. Seedance 2.5 remains release-gated.
