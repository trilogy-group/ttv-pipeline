# TTV Pipeline Recovery and Product Roadmap

Status: Phase 2 complete; Phase 3 pending
Last updated: 2026-07-18
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

## Phase 3 - Correct fal.ai support

Status: pending

Goal: keep fal as a useful provider without pretending every model shares one
schema.

Required work:

- Fix the configured Hailuo path: it accepts 6 or 10 seconds, while the pipeline
  currently sends 5.
- Correct capability claims; the current adapter requires an image despite
  advertising text-to-video.
- Add request-payload assertions to tests.
- Support only concrete model schemas needed by the product. Do not build a
  universal adapter speculatively.

## Phase 4 - Durable staged project/storyboard API

Status: pending
Execution: OpenSymphony/spec-driven

Goal: replace the monolithic ephemeral job with a resumable project workflow.

Initial architecture:

- Store a project/storyboard manifest in existing Redis.
- Store keyframes and segment-video versions in existing GCS.
- Add a database only when retention or query requirements exceed Redis.
- Keep final render/stitch operations as ordinary jobs.

Stages/actions:

1. Create project and generate storyboard.
2. Edit or replan storyboard segments.
3. Generate all keyframes.
4. Regenerate or prompt-edit one keyframe.
5. Approve keyframes individually or in one batch.
6. Render or regenerate individual video segments.
7. Select approved segment versions and stitch the final video.

Required semantics:

- Version prompts, keyframes, and video segments; do not silently overwrite.
- Changing shared boundary keyframe `N` invalidates the segments on both sides
  that consume it.
- Expose provider/model capabilities so unsupported UI actions are hidden.
- Make each stage resumable and safe to retry without duplicating paid jobs.

## Phase 5 - Low-touch frontend

Status: pending
Execution: smoke UI as one-off; full workflow spec-driven

Goal: make the common path nearly automatic while allowing intervention at
high-value review points.

Suggested UX:

1. Brief: prompt, requested duration, provider/model, aspect ratio.
2. Storyboard: editable segment cards with a single accept-all path.
3. Keyframes: generate all, then one gallery review gate with approve-all,
   regenerate, and edit-with-instructions controls.
4. Render: per-segment previews and exception-based retry/edit controls.
5. Final stitch and playback.

Implementation order:

- First, a same-origin HTML/CSS/JS smoke frontend served by FastAPI.
- Adopt React or another framework only after the staged workflow demonstrates
  that native UI state management is the constraint.

## Phase 6 - Instruction-based video-to-video editing

Status: pending
Execution: provider spike first; OpenSymphony/spec-driven if retained

Goal: regenerate an individual segment from its current video plus natural
language instructions.

Approach:

- Spike one concrete provider/model first; fal Kling O1 is the current candidate.
- Do not force video editing through the image-to-video generator interface.
- Keep original and edited segment versions.
- Require review before an edit becomes the selected segment.
- Re-stitch without rerendering unaffected segments.
- Generalize the provider contract only after a second implementation proves
  that schemas share useful structure.

## Deferred specifications

Create these only after the outline and relevant spike are approved:

- Veo 3.1 and requested-duration implementation spec.
- Durable project/storyboard domain and API spec.
- Artifact versioning and invalidation spec.
- Full HITL frontend interaction spec.
- Video-to-video editing spec, conditional on spike results.

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
