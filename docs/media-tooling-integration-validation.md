# Media Tooling integration validation

Validated on Python 3.14.3 with FFmpeg/FFprobe on 2026-09-22. The integration is based on `4c20bfc774ebd96c60d9b1988bf810c5165467e1`. Local checks, a live process-boundary pilot and primary MiniMax H3 Max generation, rendering and verification pass.

## Primary endpoint: MiniMax H3 Max

Use [`examples/pipeline_config.h3-max.yaml`](../examples/pipeline_config.h3-max.yaml) and set `FAL_KEY` or `FAL_API_KEY` in the process environment. The exact endpoint is `minimax/h3-max/image-to-video`. Its profile uses integer durations from 5 through 15 seconds, uppercase resolution values, safety checking enabled, and `prompt_expansion_mode: disabled` by default. It supports a first image, an optional ending image and an output soundtrack.

The live case used a funded Fal account, supplied first image, 5 seconds, `768P`, seed 42, one allowed attempt, no fallback and no paid image generation. The documented endpoint quote on 2026-09-22 was $0.20 for this case through September 30 ($0.40 afterward). The quote is separate from billing: adapter USD estimates and actual charges remain null, so the exact approval explicitly allows unknown cost with a $1 ceiling on known estimates. The provider reported `X-Fal-Billable-Units: 8.0`, retained without conversion to USD. [Fal endpoint and pricing](https://fal.ai/models/minimax/h3-max/image-to-video).

The production generator factory and generation service completed exactly one submission and one attempt in 7.86 seconds, retaining provider request ID `01a0c9a1-4567-73e1-a9d6-b0a440ec3ba8`. Duplicate approval and terminal result collection reused that result. The source is H.264 at 1344×768 and 24 fps, with a 5.166667-second video stream and a 5.184-second container. Its AAC stereo soundtrack is 32 kHz and non-silent (mean −21.6 dB, peak −8.6 dB). The source asset hash is `sha256:9c6d164040984ab2199273b32ee1a07ecca9b5160a331759f984de61117e85b6`.

The shipped Media Tooling CLI imported the canonical handoff, selected the take, rendered it with normal loudness normalization and passed standalone media verification with zero blocking findings. A separate comparison of the attempt against the approved variant confirmed matching shot, provider, model and prompt. The rendered preview is 1920×1098 at 24 fps, with a 5.166667-second video stream, a 5.205-second container and AAC stereo audio at 48 kHz (mean −16.7 dB, peak −3.7 dB).

## Automated checks

The expanded focused producer suite passes **162 tests**:

```sh
python -m pytest tests/test_generation_integration.py tests/test_fal_generator.py tests/test_veo31_duration.py tests/test_keyframe_checkpoints.py tests/test_keyframe_generator_gemini.py tests/test_models.py tests/test_queue.py tests/test_middleware.py -q --tb=short
```

Coverage includes shared contracts, exact approval/expiry/capability guards, billing and budgets, cancellation, regeneration, both workers, HTTP/file canonical parity, prepared input snapshots, durable submission receipts and terminal recovery. Runtime regressions additionally cover v2 authentication and request validation, explicit local publication, optional ending frames, H3 integer payloads and metadata, unknown/resolution-specific Veo prices, safe HTTP failure diagnostics, no repeated paid POST after 429/503, and retaining an accepted request ID before validating lifecycle URLs.

Ruff and Black pass on the integration modules, contract producer, pilot and acceptance tests. Python compilation and `git diff --check` pass. `uv build --wheel --out-dir /tmp/ttv-integration-wheel` succeeds; the wheel contains all five schemas, ten fixtures, the hash manifest and contract README. The shared validator corpus matches in both repositories: seven accepted and fifteen rejected cases.

The existing worker/backend regression set has **90 passed and 21 failed**, with the same failing test names on the immutable base and the initial integration implementation:

```sh
python -m pytest tests/test_models.py tests/test_queue.py tests/test_video_worker.py tests/test_trio_video_worker.py tests/test_trio_integration.py tests/test_integration_mocked_backends.py tests/test_keyframe_generator_gemini.py -q --tb=short
```

Those failures include deployment configuration/credential-dependent legacy mocks, existing Trio expectations and API authentication fixtures. The artifact endpoint suite likewise matched its baseline: **1 passed, 17 failed**. The complete pre-existing suite is not green in this environment.

## Live HTTP, Redis and RQ

A real Hypercorn server, dedicated loopback Redis server and RQ `SpawnWorker` processed requests from the shipped Media Tooling CLI. The API used its normal lifespan/configuration and routes; deterministic providers replaced only the paid generation boundary. The API and worker shared a task-local durable filesystem. Redis used its own port, with no existing jobs affected.

Live requests confirmed HTTP 401 for missing/invalid tokens and normal handler responses only with a valid token on v2 planning, approval, results and cancellation. Capability/schema discovery remains public. `integration_publish_gcs: false` selects local file assets; deployed configurations retain GCS publication by default.

The three-scene job produced a single clip, a three-clip take, and an approved fallback after failure. It retained eight attempts. Targeted middle-scene regeneration retained four attempts. Each job had one worker completion, and duplicate approvals before and after completion returned the same job ID. SQLite job/results remained available after transient RQ records expired. Both edits rendered and passed Media Tooling verification, preserving outer take IDs and hashes and requiring an explicit stale-continuity decision.

The API, worker and dedicated Redis processes were stopped; their loopback ports were checked closed.

## Secondary live provider and storage

One real `veo-3.1-generate-preview` invocation completed through the production factory and generation service: a supplied image, 4 seconds, 1280×720, 24 fps, one attempt, no fallback and no image-generation charges. Its approved ceiling was $2 and its estimate was $1.60; actual USD remains null. Duplicate approval/result collection did not repeat the operation. Media Tooling imported the actual artifact chain, rendered the clip and passed verification. Its silent audio track required the existing `--no-loudnorm` option. [Google pricing](https://ai.google.dev/gemini-api/docs/pricing).

ADC was available, but a read of the configured GCS bucket returned HTTP 403. No smoke objects were created, and upload/download round-trip verification remains unavailable for that identity. No IAM, bucket or production configuration was changed.

## Reproduction and evidence

The provider-free pilot can be reproduced with:

```sh
python scripts/integration_pilot.py /tmp/ttv-media-pilot
# In Media Tooling:
uv run python scripts/ttv_offline_acceptance.py \
  --handoff /tmp/ttv-media-pilot/handoff \
  --asset-root /tmp/ttv-media-pilot --project /tmp/ttv-media-edit
```

The live task evidence directory is `/Users/magos/.codex/worktrees/ttv-media-integration/live-smoke-fprv23t8`. Sanitized evidence includes `after-auth-fix.json`, `queue-dedup.json`, `cleanup.json`, `gcs-readiness.json`, `media-client-fixed/live-acceptance.json`, `media-google-secondary/provider-acceptance.json`, `h3-funded-provider/{summary,wire-summary,source-probe}.json`, and `media-h3-funded/{provider-acceptance,approved-provenance,technical-verification}.json`. Canonical provider handoffs and immutable results remain alongside these records. Private logs and credentials are excluded from repository artifacts.

## Operational limits

- API and workers require the same durable integration filesystem.
- An interrupted active provider operation requires operator investigation; its durable receipt prevents replay under the same operation key. Terminal publication can be retried without provider replay.
- Provider support for ending frames does not schedule one automatically. Supplied `last_frame` references or explicit `integration_generate_last_frame: true` appear in the reviewed plan; an empty `last_frame_prompt` schedules no ending-frame work.
- A reviewed-keyframe video approval can use only variants represented in that immutable keyframe result.
- Standalone/v1 provenance sidecars retain their invocation and media; full Media Tooling import uses the v2 request/plan/approval/result chain.
