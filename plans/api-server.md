# SPEC — `ttv-pipeline` Service (Prompt-Only API; **Angie ⇄ Hypercorn HTTP/3 end-to-end**; GCS delivery)
**Mode:** Autonomous, spec-driven development agent
**Contract:** Prompt-only API → immediate Task ID → poll status → obtain GCS URL
**Source of Truth:** Existing repo configuration (credentials, buckets, backends, sizes, etc.)
**Stack Goals:** HTTP/3 end-to-end; Trio-first structured concurrency; safe cancellation; future-proof; asyncio fallback

---

## 1) Summary & Intent
Expose `ttv-pipeline` as a network service that accepts a **single input field (`prompt`)**, immediately returns a **Task ID**, supports **polling for status/logs**, and delivers the final video via **Google Cloud Storage (GCS)**. The service **must not** accept or require any other input parameters; all operational settings (backend, modes, sizes, credentials, buckets) come **exclusively from the repository’s existing configuration**.
**Infrastructure:** Public edge proxy is **Angie** (NGINX-compatible with **upstream HTTP/3**), upstream app server is **Hypercorn** (ASGI) with **native HTTP/3 (QUIC)**. This enables **H3 (client) → Angie → H3 (upstream) → Hypercorn** for a full QUIC pipeline. Trio-first with AnyIO; can fall back to asyncio where dependencies require.

---

## 2) Scope & Goals
- **Immediate acceptance** (`POST /v1/jobs`) returns **202 + Task ID**; no blocking on render.
- **Polling interface** to track job lifecycle + recent logs.
- **Final artifact delivery** via **GCS** (`gs://…`), with on-demand **signed HTTPS URL**.
- **Prompt override parity with CLI:** The HTTP `prompt` must override the config prompt **using the exact precedence rules** currently used by CLI arguments in `pipeline.py`.
- **Transport:** **HTTP/3 end-to-end** (client ↔ Angie ↔ Hypercorn) with TLS; support HTTP/2/1.1 fallback.
- **Concurrency model:** **Trio-first** (AnyIO), structured concurrency, cooperative cancellation; **asyncio fallback** allowed per dependency.

---

## 3) Constraints & Non-Goals
- **Do NOT add request-time parameters** for backend, sizes, modes, credentials, or buckets. The web API accepts only `prompt`.
- **Do NOT introduce new env vars** for pipeline settings, credentials, or bucket names. Use the repo configuration already in place.
- **Do NOT persist prompts** to config; overrides apply **per request**.
- **No UI** beyond HTTP API.
-**OpenAPI** expose standard OpenAPI /docs endpoint and openapi.yaml file with API specs
- This SPEC provides **requirements and guidance**; it **does not** produce file artifacts (app code, Dockerfiles, compose, or proxy configs).

---

## 4) Architecture (Logical)

1) Client (HTTP/3 over QUIC)
   - Users send requests over HTTP/3 (QUIC) and may transparently fall back to HTTP/2 or HTTP/1.1 when needed.

2) Edge Reverse Proxy — **Angie**
   - Public entry point on TCP/443 and UDP/443 (TLS 1.3).
   - Terminates client TLS, performs L7 routing, header management, and rate limiting.
   - Proxies **upstream via HTTP/3** to the application server to enable an end-to-end QUIC path (H3 client → Angie → H3 upstream).

3) Application Server — **Hypercorn** (ASGI; FastAPI app)
   - Runs the API service with native **HTTP/3** support (QUIC) and HTTP/2/1.1 fallback.
   - **Trio-first** via AnyIO for structured concurrency and cancellation; asyncio fallback where dependencies require.
   - Stateless request handlers:
     - Validate **prompt-only** input.
     - Enqueue background job.
     - Return **202 Accepted** with **Task ID** and `Location: /v1/jobs/{id}`.
     - Expose polling, logs, and artifact URL endpoints.

4) Queue / Broker (e.g., Redis/RQ)
   - Stores transient job state and metadata.
   - Decouples HTTP request latency from long-running video generation.
   - Supports cooperative cancellation signals propagated from the API.

5) Worker(s)
   - Consume queued jobs and execute the pipeline.
   - Build an **effective config** where the **HTTP `prompt` overrides** the config prompt with **the same precedence as CLI** in `pipeline.py` (HTTP > CLI > config).
   - Orchestrate pipeline subprocesses with structured cancellation (graceful `SIGTERM`, timed escalation to `SIGKILL`).
   - Emit structured logs and coarse progress milestones.

6) Staging Storage (Ephemeral)
   - Temporary workspace for intermediate frames, logs, and the locally rendered MP4 prior to upload.
   - Cleaned up on completion or cancellation.

7) Artifact Delivery — **Google Cloud Storage (GCS)**
   - Use **existing repo configuration** for credentials, bucket, and prefix (no new envs).
   - Upload final artifact to `gs://{bucket}/{prefix}/{YYYY-MM}/{job_id}/final_video.mp4`.
   - Record `gcs_uri` in job metadata for retrieval via the API.

8) Retrieval & Observability Paths
   - Clients **poll** `/v1/jobs/{id}` for status and `gcs_uri`.
   - Clients call `/v1/jobs/{id}/artifact` to receive an **on-demand signed HTTPS URL** (or public URL if signing is disabled); optional redirect.
   - `/v1/jobs/{id}/logs?tail=…` for recent log lines.
   - Health (`/healthz`), readiness (`/readyz`), and metrics (`/metrics`) for operations.

9) Alternate Deployment Mode (Dev/Test or Specialized Prod)
   - **Direct Hypercorn H3** exposure without Angie (end-to-end QUIC still preserved).
   - Maintain HTTP/2/1.1 fallback listeners for compatibility.

---

## 5) Configuration Source of Truth (Pipeline)
- On app startup, **load and validate the repository’s configuration**:
  - Backends/modes/sizes (e.g., `default_backend: veo3`, `generation_mode: "keyframe"`, `image_generation_model: "gemini-2.5-flash-image-preview"`, `image_size: "1024x1024"`), output roots, **credentials**, **GCS bucket/prefix**, provider keys, etc.
- The service layer must not introduce additional env vars for these pipeline settings.
- The only client-supplied value is `prompt` (string).
- Config refresh may be manual (restart) or watched; hot-reload optional.

---

## 6) API Design (Data Contracts)
### 6.1 Authentication
- Bearer token required for all endpoints except health/ready/metrics: `Authorization: Bearer <token>` (service-specific auth config; do not log tokens).

### 6.2 Endpoints & Contracts
**POST `/v1/jobs`** — *Create job (immediate accept)*
- **Request (JSON):**
  ```json
  {
    "prompt": "A man stands at a podium on a stage in front of a large audience, giving a presentation..."
  }
  ```
  - Required: `"prompt"` (non-empty string). **No other fields permitted.**
- **Response:** `202 Accepted` (recommended)
  ```json
  { "id": "job_123", "status": "queued" }
  ```
  - **Headers:** `Location: /v1/jobs/job_123`

**GET `/v1/jobs/{id}`** — *Poll status*
- **Response:** `200 OK`
  ```json
  {
    "id": "job_123",
    "status": "queued|started|progress|finished|failed|canceled",
    "progress": 0,
    "created_at": "2025-08-31T12:34:56Z",
    "started_at": "2025-08-31T12:35:10Z",
    "finished_at": null,
    "gcs_uri": null,
    "error": null
  }
  ```
  - `gcs_uri` populated after successful GCS upload:
    - e.g., `gs://<bucket>/<prefix>/<YYYY-MM>/<job_id>/final_video.mp4`

**GET `/v1/jobs/{id}/artifact`** — *Obtain downloadable URL*
- **Query:**
  - `expires_in` (seconds; optional; default from config)
  - `redirect` (boolean; default `false`)
- **Response (JSON or redirect):**
  ```json
  {
    "gcs_uri": "gs://bucket/prefix/2025-08/job_123/final_video.mp4",
    "url": "https://storage.googleapis.com/...signed...",
    "expires_in": 3600
  }
  ```
  - If signing disabled/unnecessary (public object), return public HTTPS URL with `expires_in: 0`.
  - `404` if artifact not ready; `302/307` redirect if `redirect=true`.

**GET `/v1/jobs/{id}/logs?tail=200`** — *Recent log lines* → `{ "lines": ["…","…"] }`

**POST `/v1/jobs/{id}/cancel`** — *Best-effort cancellation* → `{ "ok": true }`

**GET `/healthz`**, **GET `/readyz`**, **GET `/metrics`** — ops endpoints.

### 6.3 Errors (illustrative)
- `400` invalid/missing `prompt`; `401/403` auth failures; `404` unknown task/artifact not ready; `409` cancel race (optional); `5xx` internal (never leak secrets/URLs/paths).

---

## 7) Job Lifecycle, State, and Cancellation Semantics
- **States:** `queued → started → progress (n%) → finished | failed | canceled`
- **Progress:** Coarse milestones (plan → keyframes → generation → stitch → upload).
- **Timestamps:** `created_at`, `started_at`, `finished_at`.
- **Cancellation (Trio-first):**
  - API `cancel` requests place a **cancellation token** on the job.
  - Worker logic must **cooperatively** cancel:
    - Cancel AnyIO/Trio task groups,
    - Gracefully terminate child subprocess(es) running the pipeline (send `SIGTERM`, escalate to `SIGKILL` on deadline),
    - Ensure partial files are cleaned or marked, and job status becomes `canceled`.
- **Retention:** Metadata kept per policy (e.g., 7–30 days). Artifacts canonical in GCS; lifecycle rules govern retention.

---

## 8) Prompt Override Parity (Required Pipeline Contract Change)
- **Requirement:** HTTP `prompt` must override the config default **with the same behavior and precedence** as CLI arguments in `pipeline.py`.
- **Implementation guidance:**
  - **Refactor** current CLI/config merge into a single function, e.g., `build_effective_config(base_config, cli_args, http_overrides)`.
  - **Precedence (explicit):** `HTTP prompt` **>** `CLI prompt` **>** `config prompt`.
  - HTTP override applies **only to the prompt**; **no other fields** are overridden by HTTP.
  - **Tests**:
    - (a) config-only,
    - (b) config+CLI,
    - (c) config+HTTP,
    - (d) config+CLI+HTTP (HTTP wins).

---

## 9) Storage & Artifact Delivery (GCS)
- Use **existing config** for GCS bucket, prefix, and credentials; **no new envs**.
- **Object key spec:**
  - `gs://{bucket}/{prefix}/{YYYY-MM}/{job_id}/final_video.mp4`
- **Upload:** Worker uploads the final video on success, then sets `gcs_uri` in job metadata.
- **Signed URL policy:** API generates signed HTTPS URLs **on demand** (do not store them); fallback to public HTTPS URL if signing is disabled/unnecessary.
- **Security:** Never log full signed URLs; redact query parameters.

---

## 10) Security Requirements
- **Auth:** Bearer token for non-health endpoints; service-level secret separate from pipeline config; do not log secrets.
- **Transport:** TLS on public and upstream legs.
  - **End-to-end H3:** Client (H3/TLS) → Angie (TLS) → Hypercorn (H3/TLS).
  - Maintain **HTTP/2/1.1 fallback** paths (TCP).
- **Headers:** Preserve `X-Forwarded-For`, `X-Forwarded-Proto`, `Host`.
- **Rate limiting:** Enforce at edge (Angie); optional service quotas.

---

## 11) Observability & Ops
- **Metrics:** Request counts/latencies, queue depth, job durations, failure rates, phase timings, GCS upload metrics.
- **Logs:** Structured JSON for app/worker; redact secrets and signed URL params.
- **Health/Readiness:** Liveness probes; readiness includes broker connectivity.
- **Tracing:** Optional OpenTelemetry hooks (phase-2).

---

## 12) Deployment & Networking (Guidance; not files)
### 12.1 Components
- **Edge Proxy:** **Angie**
  - **Downstream:** listen on `:443` (TLS) with **HTTP/3 + HTTP/2**.
  - **Upstream:** **HTTP/3** to Hypercorn (QUIC) to enable full H3 pipeline.
  - Features: L7 routing, header mgmt, rate limiting, keepalives.
- **App Server:** **Hypercorn** (FastAPI ASGI)
  - **Native H3** with `aioquic` (`hypercorn[h3]`).
  - **Runtime:** **Trio** via `--worker-class trio` (AnyIO); **asyncio fallback** permitted where needed.
  - **Bindings:**
    - TCP (HTTP/1.1/2) e.g., `0.0.0.0:8000` for fallback,
    - QUIC (HTTP/3) e.g., `0.0.0.0:8443` with TLS cert/key.
- **Broker/Queue:** Redis/RQ or equivalent.
- **Workers:** 1..N depending on throughput; optional GPU workers per repo config.
- **Volumes:** Ephemeral staging (/data) for logs/artifacts pre-upload; mounts for existing config and credentials.

### 12.2 Ports & Firewalls
- **Public:** TCP/443 and **UDP/443** open to clients (Angie).
- **Internal:** App QUIC (e.g., UDP/8443) open from Angie to Hypercorn; TCP/8000 for H2/H1 fallback.

### 12.3 TLS & Certificates
- **Angie (edge):** Public certificate chain & key (TLS 1.3).
- **Hypercorn (upstream H3):** TLS cert/key required (could be internal CA).
- **Trust model:** If using private CA, ensure Angie trusts Hypercorn’s cert chain.
- **Rotation:** Define renewal/rollover procedure; avoid downtime for QUIC sockets.

### 12.4 Reference Configuration (for the agent to adapt; **do not** commit as files)
- **Hypercorn (illustrative CLI):**
  ```
  hypercorn api:app \
    --bind 0.0.0.0:8000 \
    --worker-class trio --workers 2 \
    --quic-bind 0.0.0.0:8443 \
    --certfile /certs/api.crt \
    --keyfile /certs/api.key
  ```
  - TCP `:8000`: HTTP/1.1/2 fallback; QUIC `:8443`: HTTP/3.
- **Angie (illustrative snippet):**
  ```
  events {}

  http {
    upstream api_h3 {
      server api:8443;
      keepalive 16;
    }

    server {
      listen 443 ssl http2;
      listen 443 quic reuseport;

      ssl_certificate     /etc/ssl/edge/fullchain.pem;
      ssl_certificate_key /etc/ssl/edge/privkey.pem;

      add_header Alt-Svc 'h3=":443"; ma=86400' always;

      location / {
        proxy_http_version 3;        # upstream via HTTP/3 to Hypercorn
        proxy_pass https://api_h3;

        proxy_set_header Host              $host;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
      }
    }
  }
  ```
  - Ensure UDP/443 + TCP/443 open; QUIC mandates TLS 1.3 across both legs.

### 12.5 Dev & CI Guidance
- **Local dev:** Run Hypercorn on TCP only (e.g., `--bind 127.0.0.1:8000`) for simplicity; add QUIC in later stages.
- **Integration tests:** Use QUIC client (`curl --http3 -I https://…`) to validate H3 end-to-end; assert Alt-Svc and negotiated protocol.
- **Load tests:** Validate back-pressure and cancellation behavior; confirm non-blocking `POST /v1/jobs`.

---

## 13) Concurrency & Cancellation (Trio-First)
- **AnyIO + Trio** in the app for request handling, background tasks, and log tailing.
- **Task groups** for orchestration; **cancellation scopes** on `cancel`.
- **Subprocess control** in workers:
  - Spawn pipeline processes with deadline; on cancel, send `SIGTERM`, await grace period, then `SIGKILL`.
  - Ensure cleanup hooks (temp files, partial artifacts) and consistent terminal state (`canceled`).
- **Asyncio fallback**: Where libraries lack Trio support, run on asyncio executor or isolate in worker processes.

---

## 14) Non-Functional Requirements
- **Responsiveness:** `POST /v1/jobs` returns ≤ 250 ms under nominal load.
- **Throughput:** Governed by worker count and backend/provider limits.
- **Reliability:** Bounded retries for GCS upload with exponential backoff; resumable upload optional (phase-2).
- **Back-pressure:** Queue depth thresholds; return `429` with `Retry-After` or shed load (optional for phase-1).
- **Privacy:** Prompts and logs handled per policy; no PII exposure.

---

## 15) Acceptance Criteria (Definition of Done)
1. **Prompt-only** `POST /v1/jobs` returns **202 + Task ID** immediately and sets `Location: /v1/jobs/{id}`.
2. **Polling** reflects lifecycle transitions with timestamps and progress; **no** other request parameters influence run behavior.
3. Completed jobs expose **`gcs_uri`** in `GET /v1/jobs/{id}`.
4. `GET /v1/jobs/{id}/artifact` returns a **working HTTPS URL** (signed or public) consistent with config credentials/bucket.
5. **Prompt override parity** with CLI is demonstrated by tests; HTTP prompt supersedes config prompt exactly as CLI would.
6. **HTTP/3 end-to-end** validated: client ↔ Angie (Alt-Svc, QUIC negotiated) and **Angie ↔ Hypercorn** upstream over HTTP/3 (observable in proxy/app logs).
7. **Structured cancellation** works (user cancel transitions to `canceled`, subprocesses terminated safely).
8. **No secret leakage** in logs (tokens, credentials, signed URL params).

---

## 16) Implementation Plan (Agent Tasks)
- **A. Config Integration**
  - Load existing repo config; validate required keys (backend/mode/sizes, GCS bucket/prefix/creds).
- **B. Prompt Override Refactor**
  - Centralize config/CLI/HTTP merge; enforce precedence (`HTTP > CLI > config`); add unit tests.
- **C. API Layer**
  - Implement prompt-only `POST /v1/jobs` (202 + Task ID + Location); `GET /v1/jobs/{id}`, `/artifact`, `/logs`, `/cancel`, health/ready/metrics.
- **D. Worker Path**
  - Invoke pipeline with effective config (only prompt overridden); structured logging; progress milestones; upload to GCS; set `gcs_uri`.
- **E. H3 Stack Enablement**
  - Prepare Hypercorn H3 bindings (QUIC + TCP fallback).
  - Provide Angie upstream H3 proxy guidance (Section 12.4) and verify two-leg H3.
- **F. Observability**
  - Metrics, structured logs, health/readiness; redact secrets.
- **G. Test & Hardening**
  - Unit/contract/e2e tests; QUIC validation in CI; cancellation tests; artifact URL tests.

---

## 17) Test Plan
- **Unit:** request schema; config loader; merge precedence; signed-URL generator (mocked GCS); log redaction.
- **Contract:** `POST /v1/jobs` → 202 + Task ID; poll to `finished`; `gcs_uri` present; `/artifact` returns working URL or redirects.
- **E2E:** CPU path using config default backend; confirm object at expected `gs://…` key; downloadable via HTTPS; QUIC path validated via `curl --http3`.
- **Cancellation:** Submit job; call `cancel`; verify cooperative shutdown, subprocess termination, and `canceled` state.
- **Transport:** Verify **client ↔ Angie** and **Angie ↔ Hypercorn** both negotiate HTTP/3; fallback paths (H2/H1) remain functional.

---

## 18) Risks & Mitigations
- **Override mismatch (HTTP vs CLI)** → Single merge function + tests.
- **GCS upload failures** → Retries with backoff; clear error surfacing; local artifact retained for debugging.
- **Queue saturation** → Optional 429; autoscale workers; ops runbook.
- **TLS/QUIC complexity** → Automate certificate provisioning; validate chain/trust for both legs; expose diagnostics (Alt-Svc, negotiated protocol).
- **Library Trio gaps** → AnyIO abstractions; asyncio fallback; subprocess isolation.

---

### Quick Reference — Client Input vs System Behavior
- **Client sends:**
  - `prompt` (required): e.g., “A man stands at a podium on a stage in front of a large audience, giving a presentation…”
- **System uses (from repo config only):**
  - `default_backend` (e.g., `veo3`), `generation_mode` (e.g., `"keyframe"`),
  - `image_generation_model` (e.g., `"gemini-2.5-flash-image-preview"`), `image_size` (e.g., `"1024x1024"`),
  - **credentials**, **GCS bucket/prefix**, and all other pipeline parameters.
````


