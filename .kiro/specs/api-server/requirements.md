# Requirements Document

## Introduction

This feature implements a REST API server that exposes the existing ttv-pipeline as a network service. The API accepts only a prompt parameter, immediately returns a task ID, supports polling for status, and delivers final videos via Google Cloud Storage. The service uses HTTP/3 end-to-end (client ↔ Angie ↔ Hypercorn) with Trio-first structured concurrency for optimal performance and cancellation support.

## Requirements

### Requirement 1

**User Story:** As a client application developer, I want to submit video generation requests via HTTP API, so that I can integrate video generation into my application without running the pipeline locally.

#### Acceptance Criteria

1. WHEN a client sends a POST request to `/v1/jobs` with a valid prompt THEN the system SHALL return HTTP 202 with a task ID and Location header
2. WHEN a client sends a POST request without a prompt THEN the system SHALL return HTTP 400 with an error message
3. WHEN a client sends a POST request with additional parameters beyond prompt THEN the system SHALL reject the request with HTTP 400
4. WHEN the system receives a valid job request THEN it SHALL use only the repository's existing configuration for all pipeline settings

### Requirement 2

**User Story:** As a client application, I want to poll job status and retrieve results, so that I can track progress and obtain the generated video when complete.

#### Acceptance Criteria

1. WHEN a client polls GET `/v1/jobs/{id}` THEN the system SHALL return current status, progress, timestamps, and gcs_uri when available
2. WHEN a job is completed successfully THEN the system SHALL populate the gcs_uri field with the GCS location
3. WHEN a client requests GET `/v1/jobs/{id}/artifact` THEN the system SHALL return a signed HTTPS URL for downloading the video
4. WHEN a client requests GET `/v1/jobs/{id}/logs` THEN the system SHALL return recent log lines for the job
5. WHEN a client sends POST `/v1/jobs/{id}/cancel` THEN the system SHALL attempt cooperative cancellation and update job status

### Requirement 3

**User Story:** As a system administrator, I want the API to use HTTP/3 end-to-end with proper fallbacks, so that clients get optimal performance while maintaining compatibility.

#### Acceptance Criteria

1. WHEN clients connect to the service THEN the system SHALL support HTTP/3 over QUIC as the primary protocol
2. WHEN HTTP/3 is not available THEN the system SHALL gracefully fallback to HTTP/2 or HTTP/1.1
3. WHEN the edge proxy receives requests THEN it SHALL forward them to the application server via HTTP/3
4. WHEN TLS certificates are configured THEN the system SHALL use TLS 1.3 for all encrypted connections

### Requirement 4

**User Story:** As a developer, I want prompt override behavior to match CLI precedence, so that HTTP requests have consistent behavior with command-line usage.

#### Acceptance Criteria

1. WHEN an HTTP request provides a prompt THEN it SHALL override the configuration prompt with the same precedence as CLI arguments
2. WHEN both CLI and HTTP prompts are provided THEN the HTTP prompt SHALL take precedence
3. WHEN only configuration prompt exists THEN the system SHALL use the configuration value
4. WHEN the system builds effective configuration THEN it SHALL use the precedence: HTTP > CLI > config

### Requirement 5

**User Story:** As a system operator, I want basic security controls for internal network deployment, so that the API operates safely within our infrastructure.

#### Acceptance Criteria

1. WHEN the system logs requests THEN it SHALL NOT log sensitive credentials or signed URL parameters
2. WHEN generating signed URLs THEN the system SHALL use existing GCS credentials from repository configuration
3. WHEN handling CORS requests THEN the system SHALL allow configured origins for internal network access
4. WHEN processing requests THEN the system SHALL include security headers appropriate for internal services

### Requirement 6

**User Story:** As a system administrator, I want structured concurrency and proper cancellation, so that resources are managed efficiently and jobs can be safely terminated.

#### Acceptance Criteria

1. WHEN using Trio for concurrency THEN the system SHALL use structured task groups and cancellation scopes
2. WHEN a job is cancelled THEN the system SHALL send SIGTERM to child processes and escalate to SIGKILL if needed
3. WHEN cancellation occurs THEN the system SHALL clean up temporary files and update job status to "canceled"
4. WHEN dependencies require asyncio THEN the system SHALL provide appropriate fallback mechanisms

### Requirement 7

**User Story:** As a system operator, I want comprehensive observability and health monitoring, so that I can monitor system performance and troubleshoot issues.

#### Acceptance Criteria

1. WHEN the system is running THEN it SHALL expose `/healthz`, `/readyz`, and `/metrics` endpoints
2. WHEN processing requests THEN the system SHALL emit structured JSON logs with appropriate log levels
3. WHEN errors occur THEN the system SHALL log sufficient context without exposing sensitive information
4. WHEN jobs are processed THEN the system SHALL track metrics for request counts, latencies, and failure rates

### Requirement 8

**User Story:** As a client, I want reliable artifact delivery via Google Cloud Storage, so that I can access generated videos through standard cloud storage mechanisms.

#### Acceptance Criteria

1. WHEN a job completes successfully THEN the system SHALL upload the final video to GCS using existing bucket configuration
2. WHEN uploading to GCS THEN the system SHALL use the path format: `gs://{bucket}/{prefix}/{YYYY-MM}/{job_id}/final_video.mp4`
3. WHEN clients request artifact URLs THEN the system SHALL generate signed URLs on-demand with configurable expiration
4. WHEN GCS upload fails THEN the system SHALL retry with exponential backoff and update job status appropriately