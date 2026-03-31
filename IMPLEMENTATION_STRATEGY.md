# Blender HDRI Plugin MVP Implementation Strategy

This document turns the current repo into a concrete build plan for the first productized version of the Blender plugin and API.

## Product goal

Given a single user image:

1. generate a plausible 2:1 equirectangular panorama,
2. lift that LDR panorama into an HDRI-like `.hdr`,
3. apply it to Blender World lighting,
4. keep the Blender addon thin and provider-agnostic,
5. support an initial free trial and later credit-based billing.

The priority is **visually convincing lighting in Blender**, not physical accuracy.

## Current repo status

### What already exists

- `hdri_from_image_addon.py`
  - Blender addon UI and API client
  - uploads an image to the backend
  - downloads a returned HDR URL
  - applies it to the scene World
  - can add a preview sphere

- `hdri_api_server/app.py`
  - FastAPI backend
  - accepts `POST /v1/hdri`
  - runs panorama generation
  - applies heuristic HDR lift
  - writes `.hdr`
  - returns a signed file URL

- `hdri_api_server/panorama.py`
  - panorama backend abstraction
  - supports `resize`, `replicate`, `http_json`, `hf_dit360`

### Main gap

The repo already has working Blender-to-backend plumbing, but the default panorama path is still placeholder-quality:

- `resize` stretches the source image to 2:1
- the current HDR step is a heuristic lift, not learned HDR reconstruction

That means the next product milestone is not "build the plugin from scratch", but:

> replace the placeholder panorama generation path with a real hosted provider flow, then make the API robust enough for long-running jobs and basic monetization.

## Recommended architecture

Keep this boundary:

`Blender addon -> our API -> provider adapter -> panorama result -> HDR lift -> signed file URL`

Do **not** make the Blender addon call Replicate directly.

### Why this is the right boundary

- API keys stay server-side
- provider can change later without updating the addon contract
- usage metering and free quotas live in one place
- caching can reduce repeated spend
- long-running jobs can be tracked reliably
- later migration to self-hosted workers is easier

## Provider recommendation

### MVP provider

Use **Replicate first** as the panorama provider for experimentation and early product validation.

### But do not lock the product to Replicate

The long-term abstraction should remain generic:

- either keep `PANORAMA_MODE=replicate` temporarily,
- or prefer a provider-normalizing worker behind `http_json`

The important architectural decision is that the Blender addon should only know **our API**.

## Product rollout recommendation

### Launch shape

- output size: **1024x512**
- output format: **Radiance `.hdr`**
- Blender renderer target: **Cycles**
- generation path:
  - external panorama provider
  - server-side heuristic HDR lift
  - optional style preset

### Free usage recommendation

Start with:

- **3 free generations per account**

Reasoning:

- current hosted panorama generation can be expensive enough that 8 freebies is risky for launch
- 3 generations are enough for a user to understand the value
- it gives room for:
  - one outdoor attempt
  - one indoor attempt
  - one retry or preset variation

After launch, revisit the number using real data:

- cost per successful generation
- conversion rate after free usage
- cache hit rate
- provider failure rate

## MVP scope

### In scope

- provider-backed panorama generation
- async jobs API
- Blender polling flow
- signed `.hdr` downloads
- style presets
- basic account quota logic
- 3 free generations

### Out of scope for first release

- subscriptions
- multiple paid tiers beyond simple credits
- search engine / similar HDRIs
- database-backed HDRI marketplace
- self-hosted GPU inference
- model fine-tuning
- true physically-based HDR reconstruction

## Suggested rollout phases

## Phase 1 - Replace placeholder panorama generation

### Goal

Make the generation path actually useful for real users.

### Tasks

1. choose and wire a Replicate panorama model
2. normalize the returned image to 1024x512 RGB
3. continue using the existing heuristic HDR lift
4. preserve the signed file delivery flow

### Exit criteria

- works on a small manual test set
- results are visually usable in Blender
- major errors are surfaced clearly

## Phase 2 - Convert the API to async jobs

### Why

Hosted image generation can take much longer than a single synchronous HTTP request should tolerate.

### New flow

1. Blender uploads image and settings
2. API creates a job record
3. API returns `job_id`
4. Blender polls job status
5. API returns signed file URL when complete

### Exit criteria

- no long blocking request from Blender to provider-backed generation
- retries and failures are observable
- addon UX is resilient to provider latency

## Phase 3 - Add quota and free generation tracking

### Goal

Support an early product launch without anonymous abuse.

### Exit criteria

- authenticated requests
- quota decremented only for non-cached work
- failed jobs restore the deducted quota

## Phase 4 - Improve image quality

### Goal

Increase Blender usefulness before investing in custom training.

### Likely improvements

- seam-safe wrap refinement
- pole-aware smoothing
- stronger preset tuning
- later add sun-lobe enhancement on top of the current HDR lift

## API design

Replace the current single synchronous endpoint with job-oriented endpoints while keeping the existing file serving endpoint.

### Proposed endpoints

#### `POST /v1/jobs/hdri`

Create a generation job.

Request body:

```json
{
  "provider": "D",
  "image_b64": "<base64>",
  "scene_mode": "auto",
  "quality_mode": "balanced",
  "preset": "none",
  "output_width": 1024,
  "output_height": 512,
  "assume_upright": true,
  "panorama_prompt": null,
  "panorama_negative_prompt": null,
  "panorama_seed": null,
  "panorama_strength": null,
  "panorama_extra": null
}
```

Response:

```json
{
  "job_id": "uuid",
  "status": "queued"
}
```

#### `GET /v1/jobs/{job_id}`

Return job status and final result metadata.

Response while running:

```json
{
  "job_id": "uuid",
  "status": "running"
}
```

Response on success:

```json
{
  "job_id": "uuid",
  "status": "succeeded",
  "hdri_url": "http://127.0.0.1:8000/v1/files/<id>.hdr?exp=...&sig=...",
  "exr_url": "http://127.0.0.1:8000/v1/files/<id>.hdr?exp=...&sig=...",
  "width": 1024,
  "height": 512,
  "format": "hdr_rgbe",
  "panorama_mode": "replicate"
}
```

Response on failure:

```json
{
  "job_id": "uuid",
  "status": "failed",
  "error": "provider timeout or generation failure"
}
```

#### `GET /v1/files/{id}.hdr?exp=...&sig=...`

Keep this signed download endpoint as the file delivery mechanism.

### Optional backward compatibility

Keep `POST /v1/hdri` temporarily as:

- a compatibility endpoint for older addon versions, or
- a thin synchronous shim for local/testing modes only

But new provider-backed generation should use the jobs API.

## Job model

Add a small persistence layer. For MVP this can be SQLite.

### Suggested job fields

- `job_id`
- `user_id`
- `status`
  - `queued`
  - `running`
  - `succeeded`
  - `failed`
- `created_at`
- `updated_at`
- `input_hash`
- `scene_mode`
- `quality_mode`
- `preset`
- `panorama_mode`
- `provider_job_id`
- `result_file_id`
- `error_message`
- `charged_unit_type`
  - `free`
  - `credit`
  - `cache`
- `charged_units`

### Why `input_hash` matters

It enables:

- deduping identical requests
- cheap caching
- fairer billing behavior
- lower provider spend

The hash should include:

- normalized image bytes
- output size
- scene mode
- quality mode
- preset
- panorama-specific parameters

## Quota and billing model

### Initial user model

For launch, keep it very simple:

- `free_generations_remaining`
- `paid_credits_remaining`

### Suggested starting values

- `free_generations_remaining = 3`
- `paid_credits_remaining = 0` until purchases exist

### Charging rules

When a request arrives:

1. authenticate the user
2. check for cache hit using the request hash
3. if cached:
   - return cached result
   - do not charge a free generation
4. if uncached:
   - use a free generation if available
   - otherwise use a paid credit
5. if generation fails:
   - refund the deducted free generation or credit

### Recommended simplification for MVP

Use one charging unit for all 1024x512 jobs:

- `1 job = 1 generation`

Defer multi-tier credit pricing until the quality tiers and cost profile are stable.

## Blender addon changes

The addon should stay thin. The main change is moving from a synchronous request to a submit-and-poll flow.

### Current state

The addon submits one request and expects the result in one pass.

### Planned state

1. user chooses image and settings
2. addon calls `POST /v1/jobs/hdri`
3. addon stores `job_id`
4. addon polls `GET /v1/jobs/{job_id}`
5. addon downloads the returned signed HDR URL
6. addon applies it to World nodes

### UX considerations

- show job state in the panel:
  - queued
  - running
  - failed
  - succeeded
- allow user cancellation client-side even if provider work continues server-side
- keep yaw and exposure editable after download
- keep preview sphere optional

## Backend provider adapter design

The current repo already has a useful abstraction point in `panorama.py`.

### Recommendation

Keep the panorama provider logic behind a single internal function boundary:

- input:
  - source image
  - target width and height
  - scene mode
  - quality mode
  - optional prompt controls
- output:
  - RGB panorama image
  - provider/mode metadata

### Replicate-specific concerns

- long-running predictions
- polling timeout behavior
- output shape variability
- possible need for model-specific field names

### Recommendation for future-proofing

Even if Replicate is used first, keep its request normalization isolated so it can later be swapped with:

- Fal
- a custom `http_json` worker
- self-hosted ComfyUI
- an in-house GPU worker

## Persistence recommendation

For MVP, use SQLite rather than introducing a full external database.

SQLite is enough for:

- jobs
- user quota
- cache lookup
- basic usage records

Suggested tables:

- `users`
- `jobs`
- `job_results`
- `usage_events`

## Logging and metrics to capture from day one

Track these early, even if only to SQLite or structured logs:

- request count
- success count
- failure count
- cache hit count
- provider latency
- total end-to-end latency
- average cost per successful generation
- average free generations consumed before conversion

This data should decide whether to move from 3 free generations to 5 later.

## Manual test plan for the first provider-backed version

Build a small benchmark set:

- 5 outdoor photos
- 5 indoor photos
- 5 studio or object-focused images

For each result, judge:

- seam continuity
- horizon plausibility
- dominant light direction believability
- usefulness for reflections
- usefulness for Cycles lighting
- whether presets materially change the result in a good way

Keep a simple score sheet so provider choice and heuristic improvements are based on evidence.

## Concrete repository change plan

## Step 1 - API job scaffolding

Files likely to change:

- `hdri_api_server/app.py`

Files likely to be added:

- `hdri_api_server/db.py`
- `hdri_api_server/models.py`
- `hdri_api_server/jobs.py`

First implementation target:

- add SQLite-backed job records
- add `POST /v1/jobs/hdri`
- add `GET /v1/jobs/{job_id}`
- keep signed file serving unchanged

## Step 2 - Blender polling support

Files likely to change:

- `hdri_from_image_addon.py`

Implementation target:

- submit job
- poll status
- display status in the panel
- preserve existing World setup behavior

## Step 3 - Quota support

Files likely to change:

- `hdri_api_server/app.py`
- new persistence/helpers under `hdri_api_server/`

Implementation target:

- API key to user lookup
- free generation accounting
- refund-on-failure behavior
- cache-aware charging

## Step 4 - Provider hardening

Files likely to change:

- `hdri_api_server/panorama.py`
- optionally a new provider-specific helper file

Implementation target:

- harden Replicate integration
- normalize output handling
- capture provider metadata and latency

## Risks and mitigations

### Risk: hosted provider is too slow

Mitigation:

- async jobs
- polling UX
- queue status visibility

### Risk: provider quality is inconsistent

Mitigation:

- benchmark across a fixed test set
- keep provider boundary abstract
- switch providers without changing addon contract

### Risk: free tier is abused

Mitigation:

- require user identity or API key
- only charge for successful non-cached work
- start with 3 freebies, not 8

### Risk: current HDR lift looks fake in some scenes

Mitigation:

- improve seam handling first
- add sun-lobe enhancement later
- tune presets before attempting full model training

## Recommended next coding pass

Implement in this order:

1. add async jobs API
2. update the Blender addon to submit and poll
3. persist jobs in SQLite
4. add quota tracking with 3 free generations
5. harden the Replicate-backed panorama path

This ordering produces the smallest useful product step while preserving the current architecture.
