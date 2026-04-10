# Scaling Migration Plan

This project currently uses:

- SQLite for jobs/accounts/tokens
- local disk for generated `.hdr` files

That is acceptable for single-instance deployment. For multi-instance and higher reliability, migrate in two steps.

## Step 1: Object storage for generated files

Goal: remove local-disk dependency for download files.

Targets:

- S3, Cloudflare R2, or Backblaze B2 (S3-compatible)
- generate pre-signed download URLs instead of local `/v1/files` path-only storage

Implementation path:

1. Add storage backend abstraction (`local` and `s3`).
2. On generation success, upload `.hdr` to object storage and store object key in job result.
3. Return signed object URL in `hdri_url` / `exr_url`.
4. Keep local backend as fallback for dev.

Cutover checks:

- multiple API instances can serve the same job results
- files survive container restarts
- lifecycle policies auto-delete old artifacts

## Step 2: Managed Postgres for jobs/accounts/tokens

Goal: remove SQLite write-concurrency limits and local state coupling.

Targets:

- Neon, Supabase, RDS, or Cloud SQL

Implementation path:

1. Create a new datastore layer matching existing `JobStore` operations:
   - accounts lookup/update
   - API key lookup
   - job create/update/get
   - usage event inserts
2. Run migration script that imports:
   - `accounts`
   - `api_keys`
   - `jobs`
   - `usage_events`
3. Dual-run in staging and compare job/account reads.
4. Switch production to Postgres store via env flag.

Cutover checks:

- no token accounting drift after switch
- stale-job reaper works with new backend
- cancel/refund idempotency still guaranteed

## Suggested migration thresholds

Start migration when one or more are true:

- more than one API instance is needed
- restart persistence issues appear
- job volume exceeds SQLite comfort under peak concurrency
- ops needs point-in-time backups and stronger auditing

## Operational defaults after migration

- object store retention: 7-30 days for generated files
- database backups: daily snapshot + point-in-time recovery
- monitoring: job failure rate, refund rate, stale timeout rate
