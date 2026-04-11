# Render Setup Guide

If you are using a **notebook + tunnel** for `api.richardandrys.com` instead of Render, see [NOTEBOOK_DOMAIN_SETUP.md](NOTEBOOK_DOMAIN_SETUP.md).

This guide is the optional **Render** hosted backend setup for:

- domain + website on WEDOS
- API backend on Render
- workflow execution on RunComfy

Recommended public layout:

- `richardandrys.com` -> portfolio website
- `richardandrys.com/hdri-addon` -> add-on landing page
- `api.richardandrys.com` -> FastAPI backend on Render

## 1) Push the repo to GitHub

Render should deploy from your GitHub repository.

## 2) Create the Render web service

In Render:

1. Create a new `Web Service`
2. Connect your GitHub repo
3. Render will detect `render.yaml`
4. Confirm the service named `hdri-api`

This repo uses:

- `render.yaml`
- `hdri_api_server/Dockerfile`

## 3) Set required env vars in Render

Render should have these values:

- `HDRI_PUBLIC_BASE_URL=https://api.richardandrys.com`
- `HDRI_SIGNING_SECRET=<strong-random-secret>`
- `HDRI_ADMIN_TOKEN=<strong-random-secret>`
- `RUNCOMFY_API_TOKEN=<your-token>`
- `RUNCOMFY_DEPLOYMENT_ID=<your-deployment-id>`

Already defined in `render.yaml` with safe defaults:

- `HDRI_REMOTE_PROVIDER=runcomfy`
- `RUNCOMFY_BASE_URL=https://api.runcomfy.net`
- `HDRI_REQUIRE_API_KEY=1`

Optional later:

- `RUNCOMFY_OVERRIDES_JSON`
- `RUNCOMFY_IMAGE_NODE_IDS`
- `RUNCOMFY_PROMPT_NODE_IDS`
- `RUNCOMFY_NEGATIVE_PROMPT_NODE_IDS`
- `RUNCOMFY_SEED_NODE_IDS`
- `RUNCOMFY_STRENGTH_NODE_IDS`
- `RUNCOMFY_REFERENCE_COVERAGE_NODE_IDS`
- `RUNCOMFY_DIMENSION_NODE_IDS`
- `RUNCOMFY_STEPS_NODE_IDS`

## 4) Add custom domain in Render

In the Render service:

1. Open `Settings`
2. Open `Custom Domains`
3. Add `api.richardandrys.com`

Render will show the DNS target you need to set.

## 5) Add DNS record in WEDOS

In WEDOS DNS for `richardandrys.com`, create the record Render asks for.

Usually this is:

- Type: `CNAME`
- Name: `api`
- Value: `<render-target-provided-by-render>`

If Render instead asks for `A` records, use exactly the values Render gives you.

## 6) Wait for HTTPS to issue

After DNS propagates:

- Render validates the domain
- TLS certificate is issued automatically

Then verify:

- `https://api.richardandrys.com/v1/config`
- `https://api.richardandrys.com/docs`

## 7) Provision your first API key

From your machine:

```powershell
cd hdri_api_server
python scripts/provision_account.py --base-url https://api.richardandrys.com --admin-token "<HDRI_ADMIN_TOKEN>" --account-id richard-test --initial-tokens 25
```

## 8) Test from Blender

Use in the add-on:

- API Base URL: `https://api.richardandrys.com`
- API Key: the provisioned key

## Notes

- WEDOS hosts the website; Render hosts the API.
- Do not point Blender to `richardandrys.com/hdri-addon`; point it to `api.richardandrys.com`.
- The API host does not need model storage if RunComfy is executing the workflow.
