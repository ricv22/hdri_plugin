# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This repository is a **Blender HDRI generation tool** with two components:

| Component | Location | Description |
|---|---|---|
| **HDRI API Server** | `hdri_api_server/` | FastAPI backend (Python 3.12). Accepts images, generates equirectangular panoramas, applies HDR lift, returns signed `.hdr` download URLs. |
| **Blender Addon** | `hdri_from_image_addon.py` | Thin Blender 3.6+ addon (UI panel). Requires a running Blender instance — not testable in headless cloud VMs. |

### Running the API server

```bash
cd hdri_api_server
source .venv/bin/activate
export PANORAMA_MODE=resize  # default; no external API needed
uvicorn app:app --host 127.0.0.1 --port 8000
```

Verify: `curl http://127.0.0.1:8000/v1/config`

Swagger UI: `http://127.0.0.1:8000/docs`

### Running the stub panorama worker (optional)

For testing the `http_json` pipeline without a real AI model:

```bash
cd hdri_api_server
source .venv/bin/activate
uvicorn examples.img2pano_worker_stub:app --host 127.0.0.1 --port 8001
```

Then set `PANORAMA_MODE=http_json` and `PANORAMA_HTTP_URL=http://127.0.0.1:8001/v1/panorama` on the main server.

### Linting

```bash
source hdri_api_server/.venv/bin/activate
ruff check hdri_api_server/
```

Pre-existing lint warnings exist (E402 for imports after `_load_local_env()` in `app.py`, and an unused `PIL.Image` import). These are intentional — `_load_local_env()` must run before other imports to load `.env`.

### Testing

No automated test suite exists yet. Manual testing via Swagger UI or `curl`/`httpx` against the running server is the current approach. See `hdri_api_server/README.md` for API details.

### Key caveats

- The venv lives at `hdri_api_server/.venv/`. Always activate it before running commands.
- `app.py` imports `panorama` and `rgbe_hdr` as bare module names — uvicorn must be started from inside the `hdri_api_server/` directory (or that directory must be on `PYTHONPATH`).
- The `.env` file in `hdri_api_server/` is loaded by `app.py` at import time; environment variables set before startup take precedence.
- The Blender addon (`hdri_from_image_addon.py`) imports `bpy` and cannot be tested outside Blender.
