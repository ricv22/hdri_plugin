from __future__ import annotations

import hashlib
import os
import secrets

from fastapi import Header, HTTPException

from job_store import JobStore


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def require_api_key_enabled() -> bool:
    return os.environ.get("HDRI_REQUIRE_API_KEY", "0").strip().lower() in {"1", "true", "yes", "on"}


def bootstrap_dev_credentials(store: JobStore) -> None:
    """
    Optional bootstrapping for local/dev:
    - HDRI_BOOTSTRAP_API_KEY=...
    - HDRI_BOOTSTRAP_ACCOUNT_ID=dev-account
    - HDRI_BOOTSTRAP_TOKENS=25
    """
    raw_key = os.environ.get("HDRI_BOOTSTRAP_API_KEY", "").strip()
    if not raw_key:
        return
    account_id = os.environ.get("HDRI_BOOTSTRAP_ACCOUNT_ID", "dev-account").strip() or "dev-account"
    tokens = int(os.environ.get("HDRI_BOOTSTRAP_TOKENS", "25"))
    store.ensure_account(account_id, initial_tokens=tokens)
    store.ensure_api_key(hash_api_key(raw_key), account_id)


def authenticate_account(
    store: JobStore,
    authorization: str | None,
    *,
    required: bool,
) -> dict:
    if not authorization:
        if required:
            raise HTTPException(status_code=401, detail="Missing Authorization header.")
        return {"account_id": "anonymous", "tokens_remaining": 0, "is_anonymous": True}

    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authorization must be Bearer token.")
    raw = authorization.split(" ", 1)[1].strip()
    if not raw:
        raise HTTPException(status_code=401, detail="Missing Bearer token.")
    row = store.get_account_by_api_key_hash(hash_api_key(raw))
    if not row:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return {"account_id": row["account_id"], "tokens_remaining": row["tokens_remaining"], "is_anonymous": False}


def auth_header_value(authorization: str | None = Header(default=None)) -> str | None:
    return authorization


def generate_api_key() -> str:
    return f"hdri_{secrets.token_urlsafe(24)}"
