"""Anti-abuse helpers for self-service registration (free-token farming)."""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request

if TYPE_CHECKING:
    from job_store import JobStore

# Common disposable / one-time inbox domains (extend via env).
_DISPOSABLE_DOMAINS = frozenset(
    {
        "mailinator.com",
        "guerrillamail.com",
        "guerrillamailblock.com",
        "10minutemail.com",
        "tempmail.com",
        "temp-mail.org",
        "yopmail.com",
        "throwaway.email",
        "getnada.com",
        "sharklasers.com",
        "trashmail.com",
    }
)


def canonicalize_email(email: str) -> str:
    """
    Normalize email for uniqueness checks.
    Gmail/Googlemail: ignore dots and +tags in the local part.
    """
    norm = (email or "").strip().lower()
    local, sep, domain = norm.partition("@")
    if not sep or not domain:
        return norm
    if domain == "googlemail.com":
        domain = "gmail.com"
    if domain == "gmail.com":
        local = local.split("+", 1)[0].replace(".", "")
    return f"{local}@{domain}"


def is_disposable_email(email: str) -> bool:
    if os.environ.get("HDRI_BLOCK_DISPOSABLE_EMAIL", "1").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    domain = (email or "").strip().lower().split("@")[-1]
    if domain in _DISPOSABLE_DOMAINS:
        return True
    extra = os.environ.get("HDRI_DISPOSABLE_EMAIL_DOMAINS", "").strip()
    if extra:
        blocked = {d.strip().lower() for d in extra.split(",") if d.strip()}
        return domain in blocked
    return False


def client_ip_from_request(request: Request) -> str:
    forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    if forwarded:
        return forwarded
    if request.client and request.client.host:
        return str(request.client.host)
    return ""


def hash_client_ip(ip: str) -> str:
    secret = os.environ.get("HDRI_SIGNING_SECRET", "dev-secret-change-me").encode("utf-8")
    raw = (ip or "unknown").encode("utf-8")
    return hmac.new(secret, raw, hashlib.sha256).hexdigest()


def _register_ip_window_s() -> int:
    try:
        return max(3600, int(os.environ.get("HDRI_REGISTER_IP_WINDOW_S", str(30 * 86400))))
    except ValueError:
        return 30 * 86400


def _register_max_per_ip() -> int:
    try:
        return max(1, int(os.environ.get("HDRI_REGISTER_MAX_ACCOUNTS_PER_IP", "2")))
    except ValueError:
        return 2


def free_tokens_for_new_registration(store: JobStore, ip_hash: str, default_free: int) -> int:
    """
    Return how many free tokens to grant on signup.
    First N accounts per IP per window get ``default_free``; further signups get 0 (can still buy).
    """
    since = int(time.time()) - _register_ip_window_s()
    count = store.count_registrations_by_ip(ip_hash, since_ts=since)
    if count >= _register_max_per_ip():
        return 0
    return max(0, int(default_free))


def assert_can_register_email(email: str, store: JobStore) -> str:
    """Validate email; return canonical form. Raises HTTPException on abuse."""
    norm = (email or "").strip().lower()
    if is_disposable_email(norm):
        raise HTTPException(
            status_code=400,
            detail="Disposable email addresses are not allowed. Use a regular email provider.",
        )
    canonical = canonicalize_email(norm)
    if store.get_account_by_canonical_email(canonical):
        raise HTTPException(
            status_code=409,
            detail="An account already exists for this email address (including provider aliases).",
        )
    return canonical
