"""Token packages and optional Stripe checkout for HDRI credits."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from fastapi import HTTPException

DEFAULT_TOKEN_PACKAGES: list[dict[str, Any]] = [
    {"id": "tokens_10", "label": "10 tokens", "tokens": 10, "price_cents": 900, "currency": "usd"},
    {"id": "tokens_50", "label": "50 tokens", "tokens": 50, "price_cents": 3900, "currency": "usd"},
    {"id": "tokens_150", "label": "150 tokens", "tokens": 150, "price_cents": 9900, "currency": "usd"},
]


def register_free_tokens() -> int:
    try:
        return max(0, int(os.environ.get("HDRI_REGISTER_FREE_TOKENS", "10")))
    except ValueError:
        return 10


def token_packages() -> list[dict[str, Any]]:
    raw = os.environ.get("HDRI_TOKEN_PACKAGES_JSON", "").strip()
    if not raw:
        return list(DEFAULT_TOKEN_PACKAGES)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid HDRI_TOKEN_PACKAGES_JSON: {e}") from e
    if not isinstance(parsed, list) or not parsed:
        raise HTTPException(status_code=500, detail="HDRI_TOKEN_PACKAGES_JSON must be a non-empty list.")
    return parsed


def package_by_id(package_id: str) -> dict[str, Any]:
    pid = (package_id or "").strip()
    for pkg in token_packages():
        if str(pkg.get("id", "")).strip() == pid:
            return pkg
    raise HTTPException(status_code=404, detail="Unknown token package.")


def stripe_enabled() -> bool:
    return bool(os.environ.get("STRIPE_SECRET_KEY", "").strip())


def _stripe():
    try:
        import stripe
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail="Stripe SDK not installed. pip install stripe on the API server.",
        ) from e
    secret = os.environ.get("STRIPE_SECRET_KEY", "").strip()
    if not secret:
        raise HTTPException(status_code=503, detail="Stripe billing is not configured.")
    stripe.api_key = secret
    return stripe


def create_checkout_session(
    *,
    account_id: str,
    package_id: str,
    success_url: str,
    cancel_url: str,
) -> dict[str, str]:
    pkg = package_by_id(package_id)
    tokens = int(pkg.get("tokens", 0))
    price_cents = int(pkg.get("price_cents", 0))
    currency = str(pkg.get("currency", "usd")).lower()
    label = str(pkg.get("label", f"{tokens} tokens"))
    if tokens <= 0 or price_cents <= 0:
        raise HTTPException(status_code=500, detail="Token package misconfigured.")

    stripe = _stripe()
    session = stripe.checkout.Session.create(
        mode="payment",
        success_url=success_url,
        cancel_url=cancel_url,
        line_items=[
            {
                "price_data": {
                    "currency": currency,
                    "product_data": {"name": f"HDRI {label}"},
                    "unit_amount": price_cents,
                },
                "quantity": 1,
            }
        ],
        metadata={
            "account_id": account_id,
            "package_id": package_id,
            "tokens": str(tokens),
        },
    )
    checkout_url = str(getattr(session, "url", "") or "")
    session_id = str(getattr(session, "id", "") or "")
    if not checkout_url or not session_id:
        raise HTTPException(status_code=502, detail="Stripe did not return a checkout URL.")
    return {"checkout_url": checkout_url, "session_id": session_id}


def verify_stripe_webhook(payload: bytes, sig_header: str | None) -> dict[str, Any]:
    stripe = _stripe()
    secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()
    if not secret:
        raise HTTPException(status_code=503, detail="Stripe webhook secret is not configured.")
    if not sig_header:
        raise HTTPException(status_code=400, detail="Missing Stripe-Signature header.")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, secret)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Stripe webhook: {e}") from e
    return dict(event)


def checkout_completed_event(event: dict[str, Any]) -> tuple[str, int, str] | None:
    if str(event.get("type", "")) != "checkout.session.completed":
        return None
    obj = event.get("data", {}).get("object", {})
    if not isinstance(obj, dict):
        return None
    meta = obj.get("metadata") or {}
    account_id = str(meta.get("account_id", "")).strip()
    tokens_raw = meta.get("tokens", "0")
    session_id = str(obj.get("id", "")).strip() or str(uuid.uuid4())
    try:
        tokens = int(tokens_raw)
    except ValueError:
        return None
    if not account_id or tokens <= 0:
        return None
    return account_id, tokens, session_id
