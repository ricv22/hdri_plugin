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


def token_unit_price_cents() -> int:
    """Price per token for custom-amount purchases."""
    try:
        return max(1, int(os.environ.get("HDRI_TOKEN_UNIT_PRICE_CENTS", "90")))
    except ValueError:
        return 90


def token_currency() -> str:
    return (os.environ.get("HDRI_TOKEN_CURRENCY", "usd").strip().lower() or "usd")


def custom_token_limits() -> tuple[int, int]:
    """(min, max) tokens allowed for a single custom purchase."""
    try:
        lo = max(1, int(os.environ.get("HDRI_CUSTOM_TOKENS_MIN", "1")))
    except ValueError:
        lo = 1
    try:
        hi = max(lo, int(os.environ.get("HDRI_CUSTOM_TOKENS_MAX", "1000")))
    except ValueError:
        hi = max(lo, 1000)
    return lo, hi


def custom_tokens_enabled() -> bool:
    return os.environ.get("HDRI_CUSTOM_TOKENS_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}


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
    package_id: str | None = None,
    tokens: int | None = None,
    success_url: str,
    cancel_url: str,
) -> dict[str, str]:
    currency = token_currency()

    if tokens is not None and (package_id is None or str(package_id).strip().lower() == "custom"):
        if not custom_tokens_enabled():
            raise HTTPException(status_code=400, detail="Custom token amounts are disabled.")
        lo, hi = custom_token_limits()
        token_count = int(tokens)
        if token_count < lo or token_count > hi:
            raise HTTPException(status_code=400, detail=f"Choose between {lo} and {hi} tokens.")
        unit_price = token_unit_price_cents()
        meta_package_id = "custom"
        label = f"{token_count} tokens"
        # quantity = number of tokens so Stripe shows "N x unit price".
        line_quantity = token_count
        line_unit_amount = unit_price
        product_name = "HDRI token"
    else:
        if not package_id:
            raise HTTPException(status_code=400, detail="Provide a package_id or a custom token amount.")
        pkg = package_by_id(package_id)
        token_count = int(pkg.get("tokens", 0))
        price_cents = int(pkg.get("price_cents", 0))
        currency = str(pkg.get("currency", currency)).lower()
        label = str(pkg.get("label", f"{token_count} tokens"))
        if token_count <= 0 or price_cents <= 0:
            raise HTTPException(status_code=500, detail="Token package misconfigured.")
        meta_package_id = str(package_id)
        line_quantity = 1
        line_unit_amount = price_cents
        product_name = f"HDRI {label}"

    stripe = _stripe()

    def _flag(name: str) -> bool:
        return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}

    session_kwargs: dict[str, Any] = {
        "mode": "payment",
        "success_url": success_url,
        "cancel_url": cancel_url,
        "line_items": [
            {
                "price_data": {
                    "currency": currency,
                    "product_data": {"name": product_name},
                    # When Stripe Tax is on, prices are treated as tax-inclusive by
                    # default for EU consumers; set HDRI_STRIPE_PRICE_TAX_BEHAVIOR.
                    "tax_behavior": os.environ.get("HDRI_STRIPE_PRICE_TAX_BEHAVIOR", "inclusive").strip()
                    if _flag("HDRI_STRIPE_AUTOMATIC_TAX")
                    else None,
                    "unit_amount": line_unit_amount,
                },
                "quantity": line_quantity,
            }
        ],
        "metadata": {
            "account_id": account_id,
            "package_id": meta_package_id,
            "tokens": str(token_count),
        },
    }

    # Strip None tax_behavior so we don't send it when tax is disabled.
    price_data = session_kwargs["line_items"][0]["price_data"]
    if price_data.get("tax_behavior") is None:
        price_data.pop("tax_behavior", None)

    # EU VAT on digital goods: let Stripe Tax compute and collect VAT, and
    # collect a billing address (required for correct tax rates / invoices).
    if _flag("HDRI_STRIPE_AUTOMATIC_TAX"):
        session_kwargs["automatic_tax"] = {"enabled": True}
        session_kwargs["billing_address_collection"] = "required"

    # Generate a proper invoice/receipt for each purchase (legal record).
    if _flag("HDRI_STRIPE_CREATE_INVOICE"):
        session_kwargs["invoice_creation"] = {"enabled": True}

    # Optional: capture VAT/business IDs for B2B reverse-charge.
    if _flag("HDRI_STRIPE_COLLECT_TAX_ID"):
        session_kwargs["tax_id_collection"] = {"enabled": True}

    session = stripe.checkout.Session.create(**session_kwargs)
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
