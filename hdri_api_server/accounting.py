from __future__ import annotations

from typing import Literal

from fastapi import HTTPException

from job_store import JobStore


def token_cost_for_quality(quality_mode: Literal["fast", "balanced", "high"]) -> int:
    if quality_mode == "high":
        return 2
    return 1


def reserve_tokens_or_raise(store: JobStore, account_id: str, job_id: str, amount: int) -> None:
    if amount <= 0:
        return
    ok = store.adjust_tokens_if_possible(account_id, -amount)
    if not ok:
        raise HTTPException(status_code=402, detail="Insufficient tokens.")
    store.record_usage_event(job_id, account_id, -amount, "reserve")


def refund_tokens(store: JobStore, account_id: str, job_id: str, amount: int) -> None:
    if amount <= 0:
        return
    if store.adjust_tokens_if_possible(account_id, amount):
        store.record_usage_event(job_id, account_id, amount, "refund")


def refund_job_if_needed(store: JobStore, account_id: str | None, job_id: str) -> bool:
    if not account_id:
        return False
    row = store.get_job(job_id)
    if not row:
        return False
    if int(row.get("cost_tokens") or 0) <= 0:
        return False
    if not store.try_mark_job_refunded(job_id):
        return False
    refund_tokens(store, account_id, job_id, int(row["cost_tokens"]))
    return True
