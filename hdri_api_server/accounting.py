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
