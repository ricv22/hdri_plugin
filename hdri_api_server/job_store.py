from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any


class JobStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS accounts (
                    account_id TEXT PRIMARY KEY,
                    tokens_remaining INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    api_key_hash TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    account_id TEXT,
                    provider_job_id TEXT,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    cost_tokens INTEGER NOT NULL DEFAULT 0,
                    refunded INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    delta_tokens INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            conn.commit()

    def ensure_account(self, account_id: str, *, initial_tokens: int = 0) -> None:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO accounts(account_id, tokens_remaining, created_at)
                VALUES (?, ?, ?)
                """,
                (account_id, int(initial_tokens), now),
            )
            conn.commit()

    def ensure_api_key(self, api_key_hash: str, account_id: str) -> None:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO api_keys(api_key_hash, account_id, is_active, created_at)
                VALUES (?, ?, 1, ?)
                """,
                (api_key_hash, account_id, now),
            )
            conn.commit()

    def get_account_by_api_key_hash(self, api_key_hash: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT a.account_id, a.tokens_remaining
                FROM api_keys k
                JOIN accounts a ON a.account_id = k.account_id
                WHERE k.api_key_hash = ? AND k.is_active = 1
                """,
                (api_key_hash,),
            ).fetchone()
        if not row:
            return None
        return {
            "account_id": str(row["account_id"]),
            "tokens_remaining": int(row["tokens_remaining"]),
        }

    def get_account(self, account_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT account_id, tokens_remaining FROM accounts WHERE account_id = ?",
                (account_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "account_id": str(row["account_id"]),
            "tokens_remaining": int(row["tokens_remaining"]),
        }

    def adjust_tokens_if_possible(self, account_id: str, delta_tokens: int) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE accounts
                SET tokens_remaining = tokens_remaining + ?
                WHERE account_id = ? AND (tokens_remaining + ?) >= 0
                """,
                (int(delta_tokens), account_id, int(delta_tokens)),
            )
            conn.commit()
            return cur.rowcount > 0

    def record_usage_event(self, job_id: str, account_id: str, delta_tokens: int, event_type: str) -> None:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO usage_events(job_id, account_id, delta_tokens, event_type, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, account_id, int(delta_tokens), event_type, now),
            )
            conn.commit()

    def create_job(self, job_id: str, request_payload: dict[str, Any], *, account_id: str | None, cost_tokens: int) -> None:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs(
                    job_id, account_id, provider_job_id, status, request_json, result_json, error,
                    cost_tokens, refunded, created_at, updated_at
                )
                VALUES (?, ?, NULL, 'queued', ?, NULL, NULL, ?, 0, ?, ?)
                """,
                (job_id, account_id, json.dumps(request_payload), int(cost_tokens), now, now),
            )
            conn.commit()

    def set_job_running(self, job_id: str, provider_job_id: str | None = None) -> None:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    provider_job_id = COALESCE(?, provider_job_id),
                    updated_at = ?
                WHERE job_id = ?
                """,
                (provider_job_id, now, job_id),
            )
            conn.commit()

    def set_job_succeeded(self, job_id: str, result_payload: dict[str, Any]) -> None:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'succeeded', result_json = ?, error = NULL, updated_at = ?
                WHERE job_id = ?
                """,
                (json.dumps(result_payload), now, job_id),
            )
            conn.commit()

    def set_job_failed(self, job_id: str, error_message: str) -> None:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'failed', error = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (error_message, now, job_id),
            )
            conn.commit()

    def mark_job_refunded(self, job_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE jobs SET refunded = 1 WHERE job_id = ?", (job_id,))
            conn.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not row:
            return None
        result_payload: dict[str, Any] | None = None
        request_payload: dict[str, Any] = {}
        if row["result_json"]:
            try:
                result_payload = json.loads(str(row["result_json"]))
            except Exception:
                result_payload = None
        if row["request_json"]:
            try:
                request_payload = json.loads(str(row["request_json"]))
            except Exception:
                request_payload = {}
        return {
            "job_id": str(row["job_id"]),
            "account_id": row["account_id"],
            "provider_job_id": row["provider_job_id"],
            "status": str(row["status"]),
            "request": request_payload,
            "result": result_payload,
            "error": row["error"],
            "cost_tokens": int(row["cost_tokens"] or 0),
            "refunded": bool(row["refunded"]),
            "created_at": int(row["created_at"]),
            "updated_at": int(row["updated_at"]),
        }
