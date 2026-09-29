#!/usr/bin/env python3
"""Serve the local production outreach composer.

The app reads the latest contacts from ``production_leads.csv`` and helps
compose, edit, copy, or send one reviewed email at a time. It does not modify
the spreadsheet and never sends more than one message per request.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import smtplib
import ssl
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import send_production_emails as mailer

HERE = Path(__file__).resolve().parent
CONTACTS = HERE / "production_leads.csv"
ASSETS = HERE / "contact_app"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_SEND_BYTES = 80_000
ASSET_TYPES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


def load_contacts() -> list[dict[str, str]]:
    """Return the current CSV contents without caching them."""
    with CONTACTS.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def smtp_ready() -> dict[str, object]:
    user = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "").strip()
    sender = os.environ.get("SMTP_FROM", user).strip()
    return {
        "configured": bool(user and password),
        "from": sender if user and password else "",
    }


def send_one_email(payload: dict[str, object]) -> dict[str, str]:
    """Send exactly one reviewed email. Never queues additional recipients."""
    if payload.get("confirm") is not True:
        raise ValueError("Sending requires an explicit confirmation.")
    company = str(payload.get("company", "")).strip()
    recipient = str(payload.get("to", "")).strip()
    subject = str(payload.get("subject", "")).strip()
    body = str(payload.get("body", "")).strip()
    if not company or not subject or not body:
        raise ValueError("Company, subject, and message are required.")
    if not EMAIL_RE.fullmatch(recipient):
        raise ValueError("Enter one valid recipient email.")
    if any(marker in recipient for marker in (",", ";", " ")):
        raise ValueError("Send only one recipient at a time.")
    contacts = {row["company"]: row for row in load_contacts()}
    if company not in contacts:
        raise ValueError("Unknown company. Reload the contact sheet and try again.")
    if recipient.casefold() in mailer.sent_addresses():
        raise ValueError(f"Already sent to {recipient}.")
    host, port, user, password, sender = mailer.smtp_settings()
    message = mailer.build_message(
        {"to": recipient, "subject": subject, "body": body + "\n", "attachment": ""},
        sender,
    )
    context = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=30) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(user, password)
        server.send_message(message)
    mailer.append_sent(recipient, company, subject)
    return {"to": recipient, "from": sender, "subject": subject}


class ContactAppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - inherited HTTP method name
        path = urlparse(self.path).path
        if path == "/api/contacts":
            self.send_json({"contacts": load_contacts()})
            return
        if path == "/api/send-status":
            self.send_json(smtp_ready())
            return
        if path == "/api/health":
            self.send_json({"ok": True, "contacts": len(load_contacts())})
            return
        if path in ASSET_TYPES:
            filename, content_type = ASSET_TYPES[path]
            self.send_file(ASSETS / filename, content_type)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - inherited HTTP method name
        path = urlparse(self.path).path
        if path != "/api/send":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0 or length > MAX_SEND_BYTES:
            self.send_json({"ok": False, "error": "Message is empty or too large."}, HTTPStatus.BAD_REQUEST)
            return
        try:
            payload = json.loads(self.rfile.read(length).decode())
            result = send_one_email(payload)
        except (json.JSONDecodeError, ValueError, RuntimeError, smtplib.SMTPException, OSError) as exc:
            status = HTTPStatus.BAD_REQUEST
            if isinstance(exc, RuntimeError) and "SMTP_" in str(exc):
                status = HTTPStatus.SERVICE_UNAVAILABLE
            self.send_json({"ok": False, "error": str(exc)}, status)
            return
        self.send_json({"ok": True, **result})

    def send_json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def send_file(self, path: Path, content_type: str) -> None:
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        # Keep the terminal useful; malformed requests and exceptions still
        # surface through BaseHTTPRequestHandler.
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ContactAppHandler)
    print(f"Outreach composer: http://{args.host}:{args.port}")
    print(f"Reading contacts from: {CONTACTS}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
