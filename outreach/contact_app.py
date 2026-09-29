#!/usr/bin/env python3
"""Serve the local production outreach composer.

The app is deliberately read-only: it reads the latest contacts from
``production_leads.csv`` and helps compose/copy a message, but it cannot send
email or change the spreadsheet.
"""

from __future__ import annotations

import argparse
import csv
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
CONTACTS = HERE / "production_leads.csv"
ASSETS = HERE / "contact_app"
ASSET_TYPES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


def load_contacts() -> list[dict[str, str]]:
    """Return the current CSV contents without caching them."""
    with CONTACTS.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class ContactAppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - inherited HTTP method name
        path = urlparse(self.path).path
        if path == "/api/contacts":
            self.send_json({"contacts": load_contacts()})
            return
        if path == "/api/health":
            self.send_json({"ok": True, "contacts": len(load_contacts())})
            return
        if path in ASSET_TYPES:
            filename, content_type = ASSET_TYPES[path]
            self.send_file(ASSETS / filename, content_type)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def send_json(self, value: object) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(HTTPStatus.OK)
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
