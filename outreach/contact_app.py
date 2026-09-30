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
import subprocess
import sys
import threading
import webbrowser
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import send_production_emails as mailer

HERE = Path(__file__).resolve().parent
CONTACTS = HERE / "production_leads.csv"
REACHED = HERE / "reached.csv"
ASSETS = HERE / "contact_app"
REACHED_FIELDS = ["company", "reached_on", "source"]
CREDENTIALS_FILE = HERE / ".smtp.json"
CREDENTIALS_LOCK = threading.Lock()
REACHED_LOCK = threading.Lock()
SESSION_CREDENTIALS: dict[str, str] = {}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_SEND_BYTES = 80_000
ASSET_TYPES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/contacts.js": ("contacts.js", "text/javascript; charset=utf-8"),
}


def load_contacts() -> list[dict[str, str]]:
    """Return the current CSV contents without caching them."""
    with CONTACTS.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_contacts_js() -> Path:
    """Snapshot contacts so the HTML file can open without a running server."""
    payload = json.dumps({"contacts": load_contacts()}, ensure_ascii=False)
    path = ASSETS / "contacts.js"
    path.write_text(f"window.EMBEDDED_CONTACTS = {payload};\n", encoding="utf-8")
    return path


def write_standalone_html() -> Path:
    """Write one self-contained HTML file that opens in a browser with no server."""
    write_contacts_js()
    html = (ASSETS / "index.html").read_text(encoding="utf-8")
    css = (ASSETS / "styles.css").read_text(encoding="utf-8")
    contacts = (ASSETS / "contacts.js").read_text(encoding="utf-8")
    script = (ASSETS / "app.js").read_text(encoding="utf-8")
    html = html.replace(
        '<link rel="stylesheet" href="styles.css">',
        f"<style>\n{css}\n</style>",
    )
    html = html.replace(
        '  <script src="contacts.js"></script>\n  <script src="app.js" defer></script>',
        f"<script>\n{contacts}\n</script>\n<script>\n{script}\n</script>",
    )
    path = HERE / "composer.html"
    path.write_text(html, encoding="utf-8")
    return path


def known_companies() -> set[str]:
    return {row["company"].strip() for row in load_contacts() if row.get("company")}


def load_reached() -> dict[str, dict[str, str]]:
    """Return reached companies from the local tracker.

    If the tracker file does not exist yet, seed it from the send log so
    previously sent emails still count.
    """
    known = known_companies()
    records: dict[str, dict[str, str]] = {}
    if REACHED.exists():
        with REACHED.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                company = row.get("company", "").strip()
                if company in known:
                    records[company] = {
                        "company": company,
                        "reached_on": row.get("reached_on") or date.today().isoformat(),
                        "source": row.get("source") or "manual",
                    }
        return _with_sheet_status(records)
    if mailer.SENT_LOG.exists():
        with mailer.SENT_LOG.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                company = row.get("company", "").strip()
                if company in known:
                    records[company] = {
                        "company": company,
                        "reached_on": row.get("sent_on", date.today().isoformat()),
                        "source": "sent",
                    }
    return _with_sheet_status(records)


def _with_sheet_status(records: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """Treat studios already marked sent or replied on the sheet as reached."""
    for row in load_contacts():
        company = row.get("company", "").strip()
        status = row.get("status", "").strip().lower()
        if company and status in {"sent", "replied", "won"} and company not in records:
            records[company] = {
                "company": company,
                "reached_on": row.get("verified_on") or date.today().isoformat(),
                "source": "sent" if status == "sent" else "manual",
            }
    return records


def write_reached(records: dict[str, dict[str, str]]) -> None:
    with REACHED.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REACHED_FIELDS)
        writer.writeheader()
        for company in sorted(records):
            writer.writerow(records[company])


def reached_payload() -> dict[str, object]:
    records = load_reached()
    return {
        "reached": list(records.values()),
        "count": len(records),
        "total": len(known_companies()),
    }


def set_reached(company: str, reached: bool, source: str = "manual") -> dict[str, object]:
    company = company.strip()
    if company not in known_companies():
        raise ValueError("Unknown company. Reload the contact sheet and try again.")
    if source not in {"manual", "sent"}:
        raise ValueError("Reached source must be manual or sent.")
    with REACHED_LOCK:
        records = load_reached()
        if reached:
            existing = records.get(company)
            records[company] = {
                "company": company,
                "reached_on": existing["reached_on"] if existing else date.today().isoformat(),
                "source": "sent" if (existing and existing.get("source") == "sent") or source == "sent" else "manual",
            }
        else:
            records.pop(company, None)
        write_reached(records)
        return reached_payload()


def load_saved_credentials() -> dict[str, str]:
    if not CREDENTIALS_FILE.exists():
        return {}
    try:
        data = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    user = str(data.get("user", "")).strip()
    password = str(data.get("password", "")).strip()
    if not user or not password:
        return {}
    saved = {"user": user, "password": password}
    if data.get("port"):
        saved["port"] = str(data.get("port"))
    return saved


def current_credentials() -> dict[str, str]:
    with CREDENTIALS_LOCK:
        if SESSION_CREDENTIALS.get("user") and SESSION_CREDENTIALS.get("password"):
            return dict(SESSION_CREDENTIALS)
        saved = load_saved_credentials()
        if saved:
            SESSION_CREDENTIALS.update(saved)
        return dict(saved)


def save_credentials(user: str, password: str, persist: bool, port: int | None = None) -> None:
    with CREDENTIALS_LOCK:
        SESSION_CREDENTIALS["user"] = user
        SESSION_CREDENTIALS["password"] = password
        if port:
            SESSION_CREDENTIALS["port"] = str(port)
        if persist:
            payload = {"user": user, "password": password}
            if port:
                payload["port"] = port
            CREDENTIALS_FILE.write_text(json.dumps(payload), encoding="utf-8")
            CREDENTIALS_FILE.chmod(0o600)
        elif CREDENTIALS_FILE.exists():
            CREDENTIALS_FILE.unlink()


def clear_credentials() -> None:
    with CREDENTIALS_LOCK:
        SESSION_CREDENTIALS.clear()
        if CREDENTIALS_FILE.exists():
            CREDENTIALS_FILE.unlink()


def normalize_app_password(password: str) -> str:
    """Gmail App Passwords are often copied with spaces; SMTP wants 16 characters."""
    return "".join(password.split())


def smtp_settings() -> tuple[str, int, str, str, str]:
    stored = current_credentials()
    user = stored.get("user") or os.environ.get("SMTP_USER", "").strip()
    password = normalize_app_password(
        stored.get("password") or os.environ.get("SMTP_PASSWORD", "")
    )
    sender = os.environ.get("SMTP_FROM", user).strip()
    if not user or not password:
        raise RuntimeError("Log in with your Gmail address and App Password first.")
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    stored_port = stored.get("port")
    port = int(stored_port or os.environ.get("SMTP_PORT", "587"))
    return host, port, user, password, sender or user


def smtp_send(host: str, port: int, user: str, password: str, message: object) -> int:
    """Send one message, falling back to Gmail SSL 465 if STARTTLS 587 drops."""
    context = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, 465, context=context, timeout=30) as server:
            server.ehlo()
            server.login(user, password)
            server.send_message(message)
        return 465
    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(user, password)
            server.send_message(message)
        return port
    except smtplib.SMTPAuthenticationError:
        raise
    except (smtplib.SMTPException, OSError, TimeoutError):
        with smtplib.SMTP_SSL(host, 465, context=context, timeout=30) as server:
            server.ehlo()
            server.login(user, password)
            server.send_message(message)
        return 465


def verify_smtp(user: str, password: str) -> int:
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP(host, port, timeout=20) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(user, password)
            return port
    except smtplib.SMTPAuthenticationError:
        raise
    except (smtplib.SMTPException, OSError, TimeoutError):
        with smtplib.SMTP_SSL(host, 465, context=context, timeout=20) as server:
            server.ehlo()
            server.login(user, password)
        return 465


def login(user: str, password: str, persist: bool) -> dict[str, object]:
    user = user.strip()
    password = normalize_app_password(password)
    if not EMAIL_RE.fullmatch(user):
        raise ValueError("Enter a valid Gmail address.")
    if len(password) < 8:
        raise ValueError("Enter a Gmail App Password. Google shows it as 16 characters.")
    try:
        port = verify_smtp(user, password)
    except smtplib.SMTPAuthenticationError as exc:
        raise ValueError(
            "Gmail rejected the login. Use an App Password from myaccount.google.com/apppasswords, not your normal password."
        ) from exc
    except smtplib.SMTPServerDisconnected as exc:
        raise ValueError(
            "Gmail closed the login. That usually means the App Password is wrong, or Google is blocking this computer. Try a new App Password, or run the composer on your Mac."
        ) from exc
    except (smtplib.SMTPException, OSError, TimeoutError) as exc:
        raise ValueError(
            "Could not finish the Gmail login from this computer. Double-click Open Composer.command on your Mac and log in there — Gmail often blocks cloud servers."
        ) from exc
    save_credentials(user, password, persist, port=port)
    return smtp_ready()


def smtp_ready() -> dict[str, object]:
    try:
        _host, _port, user, _password, sender = smtp_settings()
    except RuntimeError:
        return {"configured": False, "from": ""}
    return {"configured": True, "from": sender or user}


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
    host, port, user, password, sender = smtp_settings()
    message = mailer.build_message(
        {"to": recipient, "subject": subject, "body": body + "\n", "attachment": ""},
        sender,
    )
    smtp_send(host, port, user, password, message)
    mailer.append_sent(recipient, company, subject)
    reached = set_reached(company, True, source="sent")
    return {"to": recipient, "from": sender, "subject": subject, "reached": reached}


class ContactAppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - inherited HTTP method name
        path = urlparse(self.path).path
        if path == "/api/contacts":
            self.send_json({"contacts": load_contacts()})
            return
        if path == "/api/send-status":
            self.send_json(smtp_ready())
            return
        if path == "/api/reached":
            self.send_json(reached_payload())
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
        if path not in {"/api/send", "/api/reached", "/api/login", "/api/logout"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length") or "0")
        if path == "/api/logout":
            clear_credentials()
            self.send_json({"ok": True, **smtp_ready()})
            return
        if length <= 0 or length > MAX_SEND_BYTES:
            self.send_json({"ok": False, "error": "Message is empty or too large."}, HTTPStatus.BAD_REQUEST)
            return
        try:
            payload = json.loads(self.rfile.read(length).decode())
            if path == "/api/login":
                result = login(
                    str(payload.get("user", "")),
                    str(payload.get("password", "")),
                    bool(payload.get("persist", True)),
                )
                self.send_json({"ok": True, **result})
                return
            if path == "/api/reached":
                result = set_reached(
                    str(payload.get("company", "")),
                    bool(payload.get("reached")),
                    str(payload.get("source") or "manual"),
                )
                self.send_json({"ok": True, **result})
                return
            result = send_one_email(payload)
        except (json.JSONDecodeError, ValueError, RuntimeError, smtplib.SMTPException, OSError, KeyError) as exc:
            status = HTTPStatus.BAD_REQUEST
            if isinstance(exc, RuntimeError) and "Log in" in str(exc):
                status = HTTPStatus.UNAUTHORIZED
            elif isinstance(exc, smtplib.SMTPAuthenticationError):
                status = HTTPStatus.UNAUTHORIZED
            self.send_json({"ok": False, "error": str(exc)}, status)
            return
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
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


class ReusableComposerServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/api/health"


def existing_server_running(port: int) -> bool:
    try:
        with urlopen(health_url(port), timeout=1) as response:
            return response.status == 200
    except (URLError, OSError, TimeoutError):
        return False


def open_in_browser(url: str) -> None:
    """Open Safari on a Mac. Other systems use the default browser."""
    if sys.platform == "darwin":
        subprocess.run(["open", "-a", "Safari", url], check=False)
        return
    try:
        webbrowser.open(url)
    except Exception:
        pass


def announce_ready(url: str, already: bool = False, open_browser: bool = True) -> None:
    print()
    print("=" * 56)
    if already:
        print("Composer uz bezi. Tohle neni chyba.")
    else:
        print("Composer bezi na tomhle Macu.")
    print(f"Safari: {url}")
    print("Nech tohle okno otevrene. Zavres ho, server se vypne.")
    print("=" * 56)
    print()
    if open_browser:
        open_in_browser(url)


def bind_server(host: str, preferred_port: int) -> tuple[ReusableComposerServer, int]:
    last_error: OSError | None = None
    for port in range(preferred_port, preferred_port + 20):
        if existing_server_running(port):
            continue
        try:
            return ReusableComposerServer((host, port), ContactAppHandler), port
        except OSError as exc:
            last_error = exc
            continue
    message = f"Could not start the composer on ports {preferred_port}-{preferred_port + 19}: {last_error}"
    print(message)
    raise SystemExit(1) from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser window")
    args = parser.parse_args()
    write_standalone_html()
    open_url = f"http://127.0.0.1:{args.port}"

    if existing_server_running(args.port):
        announce_ready(open_url, already=True, open_browser=not args.no_browser)
        if sys.stdin.isatty():
            try:
                input("Press Enter to close this window. The composer keeps running.\n")
            except EOFError:
                pass
        return

    try:
        server, port = bind_server(args.host, args.port)
    except SystemExit:
        if existing_server_running(args.port):
            announce_ready(open_url, already=True, open_browser=not args.no_browser)
            return
        raise
    open_url = f"http://127.0.0.1:{port}"
    print(f"Reading contacts from: {CONTACTS}")
    print("Press Ctrl+C to stop.")
    announce_ready(open_url, already=False, open_browser=not args.no_browser)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
