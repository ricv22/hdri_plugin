#!/usr/bin/env python3
"""Serve the local production outreach composer.

The app reads the latest contacts from ``production_leads.csv`` and helps
compose, edit, copy, or send one reviewed email at a time. It does not modify
the spreadsheet and never sends more than one message per request.
"""

from __future__ import annotations

import argparse
import csv
import email
import imaplib
import json
import os
import re
import smtplib
import ssl
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import date, datetime, timezone
from email.header import decode_header
from email.message import Message
from email.utils import parsedate_to_datetime
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
RESPONSES = HERE / "responses.json"
CREDENTIALS_LOCK = threading.Lock()
REACHED_LOCK = threading.Lock()
CONTACTS_LOCK = threading.Lock()
RESPONSE_LOCK = threading.Lock()
SESSION_CREDENTIALS: dict[str, str] = {}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
COPY_ID = "vfx-spoluprace-1"
MAX_SEND_BYTES = 80_000
MAX_RESPONSE_CHARS = 12_000
CONTACT_STATUSES = {"new", "sent", "replied", "won", "lost"}
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
        '<link rel="stylesheet" href="styles.css?v=vfx-spoluprace-1">',
        f"<style>\n{css}\n</style>",
    )
    html = html.replace(
        '  <script src="contacts.js?v=vfx-spoluprace-1"></script>\n  <script src="app.js?v=vfx-spoluprace-1" defer></script>',
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


def read_contact_table() -> tuple[list[str], list[dict[str, str]]]:
    with CONTACTS.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        return fields, list(reader)


def write_contact_table(fields: list[str], rows: list[dict[str, str]]) -> None:
    with CONTACTS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_responses() -> dict[str, dict[str, str]]:
    if not RESPONSES.exists():
        return {}
    try:
        data = json.loads(RESPONSES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): value for key, value in data.items() if isinstance(value, dict)}


def write_responses(records: dict[str, dict[str, str]]) -> None:
    RESPONSES.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    RESPONSES.chmod(0o600)


def contacts_for_client() -> list[dict[str, str]]:
    """Return sheet rows plus any saved reply, which stays out of git."""
    saved = load_responses()
    contacts: list[dict[str, str]] = []
    for row in load_contacts():
        item = dict(row)
        reply = saved.get(row.get("company", "").strip(), {})
        item["response_from"] = str(reply.get("from", ""))
        item["response_subject"] = str(reply.get("subject", ""))
        item["response_date"] = str(reply.get("date", ""))
        item["response_body"] = str(reply.get("body", ""))
        item["response_source"] = str(reply.get("source", ""))
        contacts.append(item)
    return contacts


def contact_mutation_payload() -> dict[str, object]:
    return {"contacts": contacts_for_client(), **reached_payload()}


def set_contact_status(company: str, status: str) -> None:
    company = company.strip()
    if status not in CONTACT_STATUSES:
        raise ValueError("Status must be new, sent, replied, won, or lost.")
    with CONTACTS_LOCK:
        fields, rows = read_contact_table()
        found = False
        for row in rows:
            if row.get("company", "").strip() == company:
                row["status"] = status
                found = True
                break
        if not found:
            raise ValueError("Unknown company. Reload the contact sheet and try again.")
        write_contact_table(fields, rows)


def delete_contact(company: str) -> dict[str, object]:
    """Remove one studio from the sheet, the reached list, and any saved reply."""
    company = company.strip()
    with CONTACTS_LOCK:
        fields, rows = read_contact_table()
        kept = [row for row in rows if row.get("company", "").strip() != company]
        if len(kept) == len(rows):
            raise ValueError("Unknown company. Reload the contact sheet and try again.")
        write_contact_table(fields, kept)
    with REACHED_LOCK:
        if REACHED.exists():
            records = {}
            with REACHED.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    name = row.get("company", "").strip()
                    if name and name != company:
                        records[name] = {
                            "company": name,
                            "reached_on": row.get("reached_on") or date.today().isoformat(),
                            "source": row.get("source") or "manual",
                        }
            write_reached(records)
    with RESPONSE_LOCK:
        saved = load_responses()
        saved.pop(company, None)
        write_responses(saved)
    return contact_mutation_payload()


def store_response(
    company: str,
    body: str,
    *,
    subject: str = "",
    sender: str = "",
    when: str = "",
    source: str = "manual",
) -> dict[str, object]:
    company = company.strip()
    body = body.strip()
    if company not in known_companies():
        raise ValueError("Unknown company. Reload the contact sheet and try again.")
    if not body:
        raise ValueError("The reply is empty.")
    if source not in {"manual", "gmail"}:
        raise ValueError("Reply source must be manual or gmail.")
    record = {
        "from": sender.strip(),
        "subject": subject.strip(),
        "date": when.strip() or date.today().isoformat(),
        "body": body[:MAX_RESPONSE_CHARS],
        "source": source,
        "saved_on": date.today().isoformat(),
    }
    with RESPONSE_LOCK:
        saved = load_responses()
        saved[company] = record
        write_responses(saved)
    set_contact_status(company, "replied")
    set_reached(company, True, source="manual")
    return contact_mutation_payload()


def clear_response(company: str) -> dict[str, object]:
    company = company.strip()
    if company not in known_companies():
        raise ValueError("Unknown company. Reload the contact sheet and try again.")
    with RESPONSE_LOCK:
        saved = load_responses()
        saved.pop(company, None)
        write_responses(saved)
    set_contact_status(company, "sent")
    return contact_mutation_payload()


def decode_mime_header(value: str | None) -> str:
    if not value:
        return ""
    chunks: list[str] = []
    for text, charset in decode_header(value):
        if isinstance(text, bytes):
            chunks.append(text.decode(charset or "utf-8", errors="replace"))
        else:
            chunks.append(text)
    return "".join(chunks).strip()


def message_text(message: Message) -> str:
    preferred: str = ""
    html_fallback: str = ""
    parts = message.walk() if message.is_multipart() else [message]
    for part in parts:
        if part.get_content_maintype() == "multipart":
            continue
        disposition = (part.get("Content-Disposition") or "").lower()
        if "attachment" in disposition:
            continue
        payload = part.get_payload(decode=True)
        if not isinstance(payload, bytes):
            continue
        charset = part.get_content_charset() or "utf-8"
        text = payload.decode(charset, errors="replace").strip()
        if part.get_content_type() == "text/plain" and text:
            preferred = text
            break
        if part.get_content_type() == "text/html" and text and not html_fallback:
            html_fallback = text
    if preferred:
        return preferred
    if not html_fallback:
        return ""
    no_tags = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html_fallback)
    no_tags = re.sub(r"(?s)<[^>]+>", " ", no_tags)
    return re.sub(r"\s+", " ", no_tags).strip()


def list_mailbox_token(name: str) -> str:
    """Mailbox argument for imaplib, which does not quote names itself."""
    if name == "INBOX" or (name.startswith('"') and name.endswith('"')):
        return name
    escaped = name.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def mailboxes_from_list(lines: list[object]) -> list[str]:
    """Prefer Gmail All Mail, then INBOX. Names stay in the server's own spelling."""
    all_mail = ""
    for item in lines:
        if not isinstance(item, bytes):
            continue
        line = item.decode("utf-8", "replace")
        match = re.match(r'^\((?P<flags>[^)]*)\)\s+"(?P<delim>[^"]*)"\s+(?P<name>.+)$', line)
        if not match:
            continue
        if "\\All" in match.group("flags").split():
            all_mail = match.group("name").strip()
    ordered: list[str] = []
    if all_mail:
        ordered.append(list_mailbox_token(all_mail))
    ordered.append("INBOX")
    return ordered


def imap_detail(exc: BaseException) -> str:
    text = str(exc).replace("\n", " ").strip()
    return text[:240] or exc.__class__.__name__


def select_gmail_mailbox(imap: imaplib.IMAP4_SSL) -> str:
    """Open a mailbox without aborting when one name is rejected."""
    candidates = ["INBOX"]
    try:
        status, data = imap.list()
    except imaplib.IMAP4.error:
        status, data = "NO", []
    if status == "OK" and data:
        candidates = mailboxes_from_list(data)
    errors: list[str] = []
    for mailbox in candidates:
        try:
            status, _data = imap.select(mailbox, readonly=True)
        except imaplib.IMAP4.error as exc:
            errors.append(imap_detail(exc))
            continue
        if status == "OK":
            return mailbox
        errors.append(f"{mailbox} {status}")
    detail = "; ".join(errors) or "no mailbox could be selected"
    raise ValueError(f"Could not open a Gmail mailbox ({detail}).")


GENERIC_MAIL_DOMAINS = {
    "gmail.com",
    "googlemail.com",
    "seznam.cz",
    "email.cz",
    "post.cz",
    "centrum.cz",
    "outlook.com",
    "hotmail.com",
    "live.com",
    "yahoo.com",
    "icloud.com",
    "me.com",
    "proton.me",
    "protonmail.com",
    "aol.com",
}


def sender_address(value: str) -> str:
    _name, address = email.utils.parseaddr(value or "")
    return address.casefold()


def message_timestamp(value: str) -> datetime:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


OUTREACH_SUBJECT_MARKERS = (
    "vfx collaboration",
    "vfx spolupráce",
    "vfx spoluprace",
    "freelance 3d",
    "vfx support",
    "3d grafik",
    "externí spolupráce",
    "externi spoluprace",
)


def subject_matches_outreach(value: str) -> bool:
    folded = (value or "").casefold()
    return any(marker in folded for marker in OUTREACH_SUBJECT_MARKERS)


def choose_reply(messages: list[dict[str, str]], own_addresses: set[str]) -> dict[str, str] | None:
    """Pick the newest message in the conversation that we did not send."""
    replies = [item for item in messages if sender_address(item.get("from", "")) not in own_addresses]
    replies = [item for item in replies if sender_address(item.get("from", ""))]
    if not replies:
        return None
    return max(replies, key=lambda item: message_timestamp(item.get("date", "")))


def reply_candidates(
    messages: list[dict[str, str]],
    own_addresses: set[str],
    saved_address: str,
    anchor_threads: set[str],
) -> list[dict[str, str]]:
    """Keep mail for this contact that was sent by someone else.

    A message counts when it is from the saved address, from any address in a
    Gmail conversation that already includes that address, or from another
    address at the same company domain when the subject is the outreach email.
    """
    saved = saved_address.casefold()
    domain = company_domain(saved)
    kept: list[dict[str, str]] = []
    for item in messages:
        sender = sender_address(item.get("from", ""))
        if not sender or sender in own_addresses:
            continue
        thread = item.get("thread") or ""
        if sender == saved or (thread and thread in anchor_threads):
            kept.append(item)
            continue
        if domain and company_domain(sender) == domain and subject_matches_outreach(item.get("subject", "")):
            kept.append(item)
    return kept


def company_domain(address: str) -> str:
    domain = address.rsplit("@", 1)[-1].casefold()
    if domain in GENERIC_MAIL_DOMAINS:
        return ""
    return domain


def imap_search(imap: imaplib.IMAP4_SSL, *criteria: str) -> list[bytes]:
    try:
        status, data = imap.search(None, *criteria)
    except imaplib.IMAP4.error:
        return []
    if status != "OK" or not data or not data[0]:
        return []
    return data[0].split()


def fetch_message_index(imap: imaplib.IMAP4_SSL, ids: list[bytes]) -> list[dict[str, str]]:
    """Read sender, date, and Gmail thread id without downloading bodies."""
    found: list[dict[str, str]] = []
    for start in range(0, len(ids), 20):
        batch = b",".join(ids[start : start + 20]).decode()
        try:
            status, fetched = imap.fetch(batch, "(X-GM-THRID BODY.PEEK[HEADER.FIELDS (FROM DATE SUBJECT)])")
        except imaplib.IMAP4.error:
            continue
        if status != "OK" or not fetched:
            continue
        for item in fetched:
            if not isinstance(item, tuple) or len(item) < 2 or not isinstance(item[1], bytes):
                continue
            prefix = item[0].decode("utf-8", "replace") if isinstance(item[0], bytes) else str(item[0])
            header = email.message_from_bytes(item[1])
            sequence = prefix.split(" ", 1)[0]
            thread = ""
            match = re.search(r"X-GM-THRID (\d+)", prefix)
            if match:
                thread = match.group(1)
            found.append(
                {
                    "seq": sequence,
                    "thread": thread,
                    "from": decode_mime_header(header.get("From")),
                    "date": decode_mime_header(header.get("Date")),
                    "subject": decode_mime_header(header.get("Subject")),
                }
            )
    return found


def fetch_gmail_reply(company: str) -> dict[str, object]:
    """Save the newest reply onto the contact.

    A reply counts when it comes from the saved address, from any address in
    that Gmail conversation, or from another address at the same company domain
    when the subject is the outreach email.
    """
    company = company.strip()
    contacts = {row["company"]: row for row in load_contacts()}
    row = contacts.get(company)
    if row is None:
        raise ValueError("Unknown company. Reload the contact sheet and try again.")
    address = row.get("contact_email", "").strip()
    if not EMAIL_RE.fullmatch(address):
        raise ValueError("This contact has no email address to look up in Gmail.")
    _host, _port, user, password, sender = smtp_settings()
    own = {sender_address(user), sender_address(sender)}
    own.discard("")
    try:
        imap = imaplib.IMAP4_SSL("imap.gmail.com", 993, timeout=30)
    except (imaplib.IMAP4.error, OSError, TimeoutError) as exc:
        raise ValueError("Could not reach Gmail. Run the composer on your Mac and log in there.") from exc
    try:
        try:
            imap.login(user, password)
        except imaplib.IMAP4.error as exc:
            raise ValueError(
                "Gmail rejected the IMAP login ("
                + imap_detail(exc)
                + "). Sending can work while mailbox access is still off: in Gmail open Settings, See all settings, Forwarding and POP/IMAP, and enable IMAP."
            ) from exc
        select_gmail_mailbox(imap)
        anchor_ids: set[bytes] = set(imap_search(imap, "FROM", f'"{address}"'))
        anchor_ids.update(imap_search(imap, "TO", f'"{address}"'))
        domain = company_domain(address)
        domain_ids: set[bytes] = set()
        if domain:
            for term in ("Freelance", "VFX", "grafik"):
                domain_ids.update(imap_search(imap, "FROM", f'"{domain}"', "SUBJECT", f'"{term}"'))
        if not anchor_ids and not domain_ids:
            raise ValueError(
                f"No email to or from {address}. A reply from a different address is attached when it is in the same conversation, or from @{domain or 'the company domain'} with your outreach subject."
            )
        anchor_index = fetch_message_index(imap, sorted(anchor_ids, key=lambda item: int(item))[-40:])
        threads = {item["thread"] for item in anchor_index if item.get("thread")}
        thread_ids: set[bytes] = set()
        for thread in list(threads)[:12]:
            thread_ids.update(imap_search(imap, "X-GM-THRID", thread))
        conversation_ids = anchor_ids | thread_ids
        extra_ids = sorted(domain_ids - conversation_ids, key=lambda item: int(item))[-20:]
        selected = sorted(conversation_ids, key=lambda item: int(item))[-50:] + extra_ids
        indexed = fetch_message_index(imap, selected)
        chosen = choose_reply(reply_candidates(indexed, own, address, threads), own)
        if chosen is None and anchor_ids:
            raise ValueError(
                f"Gmail has your mail with {address}, but no reply yet. A later reply from a different address in that conversation will be attached."
            )
        if chosen is None:
            raise ValueError(
                f"No reply from {address} or another @{domain or 'company'} address about this outreach."
            )
        status, fetched = imap.fetch(chosen["seq"], "(BODY.PEEK[])")
        if status != "OK" or not fetched:
            raise ValueError("Gmail found a reply, but the message could not be read.")
        raw = next((item[1] for item in fetched if isinstance(item, tuple) and isinstance(item[1], bytes)), b"")
        if not raw:
            raise ValueError("Gmail found a reply, but the message was empty.")
        message = email.message_from_bytes(raw)
        body = message_text(message)
        if not body:
            raise ValueError("The reply has no text to attach.")
        return store_response(
            company,
            body,
            subject=decode_mime_header(message.get("Subject")),
            sender=decode_mime_header(message.get("From")),
            when=decode_mime_header(message.get("Date")),
            source="gmail",
        )
    finally:
        try:
            imap.logout()
        except (imaplib.IMAP4.error, OSError):
            pass


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
            self.send_json({"contacts": contacts_for_client()})
            return
        if path == "/api/send-status":
            self.send_json(smtp_ready())
            return
        if path == "/api/reached":
            self.send_json(reached_payload())
            return
        if path == "/api/health":
            self.send_json({"ok": True, "contacts": len(load_contacts()), "copy": COPY_ID})
            return
        if path in ASSET_TYPES:
            filename, content_type = ASSET_TYPES[path]
            self.send_file(ASSETS / filename, content_type)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - inherited HTTP method name
        path = urlparse(self.path).path
        if path not in {
            "/api/send",
            "/api/reached",
            "/api/login",
            "/api/logout",
            "/api/contact/delete",
            "/api/contact/replied",
            "/api/contact/response",
        }:
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
            if path == "/api/contact/delete":
                result = delete_contact(str(payload.get("company", "")))
                self.send_json({"ok": True, **result})
                return
            if path == "/api/contact/replied":
                company = str(payload.get("company", "")).strip()
                replied = payload.get("replied") is not False
                if company not in known_companies():
                    raise ValueError("Unknown company. Reload the contact sheet and try again.")
                set_contact_status(company, "replied" if replied else "sent")
                if replied:
                    set_reached(company, True, source="manual")
                result = contact_mutation_payload()
                self.send_json({"ok": True, **result})
                return
            if path == "/api/contact/response":
                company = str(payload.get("company", ""))
                if payload.get("fetch") is True:
                    result = fetch_gmail_reply(company)
                elif payload.get("clear") is True:
                    result = clear_response(company)
                else:
                    result = store_response(
                        company,
                        str(payload.get("body", "")),
                        subject=str(payload.get("subject", "")),
                        source="manual",
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
        self.send_header("Cache-Control", "no-store")
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


def read_health(port: int) -> dict[str, object]:
    try:
        with urlopen(health_url(port), timeout=1) as response:
            if response.status != 200:
                return {}
            data = json.loads(response.read().decode("utf-8"))
    except (URLError, OSError, TimeoutError, json.JSONDecodeError, UnicodeError):
        return {}
    return data if isinstance(data, dict) else {}


def existing_server_running(port: int) -> bool:
    return bool(read_health(port).get("ok"))


def stop_listener(port: int) -> None:
    """Stop an older composer that is still serving the previous email copy."""
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-tiTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            check=False,
        )
        pids = [int(item) for item in result.stdout.split() if item.isdigit()]
    except (FileNotFoundError, ValueError, OSError):
        pids = []
    for pid in pids:
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, 15)
        except OSError:
            continue
    for _ in range(20):
        if not existing_server_running(port):
            return
        time.sleep(0.1)


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
    print("Predmet: VFX spolupráce  /  VFX collaboration")
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
    open_url = f"http://127.0.0.1:{args.port}/?v={COPY_ID}"
    health = read_health(args.port)
    if health.get("ok") and health.get("copy") == COPY_ID:
        announce_ready(open_url, already=True, open_browser=not args.no_browser)
        if sys.stdin.isatty():
            try:
                input("Press Enter to close this window. The composer keeps running.\n")
            except EOFError:
                pass
        return
    if health.get("ok"):
        print("Zastavuji starsi composer, aby se nacetl novy predmet a hlavicka.")
        stop_listener(args.port)
        if read_health(args.port).get("ok"):
            print("Stary composer stale bezi. V tom druhem okne Terminalu stiskni Ctrl+C a spust tento prikaz znovu.")
            raise SystemExit(1)

    try:
        server, port = bind_server(args.host, args.port)
    except SystemExit:
        if existing_server_running(args.port):
            announce_ready(open_url, already=True, open_browser=not args.no_browser)
            return
        raise
    open_url = f"http://127.0.0.1:{port}/?v={COPY_ID}"
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
