#!/usr/bin/env python3
"""Draft and send production outreach from production_leads.csv.

Nothing is sent unless you pass ``send``. A draft is sent only after its
``Approved`` line is changed to ``yes``. SMTP credentials stay in the
environment and are never written into this repository.

Examples:
  python3 outreach/send_production_emails.py draft --priority 5
  python3 outreach/send_production_emails.py approve --company "Blue Faces"
  python3 outreach/send_production_emails.py send --limit 5
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import smtplib
import ssl
import sys
import time
from datetime import date
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEADS = ROOT / "outreach" / "production_leads.csv"
DRAFTS = ROOT / "outreach" / "drafts"
SENT_LOG = ROOT / "outreach" / "sent_log.csv"
MAX_PER_RUN = 20

CSV_FIELDS = [
    "company",
    "country",
    "city",
    "website",
    "contact_email",
    "contact_page",
    "focus",
    "signal",
    "why_fit",
    "personalization_hook",
    "priority",
    "outreach_language",
    "status",
    "source_url",
    "verified_on",
]


def load_leads() -> list[dict[str, str]]:
    with LEADS.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sent_addresses() -> set[str]:
    if not SENT_LOG.exists():
        return set()
    with SENT_LOG.open(newline="", encoding="utf-8") as handle:
        return {row["to"].casefold() for row in csv.DictReader(handle) if row.get("to")}


def slug(company: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", company.casefold()).strip("-")
    return cleaned or "studio"


def subject_and_body(row: dict[str, str]) -> tuple[str, str]:
    company = row["company"].strip()
    if czech_email(row):
        subject = "VFX spolupráce"
        body = f"""Dobrý den,

Píšu vám, protože tvoříte reklamy a spoty, u kterých se 3D a VFX občas hodí.

Jsem 3D grafik a produkcím pomáhám externě, když potřebují pokrýt konkrétní záběr nebo jen doplnit kapacitu.

Shot umím převzít celý, stejně tak se můžu zapojit jen do části postprodukce, kterou právě řešíte, od rotoscope, camera trackingu a matchmove přes tvorbu a animaci modelů až po compositing, střih a finální render.

Ukázky mé práce:
https://richardandrys.com

Pokud právě něco podobného řešíte, jsem k dispozici.

Děkuji za váš čas,
Richard Andrýs
rich.andrys@gmail.com
"""
        return subject, body

    subject = "VFX collaboration"
    body = f"""Hello {company} team,

I'm writing because you make ads and campaigns where 3D and VFX sometimes come in handy.

I'm a 3D artist, and I help productions on a freelance basis when they need a specific shot covered, or just some extra capacity.

I can take on a full shot, or jump in on just the part of post you're working on, from rotoscope, camera tracking, and matchmove through modeling and animation to compositing, editing, and the final render.

Work samples:
https://richardandrys.com

If you're on something like this right now, I'm available.

Thank you for your time,
Richard Andrýs
rich.andrys@gmail.com
"""
    return subject, body


def czech_email(row: dict[str, str]) -> bool:
    return row.get("outreach_language", "").strip().casefold().startswith("czech")


def draft_path(row: dict[str, str]) -> Path:
    return DRAFTS / f"{slug(row['company'])}.txt"


def render_draft(row: dict[str, str], approved: str = "no") -> str:
    subject, body = subject_and_body(row)
    note = row["personalization_hook"].replace("\n", " ").strip()
    return (
        f"To: {row['contact_email'].strip()}\n"
        f"Subject: {subject}\n"
        f"Approved: {approved}\n"
        "Attachment:\n"
        f"Note: {note}\n"
        "\n"
        "---\n"
        f"{body}"
    )


def parse_draft(path: Path) -> dict[str, str]:
    raw = path.read_text(encoding="utf-8")
    header, separator, body = raw.partition("\n---\n")
    if not separator:
        raise ValueError(f"{path.name} is missing the --- body separator")
    fields: dict[str, str] = {}
    for line in header.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().casefold()] = value.strip()
    fields["body"] = body.strip() + "\n"
    fields["path"] = str(path)
    return fields


def selected(rows: list[dict[str, str]], company: str | None, priority: int) -> list[dict[str, str]]:
    chosen = []
    for row in rows:
        if company and row["company"].casefold() != company.casefold():
            continue
        try:
            rank = int(row["priority"])
        except ValueError:
            continue
        if rank < priority or not row["contact_email"].strip():
            continue
        chosen.append(row)
    return chosen


def write_drafts(rows: list[dict[str, str]]) -> tuple[list[Path], int]:
    DRAFTS.mkdir(exist_ok=True)
    written = []
    kept = 0
    for row in rows:
        path = draft_path(row)
        if path.exists():
            kept += 1
            continue
        path.write_text(render_draft(row), encoding="utf-8")
        written.append(path)
    return written, kept


def approve(rows: list[dict[str, str]]) -> int:
    count = 0
    for row in rows:
        path = draft_path(row)
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            path.write_text(render_draft(row), encoding="utf-8")
        text = path.read_text(encoding="utf-8")
        updated, replacements = re.subn(r"(?m)^Approved:\s*.*$", "Approved: yes", text, count=1)
        if replacements != 1:
            raise ValueError(f"{path.name} has no Approved line")
        path.write_text(updated, encoding="utf-8")
        count += 1
    return count


def append_sent(to: str, company: str, subject: str) -> None:
    new_file = not SENT_LOG.exists()
    with SENT_LOG.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sent_on", "company", "to", "subject"])
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "sent_on": date.today().isoformat(),
                "company": company,
                "to": to,
                "subject": subject,
            }
        )


def build_message(item: dict[str, str], sender: str) -> EmailMessage:
    message = EmailMessage()
    message["From"] = sender
    message["To"] = item["to"]
    message["Subject"] = item["subject"]
    message.set_content(item["body"])
    attachment = item.get("attachment", "")
    if attachment:
        path = ROOT / attachment
        if not path.is_file():
            raise FileNotFoundError(f"Attachment not found: {path}")
        message.add_attachment(
            path.read_bytes(),
            maintype="application",
            subtype="pdf",
            filename=path.name,
        )
    return message


def smtp_settings() -> tuple[str, int, str, str, str]:
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER", "")
    password = os.environ.get("SMTP_PASSWORD", "")
    sender = os.environ.get("SMTP_FROM", user)
    missing = [name for name, value in (("SMTP_USER", user), ("SMTP_PASSWORD", password)) if not value]
    if missing:
        raise RuntimeError(
            "Missing "
            + ", ".join(missing)
            + ". For Gmail, create an App Password and export SMTP_USER and SMTP_PASSWORD."
        )
    return host, port, user, password, sender


def send(rows: list[dict[str, str]], limit: int, delay: float, check_only: bool) -> int:
    if limit < 1 or limit > MAX_PER_RUN:
        raise RuntimeError(f"--limit must be between 1 and {MAX_PER_RUN}")
    host, port, user, password, sender = smtp_settings()
    already_sent = sent_addresses()
    queued: list[tuple[dict[str, str], dict[str, str]]] = []
    for row in rows:
        path = draft_path(row)
        if not path.exists():
            print(f"skip {row['company']}: draft missing")
            continue
        item = parse_draft(path)
        recipient = item.get("to", "")
        if item.get("approved", "").casefold() != "yes":
            print(f"skip {row['company']}: draft is not approved")
            continue
        if recipient.casefold() in already_sent:
            print(f"skip {row['company']}: already sent")
            continue
        if "Note:" in item["body"] or "personalization_hook" in item["body"]:
            raise RuntimeError(f"{path.name} would leak an internal note")
        queued.append((row, item))
        if len(queued) >= limit:
            break

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=30) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(user, password)
        if check_only:
            print(f"SMTP login works for {sender}")
            return 0
        for index, (row, item) in enumerate(queued):
            server.send_message(build_message(item, sender))
            append_sent(item["to"], row["company"], item["subject"])
            print(f"sent {row['company']} -> {item['to']}")
            if index < len(queued) - 1:
                time.sleep(delay)
    return len(queued)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("draft", "approve", "send", "check"))
    parser.add_argument("--company", help="Only this company name")
    parser.add_argument("--priority", type=int, default=5, help="Minimum lead priority (default: 5)")
    parser.add_argument("--limit", type=int, default=5, help="Maximum messages in one send (max 20)")
    parser.add_argument("--delay", type=float, default=45, help="Seconds between sent messages")
    args = parser.parse_args()

    rows = selected(load_leads(), args.company, args.priority)
    if args.company and not rows:
        print(f"No matching lead with an email and priority >= {args.priority}", file=sys.stderr)
        return 1

    if args.command == "draft":
        paths, kept = write_drafts(rows)
        print(f"Wrote {len(paths)} drafts to {DRAFTS}")
        if kept:
            print(f"Kept {kept} existing drafts so manual edits stay intact")
        print("Edit the text below --- if needed, then approve one company before send.")
        return 0
    if args.command == "approve":
        if not args.company:
            print("Refusing to approve every draft at once. Pass --company.", file=sys.stderr)
            return 1
        print(f"Approved {approve(rows)} draft")
        return 0
    if args.command == "check":
        send(rows, limit=1, delay=args.delay, check_only=True)
        return 0
    sent = send(rows, limit=args.limit, delay=args.delay, check_only=False)
    print(f"Sent {sent} messages")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, FileNotFoundError, smtplib.SMTPException) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
