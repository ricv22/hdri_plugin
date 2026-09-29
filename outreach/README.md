# Outreach system — Richard Andrys

Goal: **100 000 CZK this month** through 2–4 smaller projects (20k–50k CZK each), not one prestige client.

## Buckets

| Bucket | Who | Why |
|--------|-----|-----|
| **A — Easy wins** | Czech product/e-commerce brands (supplements, coffee, cosmetics, pet, outdoor, furniture, design) | Every launch needs hero visuals, reels, social ads. Often no in-house CGI. |
| **B — Agencies** | Small/medium agencies & studios (?5–30 people) | Need freelance overflow for 3D, motion, VFX, product shots. |
| **C — Industrial** | Manufacturers with technical products | Weak visuals vs. product quality; CAD/drawings available; expo & sales use cases. |

## Spreadsheet columns

| Column | Purpose |
|--------|---------|
| `company` | Legal or brand name |
| `bucket` | A / B / C |
| `website` | Main site |
| `city` | HQ or main office |
| `size_estimate` | Rough employee count |
| `decision_maker` | Name + role if known |
| `email` | Best direct email |
| `linkedin_instagram` | Profile or @handle |
| `product_trigger` | Specific product, launch, campaign, or page to reference |
| `why_cgi` | One-line reason CGI helps them |
| `visual_weakness` | What looks improvable today |
| `proposed_offer` | product sprint / launch pack / agency support / industrial explainer |
| `estimated_budget` | Internal guess only — not shown on portfolio |
| `priority_score` | 1–5 (see below) |
| `portfolio_link` | `https://richardandrys.com/?audience=brand` or `?audience=agency` |
| `first_message_sent` | Date |
| `follow_up_1` | Date |
| `follow_up_2` | Date |
| `status` | new / sent / replied / call / won / lost / nurture |
| `notes` | Free text |

## Priority scoring (1–5)

Add **+1** for each true signal (max 5):

1. **Own product** — they sell physical goods or own-brand SKUs
2. **Active marketing** — recent posts, campaigns, launches, paid social
3. **Visible launch** — new product line, seasonal push, rebrand
4. **Weak visuals** — packshots only, stock feel, no motion, dated site
5. **Easy contact** — named person, direct email, responsive social

**5 = contact first.** **3–4 = strong batch.** **1–2 = nurture later.**

## Outreach links

- **Brands:** `https://richardandrys.com/?audience=brand`
- **Agencies:** `https://richardandrys.com/?audience=agency`
- **Optional gate (rare):** `?gate=1`

## Sprint targets (this month)

| Metric | Target |
|--------|--------|
| Leads researched | 120+ |
| First batch sent | 25–40 (see `sprint.md`) |
| Replies | 10–20 |
| Calls | 3–6 |
| Paid projects | 2–4 |

## Files

- [`leads.csv`](./leads.csv) — master lead list (55 starter leads)
- [`production_leads.csv`](./production_leads.csv) — production and VFX studio contacts
- [`messages.md`](./messages.md) — Czech templates + follow-ups
- [`sprint.md`](./sprint.md) — first outreach batch queue & tracking
- [`contact_app.py`](./contact_app.py) — local contact browser and message composer
- [`send_production_emails.py`](./send_production_emails.py) — draft, approve, and send production emails

## Contact and message app

Run the local, copy-only composer:

```bash
python3 outreach/contact_app.py
```

Then open `http://127.0.0.1:8765`. The app reads the latest
`production_leads.csv` whenever the page loads. It shows the current company
and research context, suggests three opening lines, switches between Czech and
English, and copies either the opening or the complete message. The fixed body
does not change between companies. This app cannot send email or modify the
contact sheet.

## Production email sending

The optional mailer reads `production_leads.csv`, writes one editable draft per studio, and sends only drafts you have marked `Approved: yes`. It will not send the same address twice. One run sends at most 20 messages, with a pause between them. It does not attach a CV.

Gmail needs an App Password, not your normal password:

```bash
export SMTP_USER="rich.andrys@gmail.com"
export SMTP_PASSWORD="your-gmail-app-password"
python3 outreach/send_production_emails.py draft --priority 5
python3 outreach/send_production_emails.py approve --company "Blue Faces"
python3 outreach/send_production_emails.py check
python3 outreach/send_production_emails.py send --limit 5
```

`check` only logs in. `send` delivers the next approved drafts that have not already been logged in `outreach/sent_log.csv`. Drafts and the send log stay outside git.
