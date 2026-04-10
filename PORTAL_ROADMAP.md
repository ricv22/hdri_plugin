# Portal Roadmap (Deferred by Design)

This project should remain Blender-first until concrete business triggers justify a second product surface.

## Current position (V1)

Keep user workflow in Blender:

- API key entry
- token visibility
- async generation + cancel
- result application to world lighting

Backend remains hosted and production-ready, but no public signup/login web app yet.

## Build a portal only when triggers are met

Create the web portal when at least two of these are true:

1. Users request self-service signup/login repeatedly.
2. You add paid plans or token checkout.
3. Users need generation history/downloads outside Blender.
4. Team/organization account management becomes necessary.
5. HDRI library size is large enough to require web browsing/search.

## Portal Phase 1 (small, practical)

Scope:

- login
- API key management
- token balance and refill
- job history and download links

Do not include at Phase 1:

- browser-side generation UI
- complex social/library features
- broad search/recommendation stack

## Portal Phase 2 (library/search)

Add only after enough assets and metadata exist.

Minimum requirements first:

- stored per-HDRI metadata (prompt, tags, scene type, quality, date)
- ownership and sharing policy
- indexing strategy

Then add:

- library pages
- filters
- search ranking

## Decision rule

If user success still happens inside Blender, keep investing in:

- generation quality
- latency
- reliability
- account/billing backend

Only invest in web UI once those are stable and demand is proven.
