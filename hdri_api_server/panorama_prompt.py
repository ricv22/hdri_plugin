"""Panorama prompt composition for ERP outpaint workflows."""

from __future__ import annotations

import os

DEFAULT_BASE_PANORAMA_PROMPT = (
    "Fill the green spaces according to the image. Outpaint as a seamless 360 equirectangular "
    "panorama (2:1). Keep the horizon level. Match left and right edges."
)


def base_panorama_prompt() -> str:
    custom = os.environ.get("HDRI_BASE_PANORAMA_PROMPT", "").strip()
    if custom:
        return custom
    return DEFAULT_BASE_PANORAMA_PROMPT


def compose_panorama_prompt(user_prompt: str | None) -> str | None:
    """Merge user text with the required ERP outpaint instructions.

    User text is placed before or after the base block depending on
    ``HDRI_PROMPT_USER_POSITION`` (``before`` default, or ``after``).
    Returns ``None`` when the user prompt is empty so workflow defaults apply.
    """
    user = (user_prompt or "").strip()
    if not user:
        return None

    base = base_panorama_prompt()
    position = os.environ.get("HDRI_PROMPT_USER_POSITION", "before").strip().lower()
    if position == "before":
        return f"{user} {base}".strip()
    return f"{base} {user}".strip()
