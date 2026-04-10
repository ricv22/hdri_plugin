from __future__ import annotations

import base64
import json
import os

from remote_provider import RemoteProvider


def _tiny_png_b64() -> str:
    # 1x1 transparent PNG
    raw = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7sM6QAAAAASUVORK5CYII="
    )
    return base64.b64encode(raw).decode("ascii")


def main() -> int:
    provider = RemoteProvider()
    payload = provider._runcomfy_payload(
        image_b64=_tiny_png_b64(),
        width=2048,
        height=1024,
        quality_mode="balanced",
        overrides={
            "prompt": "sunset over mountains",
            "negative_prompt": "blurry, low quality",
            "seed": 42,
            "strength": 0.65,
            "reference_coverage": 0.6,
        },
    )

    print("HDRI_REMOTE_PROVIDER:", os.environ.get("HDRI_REMOTE_PROVIDER", "legacy"))
    print("RUNCOMFY_DEPLOYMENT_ID:", os.environ.get("RUNCOMFY_DEPLOYMENT_ID", "(not set)"))
    print("workflow_api_json attached:", "workflow_api_json" in payload)
    print("overrides attached:", "overrides" in payload)
    print("\nResolved payload preview:")
    print(json.dumps(payload, indent=2)[:8000])

    if "overrides" not in payload:
        print(
            "\nWARNING: No overrides were generated. Set RUNCOMFY_*_NODE_IDS "
            "to map generic request fields into your workflow nodes."
        )
        return 1

    print("\nOK: RunComfy override mapping payload is being generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
