from __future__ import annotations

import argparse
import json
import urllib.request


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout_s: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        if v:
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Create account + API key using admin endpoint.")
    parser.add_argument("--base-url", required=True, help="API base URL, e.g. https://api.example.com")
    parser.add_argument("--admin-token", required=True, help="Value of HDRI_ADMIN_TOKEN on the server")
    parser.add_argument("--account-id", default="", help="Optional account id; if omitted server auto-generates")
    parser.add_argument("--initial-tokens", type=int, default=25, help="Initial token balance")
    parser.add_argument("--timeout-s", type=int, default=30)
    args = parser.parse_args()

    payload: dict[str, object] = {"initial_tokens": int(args.initial_tokens)}
    if args.account_id.strip():
        payload["account_id"] = args.account_id.strip()

    url = f"{args.base_url.rstrip('/')}/v1/accounts"
    headers = {"X-Admin-Token": args.admin_token}
    data = _post_json(url, payload, headers=headers, timeout_s=int(args.timeout_s))

    print("Account created:")
    print(json.dumps(data, indent=2))
    print("\nSave api_key securely now; it is returned only at create-time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
