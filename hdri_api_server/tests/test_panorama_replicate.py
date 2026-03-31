import io
import json
import os
import unittest
from unittest.mock import patch

from PIL import Image

import panorama


class _FakeHTTPResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _png_bytes(width: int, height: int) -> bytes:
    img = Image.new("RGB", (width, height), (128, 160, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class PanoramaReplicateTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "REPLICATE_API_TOKEN": "token",
            "REPLICATE_MODEL_VERSION": "verhash",
            "REPLICATE_POLL_TIMEOUT_S": "30",
            "REPLICATE_POLL_INTERVAL_S": "0.01",
        },
        clear=False,
    )
    def test_replicate_output_is_normalized_to_requested_size(self):
        create = json.dumps({"id": "pred-1"}).encode("utf-8")
        poll = json.dumps(
            {
                "id": "pred-1",
                "status": "succeeded",
                "output": "https://example.com/out.png",
            }
        ).encode("utf-8")
        img = _png_bytes(2048, 1024)
        responses = [_FakeHTTPResponse(create), _FakeHTTPResponse(poll), _FakeHTTPResponse(img)]

        def _urlopen_stub(_req, timeout=60):
            if not responses:
                raise AssertionError("Unexpected extra urlopen call")
            return responses.pop(0)

        source = Image.new("RGB", (640, 480), (255, 0, 0))
        with patch("urllib.request.urlopen", side_effect=_urlopen_stub):
            out = panorama.panorama_replicate(
                source,
                width=1024,
                height=512,
                scene_mode="outdoor",
                quality_mode="balanced",
                request_overrides=None,
            )

        self.assertEqual(out.mode, "RGB")
        self.assertEqual(out.size, (1024, 512))


if __name__ == "__main__":
    unittest.main()
