from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from remote_provider import RemoteProvider


class RemoteProviderRunComfyMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = RemoteProvider()

    def test_generic_overrides_are_mapped_to_node_inputs(self) -> None:
        env = {
            "RUNCOMFY_IMAGE_NODE_IDS": "11",
            "RUNCOMFY_PROMPT_NODE_IDS": "12",
            "RUNCOMFY_NEGATIVE_PROMPT_NODE_IDS": "13",
            "RUNCOMFY_SEED_NODE_IDS": "14",
            "RUNCOMFY_STRENGTH_NODE_IDS": "14",
            "RUNCOMFY_REFERENCE_COVERAGE_NODE_IDS": "15",
            "RUNCOMFY_DIMENSION_NODE_IDS": "16",
            "RUNCOMFY_STEPS_NODE_IDS": "14",
        }
        with patch.dict(os.environ, env, clear=False):
            out = self.provider._build_runcomfy_overrides(
                image_b64="YWJj",
                width=2048,
                height=1024,
                quality_mode="balanced",
                overrides={
                    "prompt": "test prompt",
                    "negative_prompt": "bad",
                    "seed": 9,
                    "strength": 0.4,
                    "reference_coverage": 0.6,
                },
            )
        self.assertEqual(out["11"]["inputs"]["image"], "data:image/jpeg;base64,YWJj")
        self.assertEqual(out["12"]["inputs"]["text"], "test prompt")
        self.assertEqual(out["13"]["inputs"]["text"], "bad")
        self.assertEqual(out["14"]["inputs"]["seed"], 9)
        self.assertEqual(out["14"]["inputs"]["denoise"], 0.4)
        self.assertEqual(out["14"]["inputs"]["steps"], 24)
        self.assertEqual(out["15"]["inputs"]["reference_coverage"], 0.6)
        self.assertEqual(out["16"]["inputs"]["width"], 2048)
        self.assertEqual(out["16"]["inputs"]["height"], 1024)

    def test_runcomfy_style_overrides_passthrough(self) -> None:
        src = {"5": {"inputs": {"image": "data:image/png;base64,abcd"}}}
        out = self.provider._build_runcomfy_overrides(
            image_b64="YWJj",
            width=1024,
            height=512,
            quality_mode="fast",
            overrides=src,
        )
        self.assertEqual(out, src)


if __name__ == "__main__":
    unittest.main()
