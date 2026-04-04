import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import ai_hdr
import app as app_mod


class HDRModeContractTests(unittest.TestCase):
    def test_resolve_mode_prefers_explicit_enum(self):
        req = app_mod.HdriRequest(
            image_b64="x",
            hdr_reconstruction_mode="off",
            heuristic_hdr_lift=True,
        )
        self.assertEqual(app_mod._resolve_hdr_mode(req), "off")

    def test_resolve_mode_maps_legacy_boolean(self):
        req = app_mod.HdriRequest(image_b64="x", heuristic_hdr_lift=False)
        self.assertEqual(app_mod._resolve_hdr_mode(req), "off")


class AIHDRModuleTests(unittest.TestCase):
    def test_embedded_ai_hdr_returns_non_negative_rgb(self):
        rgb_lin = np.full((32, 64, 3), 0.2, dtype=np.float32)
        out = ai_hdr.reconstruct_ai_hdr(rgb_lin, quality_mode="balanced", model_name="embedded")
        self.assertEqual(out.shape, rgb_lin.shape)
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.all(out >= 0.0))
        self.assertGreater(float(out.mean()), float(rgb_lin.mean()))


class HDRFailoverTests(unittest.TestCase):
    @patch("app.write_rgbe_hdr", autospec=True)
    @patch("app._provider.wait_for_result", autospec=True)
    @patch("app.reconstruct_ai_hdr", autospec=True)
    @patch.dict("os.environ", {"AI_HDR_FAILOVER_MODE": "heuristic"}, clear=False)
    def test_ai_mode_falls_back_to_heuristic_on_failure(
        self,
        mock_reconstruct,
        mock_wait,
        mock_write,
    ):
        mock_reconstruct.side_effect = RuntimeError("model unavailable")
        mock_wait.return_value = (Image.new("RGB", (1024, 512), (140, 120, 100)), "http_json")

        req = app_mod.HdriRequest(
            image_b64="x",
            output_width=1024,
            output_height=512,
            hdr_reconstruction_mode="ai_fast",
        )
        res = app_mod._generate_hdri(req)
        self.assertEqual(res.hdr_reconstruction_mode, "heuristic")
        self.assertTrue(mock_write.called)


if __name__ == "__main__":
    unittest.main()
