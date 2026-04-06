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

    def test_resolve_mode_accepts_comfyui_hdr(self):
        req = app_mod.HdriRequest(image_b64="x", hdr_reconstruction_mode="comfyui_hdr")
        self.assertEqual(app_mod._resolve_hdr_mode(req), "comfyui_hdr")


class AIHDRModuleTests(unittest.TestCase):
    def test_embedded_ai_hdr_returns_non_negative_rgb(self):
        rgb_lin = np.full((32, 64, 3), 0.2, dtype=np.float32)
        out = ai_hdr.reconstruct_ai_hdr(rgb_lin, quality_mode="balanced", model_name="embedded")
        self.assertEqual(out.shape, rgb_lin.shape)
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.all(out >= 0.0))
        self.assertGreater(float(out.mean()), float(rgb_lin.mean()))

    def test_embedded_ai_hdr_prioritizes_highlights_over_midtones(self):
        rgb_lin = np.array(
            [[[0.18, 0.16, 0.14], [0.45, 0.40, 0.35], [0.90, 0.85, 0.80]]],
            dtype=np.float32,
        )
        out = ai_hdr.reconstruct_ai_hdr(rgb_lin, quality_mode="balanced", model_name="embedded")
        mid_gain = float(out[0, 0, 0] / rgb_lin[0, 0, 0])
        bright_gain = float(out[0, 2, 0] / rgb_lin[0, 2, 0])
        self.assertLess(mid_gain, 2.0)
        self.assertGreater(bright_gain, mid_gain)

class HeuristicHDRLiftTests(unittest.TestCase):
    def test_heuristic_lift_does_not_apply_vertical_gradient_bias(self):
        rgb_lin = np.full((2, 1, 3), 0.6, dtype=np.float32)
        out = app_mod._fake_hdr_lift(rgb_lin, "balanced")
        self.assertAlmostEqual(float(out[0, 0, 0]), float(out[1, 0, 0]), places=6)

    def test_heuristic_lift_focuses_gain_on_bright_regions(self):
        rgb_lin = np.array(
            [[[0.04, 0.04, 0.04], [0.18, 0.16, 0.14], [0.90, 0.85, 0.80]]],
            dtype=np.float32,
        )
        out = app_mod._fake_hdr_lift(rgb_lin, "balanced")
        dark_gain = float(out[0, 0, 0] / rgb_lin[0, 0, 0])
        mid_gain = float(out[0, 1, 0] / rgb_lin[0, 1, 0])
        bright_gain = float(out[0, 2, 0] / rgb_lin[0, 2, 0])
        self.assertLess(dark_gain, 1.3)
        self.assertGreater(bright_gain, mid_gain)


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

    @patch("app.write_rgbe_hdr", autospec=True)
    @patch("app._provider.wait_for_result", autospec=True)
    @patch("app._run_comfyui_hdr_restore", autospec=True)
    @patch.dict("os.environ", {"AI_HDR_FAILOVER_MODE": "heuristic"}, clear=False)
    def test_comfyui_hdr_mode_falls_back_to_heuristic_on_failure(
        self,
        mock_external,
        mock_wait,
        mock_write,
    ):
        mock_external.side_effect = RuntimeError("worker unavailable")
        mock_wait.return_value = (Image.new("RGB", (1024, 512), (140, 120, 100)), "http_json")

        req = app_mod.HdriRequest(
            image_b64="x",
            output_width=1024,
            output_height=512,
            hdr_reconstruction_mode="comfyui_hdr",
        )
        res = app_mod._generate_hdri(req)
        self.assertEqual(res.hdr_reconstruction_mode, "heuristic")
        self.assertTrue(mock_write.called)


if __name__ == "__main__":
    unittest.main()
