from __future__ import annotations

import json
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
                scene_mode="auto",
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

    def test_coverage_to_hfov_mapping_is_camera_like(self) -> None:
        # New mapping target:
        #   0.15 -> 35 deg
        #   0.60 -> ~73.57 deg
        #   0.85 -> 95 deg
        self.assertAlmostEqual(self.provider._runcomfy_coverage_to_fov_deg(0.15), 35.0, places=3)
        self.assertAlmostEqual(self.provider._runcomfy_coverage_to_fov_deg(0.60), 73.5714285714, places=3)
        self.assertAlmostEqual(self.provider._runcomfy_coverage_to_fov_deg(0.85), 95.0, places=3)
        # Clamp behavior
        self.assertAlmostEqual(self.provider._runcomfy_coverage_to_fov_deg(0.0), 35.0, places=3)
        self.assertAlmostEqual(self.provider._runcomfy_coverage_to_fov_deg(1.0), 95.0, places=3)

    def test_panorama_stickers_nodes_get_state_json_and_preset(self) -> None:
        env = {
            "RUNCOMFY_PANORAMA_STICKERS_NODE_IDS": "56",
            "RUNCOMFY_PANORAMA_STICKERS_USE_EXTERNAL_IMAGE": "0",
            "RUNCOMFY_PROMPT_NODE_IDS": "6",
            "RUNCOMFY_NEGATIVE_PROMPT_NODE_IDS": "33",
            "RUNCOMFY_SEED_NODE_IDS": "31",
            "RUNCOMFY_STRENGTH_NODE_IDS": "31",
            "RUNCOMFY_STEPS_NODE_IDS": "31",
            "RUNCOMFY_DEFAULT_REFERENCE_COVERAGE": "0.5",
        }
        with patch.dict(os.environ, env, clear=False):
            out = self.provider._build_runcomfy_overrides(
                image_b64="YWJj",
                width=2048,
                height=1024,
                scene_mode="auto",
                quality_mode="balanced",
                overrides={
                    "prompt": "sunset",
                    "negative_prompt": "blur",
                    "seed": 7,
                    "strength": 0.55,
                },
            )
        self.assertEqual(out["56"]["inputs"]["output_preset"], "2048")
        self.assertEqual(out["56"]["inputs"]["bg_color"], "#00ff00")
        state = json.loads(out["56"]["inputs"]["state_json"])
        self.assertEqual(state["version"], 1)
        self.assertIn("data:image/jpeg;base64,YWJj", state["assets"]["asset_uploaded"]["filename"])
        self.assertEqual(out["6"]["inputs"]["text"], "sunset")
        self.assertEqual(out["33"]["inputs"]["text"], "blur")
        self.assertEqual(out["31"]["inputs"]["seed"], 7)
        self.assertEqual(out["31"]["inputs"]["denoise"], 0.55)
        self.assertEqual(out["31"]["inputs"]["steps"], 24)
        # 0.5 coverage should now be significantly narrower than old 106.25 deg.
        self.assertAlmostEqual(state["stickers"][0]["hFOV_deg"], 65.0, places=3)

    def test_runcomfy_style_overrides_passthrough(self) -> None:
        src = {"5": {"inputs": {"image": "data:image/png;base64,abcd"}}}
        out = self.provider._build_runcomfy_overrides(
            image_b64="YWJj",
            width=1024,
            height=512,
            scene_mode="auto",
            quality_mode="fast",
            overrides=src,
        )
        self.assertEqual(out, src)

    def test_panorama_stickers_native_sticker_state_mode(self) -> None:
        env = {
            "RUNCOMFY_IMAGE_NODE_IDS": "11",
            "RUNCOMFY_PANORAMA_STICKERS_NODE_IDS": "56",
            "RUNCOMFY_PANORAMA_STICKERS_USE_EXTERNAL_IMAGE": "1",
            "RUNCOMFY_INPUT_IMAGE_TRANSPORT": "data_uri",
            "RUNCOMFY_DEFAULT_REFERENCE_COVERAGE": "0.6",
        }
        with patch.dict(os.environ, env, clear=False):
            out = self.provider._build_runcomfy_overrides(
                image_b64="YWJj",
                width=2048,
                height=1024,
                scene_mode="outdoor",
                quality_mode="balanced",
                overrides={
                    "placement_coverage": 0.5,
                    "placement_yaw_deg": 12.0,
                    "placement_pitch_deg": -6.0,
                    "placement_rotation_deg": 18.0,
                    "placement_hfov_deg": 72.0,
                },
            )
        self.assertEqual(out["11"]["inputs"]["image"], "data:image/jpeg;base64,YWJj")
        self.assertEqual(out["56"]["inputs"]["output_preset"], "2048")
        sticker_state = json.loads(out["56"]["inputs"]["sticker_state"])
        self.assertEqual(sticker_state["kind"], "pano_sticker_state")
        self.assertEqual(sticker_state["version"], 1)
        self.assertEqual(sticker_state["pose"]["yaw_deg"], 12.0)
        self.assertEqual(sticker_state["pose"]["pitch_deg"], -6.0)
        self.assertEqual(sticker_state["pose"]["rot_deg"], 18.0)
        self.assertEqual(sticker_state["pose"]["roll_deg"], 18.0)
        self.assertEqual(sticker_state["pose"]["hFOV_deg"], 72.0)
        self.assertIn("vFOV_deg", sticker_state["pose"])
        legacy_state = json.loads(out["56"]["inputs"]["state_json"])
        self.assertEqual(legacy_state["stickers"][0]["yaw_deg"], 12.0)
        self.assertEqual(legacy_state["stickers"][0]["pitch_deg"], -6.0)
        self.assertEqual(legacy_state["stickers"][0]["rot_deg"], 18.0)

    def test_native_sticker_state_uses_new_coverage_mapping_when_hfov_unset(self) -> None:
        env = {
            "RUNCOMFY_IMAGE_NODE_IDS": "11",
            "RUNCOMFY_PANORAMA_STICKERS_NODE_IDS": "56",
            "RUNCOMFY_PANORAMA_STICKERS_USE_EXTERNAL_IMAGE": "1",
            "RUNCOMFY_INPUT_IMAGE_TRANSPORT": "data_uri",
        }
        with patch.dict(os.environ, env, clear=False):
            out = self.provider._build_runcomfy_overrides(
                image_b64="YWJj",
                width=2048,
                height=1024,
                scene_mode="auto",
                quality_mode="balanced",
                overrides={
                    "placement_coverage": 0.6,
                    "placement_yaw_deg": 10.0,
                },
            )
        sticker_state = json.loads(out["56"]["inputs"]["sticker_state"])
        self.assertAlmostEqual(sticker_state["pose"]["hFOV_deg"], 73.5714285714, places=3)

    def test_panorama_stickers_native_mode_with_legacy_state_fallback(self) -> None:
        env = {
            "RUNCOMFY_IMAGE_NODE_IDS": "11",
            "RUNCOMFY_PANORAMA_STICKERS_NODE_IDS": "56",
            "RUNCOMFY_PANORAMA_STICKERS_USE_EXTERNAL_IMAGE": "1",
            "RUNCOMFY_PANORAMA_STICKERS_NATIVE_STATEJSON_FALLBACK": "1",
            "RUNCOMFY_INPUT_IMAGE_TRANSPORT": "data_uri",
            "RUNCOMFY_DEFAULT_REFERENCE_COVERAGE": "0.6",
        }
        with patch.dict(os.environ, env, clear=False):
            out = self.provider._build_runcomfy_overrides(
                image_b64="YWJj",
                width=2048,
                height=1024,
                scene_mode="outdoor",
                quality_mode="balanced",
                overrides={
                    "placement_coverage": 0.5,
                    "placement_yaw_deg": 12.0,
                    "placement_pitch_deg": -6.0,
                    "placement_rotation_deg": 18.0,
                    "placement_hfov_deg": 72.0,
                },
            )
        self.assertEqual(out["11"]["inputs"]["image"], "data:image/jpeg;base64,YWJj")
        self.assertEqual(out["56"]["inputs"]["output_preset"], "2048")
        sticker_state = json.loads(out["56"]["inputs"]["sticker_state"])
        self.assertEqual(sticker_state["kind"], "pano_sticker_state")
        self.assertEqual(sticker_state["version"], 1)
        self.assertEqual(sticker_state["pose"]["yaw_deg"], 12.0)
        self.assertEqual(sticker_state["pose"]["pitch_deg"], -6.0)
        self.assertEqual(sticker_state["pose"]["rot_deg"], 18.0)
        self.assertEqual(sticker_state["pose"]["roll_deg"], 18.0)
        self.assertEqual(sticker_state["pose"]["hFOV_deg"], 72.0)
        self.assertIn("vFOV_deg", sticker_state["pose"])
        legacy_state = json.loads(out["56"]["inputs"]["state_json"])
        self.assertEqual(legacy_state["stickers"][0]["yaw_deg"], 12.0)
        self.assertEqual(legacy_state["stickers"][0]["pitch_deg"], -6.0)
        self.assertEqual(legacy_state["stickers"][0]["rot_deg"], 18.0)

    def test_select_runcomfy_image_url_skips_panorama_stickers_preview(self) -> None:
        result_data = {
            "outputs": {
                "56": {
                    "images": [
                        {
                            "type": "temp",
                            "url": "https://example.com/temp/sticker_preview.png",
                        }
                    ]
                },
                "66": {
                    "images": [
                        {
                            "type": "output",
                            "url": "https://example.com/output/final_pano.png",
                        }
                    ]
                },
            }
        }
        env = {
            "RUNCOMFY_PANORAMA_STICKERS_NODE_IDS": "56",
        }
        with patch.dict(os.environ, env, clear=False):
            url, node_id = self.provider._select_runcomfy_image_url(result_data)
        self.assertEqual(url, "https://example.com/output/final_pano.png")
        self.assertEqual(node_id, "66")

    def test_auto_detects_sticker_and_loadimage_nodes_from_workflow_json(self) -> None:
        workflow_api_json = {
            "11": {
                "class_type": "LoadImage",
                "inputs": {"image": "__RUNCOMFY_INPUT_IMAGE__"},
            },
            "56": {
                "class_type": "PanoramaStickers",
                "inputs": {
                    "sticker_image": ["11", 0],
                    "sticker_state": "",
                    "state_json": "",
                },
            },
        }
        # Clear mapping envs to verify workflow auto-detection path.
        env = {
            "RUNCOMFY_IMAGE_NODE_IDS": "",
            "RUNCOMFY_PANORAMA_STICKERS_NODE_IDS": "",
            "RUNCOMFY_PANORAMA_STICKERS_USE_EXTERNAL_IMAGE": "1",
            "RUNCOMFY_INPUT_IMAGE_TRANSPORT": "data_uri",
        }
        with patch.dict(os.environ, env, clear=False):
            out = self.provider._build_runcomfy_overrides(
                image_b64="YWJj",
                width=2048,
                height=1024,
                scene_mode="auto",
                quality_mode="balanced",
                overrides={
                    "placement_coverage": 0.5,
                    "placement_yaw_deg": 12.0,
                    "placement_pitch_deg": -6.0,
                },
                workflow_api_json=workflow_api_json,
            )
        self.assertIn("11", out)
        self.assertIn("56", out)
        self.assertEqual(out["11"]["inputs"]["image"], "data:image/jpeg;base64,YWJj")
        self.assertIn("sticker_state", out["56"]["inputs"])

    def test_placement_forces_native_sticker_mode_even_if_env_disables_it(self) -> None:
        env = {
            "RUNCOMFY_IMAGE_NODE_IDS": "11",
            "RUNCOMFY_PANORAMA_STICKERS_NODE_IDS": "56",
            "RUNCOMFY_PANORAMA_STICKERS_USE_EXTERNAL_IMAGE": "0",
            "RUNCOMFY_INPUT_IMAGE_TRANSPORT": "data_uri",
            "RUNCOMFY_LOAD_IMAGE_COMPOSE_ERP": "1",
        }
        with patch.dict(os.environ, env, clear=False):
            out = self.provider._build_runcomfy_overrides(
                image_b64="YWJj",
                width=2048,
                height=1024,
                scene_mode="auto",
                quality_mode="balanced",
                overrides={
                    "placement_yaw_deg": 25.0,
                    "placement_pitch_deg": -8.0,
                },
            )
        self.assertIn("sticker_state", out["56"]["inputs"])
        legacy_state = json.loads(out["56"]["inputs"]["state_json"])
        self.assertEqual(legacy_state["stickers"][0]["yaw_deg"], 25.0)
        self.assertEqual(legacy_state["stickers"][0]["pitch_deg"], -8.0)

    def test_placement_prefers_workflow_sticker_image_node_over_env_image_nodes(self) -> None:
        workflow_api_json = {
            "11": {
                "class_type": "LoadImage",
                "inputs": {"image": "__RUNCOMFY_INPUT_IMAGE__"},
            },
            "56": {
                "class_type": "PanoramaStickers",
                "inputs": {
                    "sticker_image": ["11", 0],
                    "sticker_state": "",
                    "state_json": "",
                },
            },
        }
        env = {
            "RUNCOMFY_IMAGE_NODE_IDS": "99",
            "RUNCOMFY_PANORAMA_STICKERS_NODE_IDS": "56",
            "RUNCOMFY_PANORAMA_STICKERS_USE_EXTERNAL_IMAGE": "1",
            "RUNCOMFY_INPUT_IMAGE_TRANSPORT": "data_uri",
        }
        with patch.dict(os.environ, env, clear=False):
            out = self.provider._build_runcomfy_overrides(
                image_b64="YWJj",
                width=2048,
                height=1024,
                scene_mode="auto",
                quality_mode="balanced",
                overrides={
                    "placement_yaw_deg": 12.0,
                },
                workflow_api_json=workflow_api_json,
            )
        self.assertIn("11", out)
        self.assertNotIn("99", out)


if __name__ == "__main__":
    unittest.main()
