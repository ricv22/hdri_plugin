import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from erp_seam import seam_fix_erp_wrap_blur
from job_store import JobStore
from panorama_prompt import DEFAULT_BASE_PANORAMA_PROMPT, compose_panorama_prompt


class PanoramaPromptTests(unittest.TestCase):
    def test_empty_user_returns_none(self) -> None:
        self.assertIsNone(compose_panorama_prompt(""))
        self.assertIsNone(compose_panorama_prompt(None))
        self.assertIsNone(compose_panorama_prompt("   "))

    def test_user_before_base_by_default(self) -> None:
        out = compose_panorama_prompt("sunset golden hour")
        self.assertIsNotNone(out)
        assert out is not None
        self.assertTrue(out.startswith("sunset golden hour"))
        self.assertIn(DEFAULT_BASE_PANORAMA_PROMPT, out)

    def test_user_after_base_when_env_set(self) -> None:
        with patch.dict(os.environ, {"HDRI_PROMPT_USER_POSITION": "after"}, clear=False):
            out = compose_panorama_prompt("misty forest")
        self.assertIsNotNone(out)
        assert out is not None
        self.assertTrue(out.startswith(DEFAULT_BASE_PANORAMA_PROMPT))
        self.assertTrue(out.endswith("misty forest"))


class ErpSeamTests(unittest.TestCase):
    def test_seam_fix_reduces_left_right_discontinuity(self) -> None:
        w, h = 512, 256
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        rgb[..., 0] = np.linspace(0.0, 1.0, w, dtype=np.float32)
        rgb[..., 1] = 0.25
        rgb[..., 2] = 0.5

        before = float(abs(rgb[10, 0, 0] - rgb[10, -1, 0]))
        fixed = seam_fix_erp_wrap_blur(rgb, band_frac=0.08, blur_sigma=12.0)
        after = float(abs(fixed[10, 0, 0] - fixed[10, -1, 0]))
        self.assertLess(after, before)


class RegistrationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self._tmp.close()
        self.store = JobStore(self._tmp.name)

    def tearDown(self) -> None:
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_register_account_by_email(self) -> None:
        self.store.ensure_account("acct-a", initial_tokens=3, email="user@example.com")
        row = self.store.get_account_by_email("user@example.com")
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["account_id"], "acct-a")
        self.assertEqual(row["tokens_remaining"], 3)

    def test_purchase_idempotent(self) -> None:
        self.store.ensure_account("acct-a", initial_tokens=0)
        ok1 = self.store.record_purchase(
            purchase_id="p1",
            account_id="acct-a",
            package_id="tokens_10",
            tokens=10,
            provider="stripe",
            provider_ref="sess_123",
        )
        ok2 = self.store.record_purchase(
            purchase_id="p2",
            account_id="acct-a",
            package_id="tokens_10",
            tokens=10,
            provider="stripe",
            provider_ref="sess_123",
        )
        self.assertTrue(ok1)
        self.assertFalse(ok2)


if __name__ == "__main__":
    unittest.main()
