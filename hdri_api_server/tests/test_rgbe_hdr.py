import tempfile
import unittest
from pathlib import Path

import numpy as np

from rgbe_hdr import read_rgbe_hdr, write_rgbe_hdr


class RGBEHDRRoundtripTests(unittest.TestCase):
    def test_roundtrip_preserves_linear_rgb_reasonably(self):
        rgb = np.array(
            [
                [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]],
                [[1.0, 2.0, 4.0], [8.0, 4.0, 2.0]],
            ],
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.hdr"
            write_rgbe_hdr(str(path), rgb)
            out = read_rgbe_hdr(str(path))

        self.assertEqual(out.shape, rgb.shape)
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.allclose(out, rgb, rtol=0.03, atol=0.03))


if __name__ == "__main__":
    unittest.main()
