import unittest


try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "torch is not installed")
class ITMTrainingModelTests(unittest.TestCase):
    def test_itm_model_forward_returns_non_negative_hdr(self):
        from training.itm_model import ITMNet

        model = ITMNet(base_channels=8)
        x = torch.rand(1, 3, 64, 128, dtype=torch.float32)
        y = model(x)
        self.assertEqual(tuple(y.shape), (1, 3, 64, 128))
        self.assertTrue(torch.all(y >= 0).item())


if __name__ == "__main__":
    unittest.main()
