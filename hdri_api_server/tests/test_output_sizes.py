import unittest

from app import _validate_output_size
from fastapi import HTTPException


class OutputSizeValidationTests(unittest.TestCase):
    def test_allows_1024_and_2048(self) -> None:
        _validate_output_size(1024, 512)
        _validate_output_size(2048, 1024)

    def test_rejects_4096_when_upscale_disabled(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            _validate_output_size(4096, 2048)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("4096x2048", str(ctx.exception.detail))


if __name__ == "__main__":
    unittest.main()
