import os
import tempfile
import unittest

from auth import hash_password, validate_password, verify_password
from fastapi import HTTPException
from job_store import JobStore


class PasswordHashTests(unittest.TestCase):
    def test_hash_and_verify_roundtrip(self) -> None:
        stored = hash_password("correct horse battery")
        self.assertTrue(verify_password("correct horse battery", stored))
        self.assertFalse(verify_password("wrong", stored))

    def test_validate_password_length(self) -> None:
        with self.assertRaises(HTTPException):
            validate_password("short")


class PasswordStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self._tmp.close()
        self.store = JobStore(self._tmp.name)

    def tearDown(self) -> None:
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_set_and_read_password_hash(self) -> None:
        self.store.ensure_account("acct-a", initial_tokens=3, email="user@example.com")
        stored = hash_password("secret-password")
        self.store.set_password_hash("acct-a", stored)
        row = self.store.get_account_by_email("user@example.com")
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["password_hash"], stored)

    def test_deactivate_api_keys(self) -> None:
        from auth import hash_api_key

        self.store.ensure_account("acct-a", initial_tokens=0)
        self.store.ensure_api_key(hash_api_key("key-one"), "acct-a")
        self.store.ensure_api_key(hash_api_key("key-two"), "acct-a")
        self.store.deactivate_api_keys_for_account("acct-a")
        self.assertIsNone(self.store.get_account_by_api_key_hash(hash_api_key("key-one")))
        self.assertIsNone(self.store.get_account_by_api_key_hash(hash_api_key("key-two")))


if __name__ == "__main__":
    unittest.main()
