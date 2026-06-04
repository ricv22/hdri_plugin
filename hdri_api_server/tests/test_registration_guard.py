from __future__ import annotations

import os
import tempfile
import unittest

from job_store import JobStore
from registration_guard import (
    canonicalize_email,
    free_tokens_for_new_registration,
    hash_client_ip,
    is_disposable_email,
)


class RegistrationGuardTests(unittest.TestCase):
    def test_gmail_aliases_canonicalize_same(self) -> None:
        a = canonicalize_email("User.Name+tag@gmail.com")
        b = canonicalize_email("username@gmail.com")
        self.assertEqual(a, b)

    def test_disposable_domain_blocked(self) -> None:
        self.assertTrue(is_disposable_email("x@mailinator.com"))

    def test_ip_limit_grants_free_tokens_once(self) -> None:
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        tmp.close()
        try:
            store = JobStore(tmp.name)
            ip_hash = hash_client_ip("203.0.113.9")
            self.assertEqual(free_tokens_for_new_registration(store, ip_hash, 10), 10)
            store.record_registration(
                account_id="a1",
                ip_hash=ip_hash,
                email_canonical="one@example.com",
                free_tokens_granted=10,
            )
            self.assertEqual(free_tokens_for_new_registration(store, ip_hash, 10), 10)
            store.record_registration(
                account_id="a2",
                ip_hash=ip_hash,
                email_canonical="two@example.com",
                free_tokens_granted=10,
            )
            self.assertEqual(free_tokens_for_new_registration(store, ip_hash, 10), 0)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
