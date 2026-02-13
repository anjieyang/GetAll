"""Fernet-based encryption for user credentials.

The key is read from GETALL_CREDENTIAL_KEY in the environment / .env file.
If the key is missing, encrypt/decrypt will raise a clear error.
"""

from __future__ import annotations

from functools import lru_cache

from cryptography.fernet import Fernet

from getall.settings import get_settings


@lru_cache
def _get_fernet() -> Fernet:
    key = get_settings().credential_key
    if not key:
        raise RuntimeError(
            "GETALL_CREDENTIAL_KEY is not set. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    return Fernet(key.encode())


def encrypt(plaintext: str) -> str:
    """Encrypt a plaintext string, return URL-safe base64 ciphertext."""
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    """Decrypt a ciphertext string back to plaintext."""
    return _get_fernet().decrypt(ciphertext.encode()).decode()
