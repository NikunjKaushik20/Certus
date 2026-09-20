"""Content-addressed blob storage.

Files are named by the sha256 of their bytes and sharded two levels deep, so the same photo
uploaded twice costs one copy and the layout ports to S3 unchanged (key = the relative path).
"""
import hashlib
import os

from .config import settings


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _path(sha: str, ext: str) -> str:
    return os.path.join(sha[:2], sha[2:4], f"{sha}{ext}")


def put(data: bytes, ext: str = ".jpg") -> tuple[str, str]:
    """Store bytes; returns (sha256, relative uri). Existing content is not rewritten."""
    sha = digest(data)
    rel = _path(sha, ext)
    full = os.path.join(settings.storage_dir, rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    if not os.path.exists(full):
        tmp = full + ".part"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, full)                       # never leave a half-written blob visible
    return sha, rel


def full_path(rel: str) -> str:
    return os.path.join(settings.storage_dir, rel)


def read(rel: str) -> bytes:
    with open(full_path(rel), "rb") as fh:
        return fh.read()
