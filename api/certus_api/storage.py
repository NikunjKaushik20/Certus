"""Content-addressed blob storage.

Files are named by the sha256 of their bytes and sharded two levels deep, so the same photo
uploaded twice costs one copy and the layout ports to S3 unchanged (key = the relative path).

Shard layout::

    storage/
      ab/
        cd/
          abcd1234...jpg
"""
import hashlib
import os
from pathlib import Path

from .config import settings


def digest(data: bytes) -> str:
    """Return the hex sha256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def _path(sha: str, ext: str) -> str:
    """Build the two-level shard path for a given hash and extension."""
    return os.path.join(sha[:2], sha[2:4], f"{sha}{ext}")


def put(data: bytes, ext: str = ".jpg") -> tuple[str, str]:
    """Store bytes; returns ``(sha256, relative_uri)``.

    Existing content is not rewritten — the sha256 check makes this idempotent.
    Writes go to a ``.part`` temporary first so readers never see a partial blob.
    """
    sha: str = digest(data)
    rel: str = _path(sha, ext)
    full: str = os.path.join(settings.storage_dir, rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    if not os.path.exists(full):
        tmp: str = full + ".part"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, full)                       # never leave a half-written blob visible
    return sha, rel


def full_path(rel: str) -> str:
    """Resolve a relative blob URI to an absolute filesystem path."""
    return os.path.join(settings.storage_dir, rel)


def exists(rel: str) -> bool:
    """Check whether a blob exists on disk without reading it."""
    return os.path.isfile(full_path(rel))


def read(rel: str) -> bytes:
    """Read and return the raw bytes of the blob at *rel*."""
    with open(full_path(rel), "rb") as fh:
        return fh.read()
