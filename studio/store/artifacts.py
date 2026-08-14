# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Artefact storage, behind an interface (ADR-004).

"Object storage" in Phase 0 is a directory on disk. MinIO or S3 substitutes without touching
callers, which is why every path a caller sees is **relative to the store root** -- an absolute path
baked into the database would be a deployment detail that outlives the deployment.

The store computes a SHA-256 as it copies. That is what makes "this artefact is still the file that
was written" a checkable claim rather than an assumption, and it costs one pass over a file that is
being read anyway.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

#: Read in 1 MiB blocks: large enough that the syscall overhead disappears, small enough that a
#: 3.6 MB npz never sits in memory twice.
_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class StoredArtifact:
    """What the database records about one stored file. Never the bytes."""

    key: str
    size_bytes: int
    sha256: str
    content_type: str


@runtime_checkable
class ArtifactStore(Protocol):
    """Where run outputs live. Nothing above this may assume a filesystem."""

    def put(self, run_id: str, kind: str, source: Path) -> StoredArtifact:
        """Copy ``source`` into the store under ``run_id``, returning what to record."""
        ...

    def open_path(self, key: str) -> Path:
        """Resolve a stored key to a readable path."""
        ...

    def exists(self, key: str) -> bool: ...


def sha256_of(path: Path) -> str:
    """Streaming SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


#: Extension -> content type, for the artefacts Studio actually writes. Deliberately small: a
#: general mimetype lookup would guess, and a wrong content type on an npz is worse than none.
_CONTENT_TYPES = {
    ".npz": "application/x-npz",
    ".json": "application/json",
    ".log": "text/plain",
    ".png": "image/png",
}


class LocalDirectoryStore:
    """The Phase-0 implementation: a directory tree, one subdirectory per run."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, run_id: str, kind: str, source: Path) -> StoredArtifact:
        """Copy in, hash, and return the record.

        Raises:
            FileNotFoundError: If the source is missing. A run that was supposed to produce an
                artefact and did not is a failure to surface, not a row to omit.
        """
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(
                f"artefact {kind!r} for run {run_id} is missing at {source}; a run that did not "
                f"produce it should fail rather than be recorded without it"
            )
        destination = self.root / run_id / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return StoredArtifact(
            key=str(destination.relative_to(self.root)),
            size_bytes=destination.stat().st_size,
            sha256=sha256_of(destination),
            content_type=_CONTENT_TYPES.get(source.suffix, "application/octet-stream"),
        )

    def open_path(self, key: str) -> Path:
        path = self.root / key
        if not path.is_file():
            raise FileNotFoundError(f"artefact {key!r} is recorded but missing from {self.root}")
        return path

    def exists(self, key: str) -> bool:
        return (self.root / key).is_file()

    def verify(self, key: str, expected_sha256: str) -> bool:
        """Whether the stored bytes still hash to what was recorded.

        The reason the checksum is stored at all: silent corruption and a helpfully "tidied"
        directory look identical from the database.
        """
        return self.exists(key) and sha256_of(self.open_path(key)) == expected_sha256


__all__ = ["ArtifactStore", "LocalDirectoryStore", "StoredArtifact", "sha256_of"]
