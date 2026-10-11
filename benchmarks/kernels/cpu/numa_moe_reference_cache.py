# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent per-layer reference outputs for offline NUMA MoE validation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

_FORMAT_VERSION = 1
_CHUNK_SIZE = 1024 * 1024


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_file(path: Path, write: Any) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as target:
            write(target)
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise RuntimeError(f"Cache file is immutable: {path.name}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _tensor_spec(tensor: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}


class OracleCache:
    """Store immutable, identity-bound BF16 oracle outputs one layer at a time."""

    def __init__(
        self, directory: Path, identity: dict, read_only: bool = False
    ) -> None:
        self.directory = Path(directory).resolve()
        identity_bytes = _json_bytes(identity)
        self.identity = json.loads(identity_bytes)
        self.identity_sha256 = hashlib.sha256(identity_bytes).hexdigest()
        self.read_only = read_only
        manifest_path = self.directory / "cache.json"
        manifest = {
            "format_version": _FORMAT_VERSION,
            "identity": self.identity,
            "identity_sha256": self.identity_sha256,
        }

        if read_only:
            self._verify_manifest(manifest_path, manifest)
            return

        self.directory.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            self._verify_manifest(manifest_path, manifest)
        else:
            if any(self.directory.iterdir()):
                raise RuntimeError("Cache has files but no identity manifest")
            encoded = _json_bytes(manifest)
            _publish_file(manifest_path, lambda target: target.write(encoded))

    def _verify_manifest(self, path: Path, expected: dict[str, Any]) -> None:
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Cache identity manifest is missing or invalid") from exc
        if actual != expected:
            raise RuntimeError("Cache identity does not match this run")

    def _paths(self, layer: int) -> tuple[Path, Path]:
        stem = f"layer-{layer}"
        return self.directory / f"{stem}.pt", self.directory / f"{stem}.json"

    def _record_specs(self, layer: int, records: list[dict]) -> list[dict[str, Any]]:
        specs = []
        for index, record in enumerate(records):
            if "layer" in record and record["layer"] != layer:
                raise ValueError(f"Record {index} belongs to another layer")
            specs.append(_tensor_spec(record["hidden"]))
        return specs

    def _verify_identity(self) -> None:
        self._verify_manifest(
            self.directory / "cache.json",
            {
                "format_version": _FORMAT_VERSION,
                "identity": self.identity,
                "identity_sha256": self.identity_sha256,
            },
        )

    def load(self, layer: int, records: list[dict]) -> list[torch.Tensor] | None:
        self._verify_identity()
        archive_path, metadata_path = self._paths(layer)
        specs = self._record_specs(layer, records)
        if not archive_path.exists() and not metadata_path.exists():
            if self.read_only:
                raise RuntimeError(f"Reference cache is missing layer {layer}")
            return None
        if not archive_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"Reference cache for layer {layer} is incomplete")

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Reference metadata for layer {layer} is invalid"
            ) from exc
        expected = {
            "format_version": _FORMAT_VERSION,
            "identity_sha256": self.identity_sha256,
            "layer": layer,
            "count": len(records),
            "outputs": specs,
        }
        if not isinstance(metadata, dict) or any(
            metadata.get(key) != value for key, value in expected.items()
        ):
            raise RuntimeError(f"Reference metadata for layer {layer} does not match")
        if metadata.get("archive_sha256") != _sha256_file(archive_path):
            raise RuntimeError(f"Reference archive checksum failed for layer {layer}")
        try:
            outputs = torch.load(archive_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"Reference archive for layer {layer} is invalid"
            ) from exc
        if not isinstance(outputs, list) or len(outputs) != len(records):
            raise RuntimeError(f"Reference archive count failed for layer {layer}")
        for index, (output, spec) in enumerate(zip(outputs, specs)):
            if (
                not isinstance(output, torch.Tensor)
                or output.dtype != torch.bfloat16
                or _tensor_spec(output) != spec
            ):
                raise RuntimeError(
                    f"Reference output {index} for layer {layer} "
                    "has an invalid shape or dtype"
                )
        return outputs

    def store(
        self, layer: int, records: list[dict], outputs: list[torch.Tensor]
    ) -> None:
        if self.read_only:
            raise RuntimeError("Cannot store references through a read-only cache")
        self._verify_identity()
        archive_path, metadata_path = self._paths(layer)
        specs = self._record_specs(layer, records)
        if not isinstance(outputs, list) or len(outputs) != len(records):
            raise ValueError("Reference output count must match the record count")

        stored = []
        for index, (output, spec) in enumerate(zip(outputs, specs)):
            if (
                not isinstance(output, torch.Tensor)
                or output.dtype != torch.bfloat16
                or _tensor_spec(output) != spec
            ):
                raise ValueError(
                    f"Reference output {index} must match its record's BF16 shape"
                )
            stored.append(output.detach().cpu().contiguous())
        if archive_path.exists() or metadata_path.exists():
            raise RuntimeError(f"Reference cache for layer {layer} is immutable")

        _publish_file(archive_path, lambda target: torch.save(stored, target))
        metadata = {
            "format_version": _FORMAT_VERSION,
            "identity_sha256": self.identity_sha256,
            "layer": layer,
            "count": len(records),
            "outputs": specs,
            "archive_sha256": _sha256_file(archive_path),
        }
        _publish_file(metadata_path, lambda target: target.write(_json_bytes(metadata)))
