#!/usr/bin/env python3
"""Recover and verify historical MW-Net flip-state checkpoints safely.

The checkpoint files are treated strictly as nested ZIP containers. This module
does not import torch and never deserializes their pickle metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickletools
import re
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STORAGE_COUNT = 161


@dataclass(frozen=True)
class CheckpointSpec:
    file_name: str
    seed: int
    role: str
    json_name: str
    arm: str


EXPECTED_CHECKPOINTS = tuple(
    spec
    for seed in range(1, 6)
    for spec in (
        CheckpointSpec(
            file_name=f"floor_seed{seed}_a_mwnet.pt",
            seed=seed,
            role="floor_a",
            json_name=f"floor_seed{seed}_a.json",
            arm="mwnet",
        ),
        CheckpointSpec(
            file_name=f"floor_seed{seed}_b_mwnet.pt",
            seed=seed,
            role="floor_b",
            json_name=f"floor_seed{seed}_b.json",
            arm="mwnet",
        ),
        CheckpointSpec(
            file_name=f"cure_seed{seed}_mwnet_probe_isolated.pt",
            seed=seed,
            role="isolated_cure",
            json_name=f"cure_seed{seed}.json",
            arm="mwnet_probe_isolated",
        ),
    )
)


@dataclass(frozen=True)
class StorageInventory:
    storage_members: tuple[tuple[int, int, str], ...]
    fingerprint: str
    total_bytes: int
    member_count: int
    archive_member_count: int


@dataclass(frozen=True)
class MetadataStringScan:
    string_operand_count: int
    sensitive_matches: tuple[str, ...]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_zip_name(name: str) -> bool:
    if "\\" in name:
        return False
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and bool(path.parts)


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)


def _find_unique_suffix(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [
        info.filename
        for info in archive.infolist()
        if not info.is_dir() and info.filename.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one archive member ending {suffix!r}, found {len(matches)}")
    name = matches[0]
    info = archive.getinfo(name)
    if not _safe_zip_name(name) or _is_symlink(info):
        raise ValueError(f"unsafe archive member: {name!r}")
    return name


def _looks_sensitive(value: str) -> bool:
    normalized = value.casefold().replace("\\", "/")
    if any(token in normalized for token in ("c:/users/", "/users/", "/home/", "wxid_")):
        return True
    return re.search(r"(?<![a-z0-9])(anonymous-host|anonymous-host|anonymous-host)(?![a-z0-9])", normalized) is not None


def metadata_string_scan(checkpoint: Path) -> MetadataStringScan:
    """Inspect pickle string operands without constructing or executing objects."""

    checkpoint = Path(checkpoint)
    with zipfile.ZipFile(checkpoint) as archive:
        metadata_members = [
            info
            for info in archive.infolist()
            if not info.is_dir() and info.filename.endswith("/data.pkl")
        ]
        if len(metadata_members) != 1:
            raise ValueError(f"expected one opaque data.pkl member in {checkpoint}")
        payload = archive.read(metadata_members[0])
    strings = []
    for opcode, argument, _position in pickletools.genops(payload):
        if opcode.name not in {
            "STRING",
            "BINSTRING",
            "SHORT_BINSTRING",
            "UNICODE",
            "BINUNICODE",
            "SHORT_BINUNICODE",
            "BINUNICODE8",
        }:
            continue
        if isinstance(argument, bytes):
            value = argument.decode("utf-8", errors="replace")
        else:
            value = str(argument)
        strings.append(value)
    return MetadataStringScan(
        string_operand_count=len(strings),
        sensitive_matches=tuple(sorted({value for value in strings if _looks_sensitive(value)})),
    )


def storage_inventory(checkpoint: Path) -> StorageInventory:
    """Hash every raw tensor-storage member without loading pickle metadata."""

    checkpoint = Path(checkpoint)
    with zipfile.ZipFile(checkpoint) as archive:
        infos = archive.infolist()
        if not infos:
            raise ValueError(f"empty checkpoint ZIP: {checkpoint}")
        for info in infos:
            if not _safe_zip_name(info.filename):
                raise ValueError(f"unsafe nested ZIP member in {checkpoint}: {info.filename!r}")
        roots = {PurePosixPath(info.filename).parts[0] for info in infos}
        if len(roots) != 1:
            raise ValueError(f"checkpoint must have one nested ZIP root: {checkpoint}")
        root = next(iter(roots))
        names = {info.filename for info in infos}
        if f"{root}/data.pkl" not in names:
            raise ValueError(f"checkpoint lacks opaque data.pkl metadata: {checkpoint}")

        prefix = f"{root}/data/"
        indexed: list[tuple[int, zipfile.ZipInfo]] = []
        for info in infos:
            if not info.filename.startswith(prefix) or info.is_dir():
                continue
            key = info.filename[len(prefix) :]
            if not key.isdecimal() or "/" in key:
                raise ValueError(f"unexpected tensor-storage member: {info.filename!r}")
            indexed.append((int(key), info))
        indexed.sort(key=lambda item: item[0])
        expected_indices = list(range(len(indexed)))
        actual_indices = [index for index, _ in indexed]
        if actual_indices != expected_indices:
            raise ValueError(f"non-contiguous tensor-storage indices in {checkpoint}")

        combined = hashlib.sha256()
        members = []
        total_bytes = 0
        for index, info in indexed:
            payload = archive.read(info)
            member_hash = _sha256_bytes(payload)
            members.append((index, len(payload), member_hash))
            total_bytes += len(payload)
            combined.update(str(index).encode("ascii"))
            combined.update(b"\0")
            combined.update(len(payload).to_bytes(8, "big"))
            combined.update(payload)

        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"nested ZIP CRC failure in {checkpoint}: {bad_member}")
    return StorageInventory(
        storage_members=tuple(members),
        fingerprint=combined.hexdigest(),
        total_bytes=total_bytes,
        member_count=len(members),
        archive_member_count=len(infos),
    )


def recover_checkpoints(source_archive: Path, output_dir: Path) -> dict[str, Any]:
    """Recover exactly the allowlisted checkpoint payloads from the outer ZIP."""

    source_archive = Path(source_archive)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recovered = 0
    reused = 0
    with zipfile.ZipFile(source_archive) as archive:
        for spec in EXPECTED_CHECKPOINTS:
            suffix = f"/logs_flip_demo/states/{spec.file_name}"
            member = _find_unique_suffix(archive, suffix)
            payload = archive.read(member)  # Reading to EOF verifies this member's CRC.
            destination = output_dir / spec.file_name
            if destination.exists():
                if _sha256_path(destination) != _sha256_bytes(payload):
                    raise ValueError(f"refusing to overwrite conflicting checkpoint: {destination}")
                reused += 1
                continue
            temporary_name = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", prefix=f".{spec.file_name}.", suffix=".tmp", dir=output_dir, delete=False
                ) as temporary:
                    temporary.write(payload)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                    temporary_name = temporary.name
                os.replace(temporary_name, destination)
                temporary_name = None
                recovered += 1
            finally:
                if temporary_name is not None:
                    Path(temporary_name).unlink(missing_ok=True)
    return {"selected": len(EXPECTED_CHECKPOINTS), "recovered": recovered, "reused": reused}


def _record_observation(record_path: Path, spec: CheckpointSpec) -> dict[str, Any]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("partial") is not False:
        raise ValueError(f"record is partial or malformed: {record_path}")
    args = record.get("args", {})
    if args.get("seed") != spec.seed or args.get("arms") != spec.arm:
        raise ValueError(f"record seed/arm mismatch: {record_path}")
    result = record.get("results", {}).get(spec.arm)
    if not isinstance(result, dict):
        raise ValueError(f"record lacks arm result {spec.arm!r}: {record_path}")
    if spec.role == "isolated_cure" and result.get("probe_fired") is not True:
        raise ValueError(f"isolated-cure record lacks probe_fired=true: {record_path}")
    state_hash = result.get("state_sha256")
    vnet_hash = result.get("vnet_state_sha256")
    for label, value in (("state_sha256", state_hash), ("vnet_state_sha256", vnet_hash)):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid {label} in {record_path}")
    return {
        "final_val_acc": result.get("final_val_acc"),
        "state_sha256": state_hash,
        "vnet_state_sha256": vnet_hash,
        "probe_fired": result.get("probe_fired"),
    }


def _archive_payload(archive: zipfile.ZipFile, suffix: str) -> tuple[str, bytes]:
    member = _find_unique_suffix(archive, suffix)
    return member, archive.read(member)


def build_manifest(
    source_archive: Path,
    checkpoint_dir: Path,
    logs_dir: Path,
    source_script: Path,
) -> dict[str, Any]:
    """Build anonymous-safe provenance after checking archive/current correspondence."""

    source_archive = Path(source_archive)
    checkpoint_dir = Path(checkpoint_dir)
    logs_dir = Path(logs_dir)
    source_script = Path(source_script)
    current_source = source_script.read_bytes()
    checkpoints = []
    inventories: dict[str, StorageInventory] = {}
    observations: dict[str, dict[str, Any]] = {}

    with zipfile.ZipFile(source_archive) as archive:
        source_member, archived_source = _archive_payload(
            archive, "/scripts/flip_demo_mwnet.py"
        )
        if archived_source != current_source:
            raise ValueError("current flip-demo source is not byte-identical to the archive member")

        for spec in EXPECTED_CHECKPOINTS:
            checkpoint_path = checkpoint_dir / spec.file_name
            if not checkpoint_path.is_file():
                raise ValueError(f"missing recovered checkpoint: {checkpoint_path}")
            checkpoint_member, archived_checkpoint = _archive_payload(
                archive, f"/logs_flip_demo/states/{spec.file_name}"
            )
            if _sha256_path(checkpoint_path) != _sha256_bytes(archived_checkpoint):
                raise ValueError(f"recovered checkpoint differs from archive: {spec.file_name}")

            record_path = logs_dir / spec.json_name
            current_record = record_path.read_bytes()
            json_member, archived_record = _archive_payload(
                archive, f"/logs_flip_demo/{spec.json_name}"
            )
            if archived_record != current_record:
                raise ValueError(f"current JSON differs from archive: {spec.json_name}")

            inventory = storage_inventory(checkpoint_path)
            if inventory.member_count != EXPECTED_STORAGE_COUNT:
                raise ValueError(
                    f"expected {EXPECTED_STORAGE_COUNT} storage members in {spec.file_name}, "
                    f"found {inventory.member_count}"
                )
            observation = _record_observation(record_path, spec)
            metadata_scan = metadata_string_scan(checkpoint_path)
            if metadata_scan.sensitive_matches:
                raise ValueError(
                    f"sensitive pickle string operands in {spec.file_name}: "
                    f"{metadata_scan.sensitive_matches}"
                )
            inventories[spec.file_name] = inventory
            observations[spec.file_name] = observation
            checkpoints.append(
                {
                    "file": spec.file_name,
                    "seed": spec.seed,
                    "role": spec.role,
                    "bytes": checkpoint_path.stat().st_size,
                    "sha256": _sha256_path(checkpoint_path),
                    "outer_archive_member": checkpoint_member,
                    "nested_zip_member_count": inventory.archive_member_count,
                    "tensor_storage_member_count": inventory.member_count,
                    "tensor_storage_bytes": inventory.total_bytes,
                    "tensor_storage_fingerprint_sha256": inventory.fingerprint,
                    "pickle_string_operand_count": metadata_scan.string_operand_count,
                    "sensitive_pickle_string_matches": [],
                    "associated_json": {
                        "file": f"logs_flip_demo/{spec.json_name}",
                        "sha256": _sha256_bytes(current_record),
                        "outer_archive_member": json_member,
                        "byte_identical_to_archive": True,
                    },
                    "recorded_observation": observation,
                }
            )

    seed_triplets = []
    for seed in range(1, 6):
        specs = [spec for spec in EXPECTED_CHECKPOINTS if spec.seed == seed]
        baseline = inventories[specs[0].file_name]
        for spec in specs[1:]:
            if inventories[spec.file_name].storage_members != baseline.storage_members:
                raise ValueError(f"tensor-storage bytes differ within seed {seed}")
        seed_observations = [observations[spec.file_name] for spec in specs]
        state_hashes = {item["state_sha256"] for item in seed_observations}
        vnet_hashes = {item["vnet_state_sha256"] for item in seed_observations}
        scores = {item["final_val_acc"] for item in seed_observations}
        if len(state_hashes) != 1 or len(vnet_hashes) != 1 or len(scores) != 1:
            raise ValueError(f"JSON model/meta-network observations disagree within seed {seed}")
        seed_triplets.append(
            {
                "seed": seed,
                "files": [spec.file_name for spec in specs],
                "all_tensor_storage_members_byte_equal": True,
                "tensor_storage_member_count": baseline.member_count,
                "tensor_storage_bytes": baseline.total_bytes,
                "tensor_storage_fingerprint_sha256": baseline.fingerprint,
                "recorded_state_sha256": next(iter(state_hashes)),
                "recorded_vnet_state_sha256": next(iter(vnet_hashes)),
                "recorded_final_val_acc": next(iter(scores)),
            }
        )

    return {
        "schema_version": 1,
        "artifact": "historical_mwnet_flip_states",
        "provenance": {
            "source_archive_basename": source_archive.name,
            "source_archive_bytes": source_archive.stat().st_size,
            "source_archive_sha256": _sha256_path(source_archive),
            "selected_checkpoint_members": len(EXPECTED_CHECKPOINTS),
            "recovery_method": "exact allowlist from outer ZIP; no broad extraction",
            "pickle_deserialized": False,
        },
        "scope": {
            "included": "final model and meta-network state_dict tensor storages (parameters and buffers)",
            "excluded": [
                "optimizer state",
                "RNG state",
                "data-cursor state",
                "every-update trajectory state",
            ],
        },
        "source": {
            "file": "scripts/flip_demo_mwnet.py",
            "sha256": _sha256_bytes(current_source),
            "outer_archive_member": source_member,
            "archive_sha256": _sha256_bytes(archived_source),
            "byte_identical_to_archive": True,
            "save_contract": "dictionary containing model.state_dict() and vnet.state_dict() only",
        },
        "checkpoints": checkpoints,
        "seed_triplets": seed_triplets,
    }


def verify_curated_set(
    manifest_path: Path,
    checkpoint_dir: Path,
    logs_dir: Path,
    source_script: Path,
    *,
    verify_original_sidecars: bool = True,
) -> dict[str, Any]:
    """Verify curated files and manifest without needing the private source archive."""

    manifest_path = Path(manifest_path)
    checkpoint_dir = Path(checkpoint_dir)
    logs_dir = Path(logs_dir)
    source_script = Path(source_script)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema")
    if manifest.get("provenance", {}).get("pickle_deserialized") is not False:
        raise ValueError("manifest does not record the no-deserialization boundary")
    if verify_original_sidecars:
        if manifest.get("source", {}).get("sha256") != _sha256_path(source_script):
            raise ValueError("source SHA-256 mismatch")
        if manifest.get("source", {}).get("byte_identical_to_archive") is not True:
            raise ValueError("source/archive identity is not recorded")

    records_by_file = {item.get("file"): item for item in manifest.get("checkpoints", [])}
    expected_names = {spec.file_name for spec in EXPECTED_CHECKPOINTS}
    if set(records_by_file) != expected_names:
        raise ValueError("manifest checkpoint set does not match the fifteen-file allowlist")

    inventories: dict[str, StorageInventory] = {}
    json_verified = 0
    for spec in EXPECTED_CHECKPOINTS:
        record = records_by_file[spec.file_name]
        checkpoint_path = checkpoint_dir / spec.file_name
        if record.get("bytes") != checkpoint_path.stat().st_size:
            raise ValueError(f"checkpoint size mismatch: {spec.file_name}")
        if record.get("sha256") != _sha256_path(checkpoint_path):
            raise ValueError(f"checkpoint SHA-256 mismatch: {spec.file_name}")
        inventory = storage_inventory(checkpoint_path)
        inventories[spec.file_name] = inventory
        if record.get("tensor_storage_member_count") != inventory.member_count:
            raise ValueError(f"storage-member count mismatch: {spec.file_name}")
        if record.get("tensor_storage_bytes") != inventory.total_bytes:
            raise ValueError(f"storage-byte count mismatch: {spec.file_name}")
        if record.get("tensor_storage_fingerprint_sha256") != inventory.fingerprint:
            raise ValueError(f"storage fingerprint mismatch: {spec.file_name}")

        metadata_scan = metadata_string_scan(checkpoint_path)
        if metadata_scan.sensitive_matches:
            raise ValueError(
                f"sensitive pickle string operands in {spec.file_name}: "
                f"{metadata_scan.sensitive_matches}"
            )
        if record.get("pickle_string_operand_count") != metadata_scan.string_operand_count:
            raise ValueError(f"pickle string-operand count mismatch: {spec.file_name}")
        if record.get("sensitive_pickle_string_matches") != []:
            raise ValueError(f"manifest does not record a clean metadata scan: {spec.file_name}")

        if verify_original_sidecars:
            json_meta = record.get("associated_json", {})
            json_path = logs_dir / spec.json_name
            if json_meta.get("sha256") != _sha256_path(json_path):
                raise ValueError(f"associated JSON SHA-256 mismatch: {spec.json_name}")
            observation = _record_observation(json_path, spec)
            if record.get("recorded_observation") != observation:
                raise ValueError(f"associated JSON observation mismatch: {spec.json_name}")
            json_verified += 1

    for seed in range(1, 6):
        specs = [spec for spec in EXPECTED_CHECKPOINTS if spec.seed == seed]
        baseline = inventories[specs[0].file_name].storage_members
        if any(inventories[spec.file_name].storage_members != baseline for spec in specs[1:]):
            raise ValueError(f"tensor-storage bytes differ within seed {seed}")

    member_counts = {inventory.member_count for inventory in inventories.values()}
    if member_counts != {EXPECTED_STORAGE_COUNT}:
        raise ValueError(f"unexpected storage-member counts: {sorted(member_counts)}")
    report = {
        "checkpoint_count": len(inventories),
        "seed_triplet_count": 5,
        "storage_members_per_checkpoint": EXPECTED_STORAGE_COUNT,
        "all_storage_members_equal_within_seed": True,
        "json_records_verified": json_verified,
        "source_verified": verify_original_sidecars,
        "pickle_deserialized": False,
    }
    if not verify_original_sidecars:
        report["sidecar_verification"] = "skipped_for_anonymized_release"
    return report


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=False) + "\n").encode("utf-8")
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recover-from", type=Path)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=ROOT / "artifacts" / "historical_flip_states"
    )
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs_flip_demo")
    parser.add_argument("--source-script", type=Path, default=ROOT / "scripts" / "flip_demo_mwnet.py")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "artifacts" / "historical_flip_states" / "manifest.json"
    )
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument(
        "--checkpoint-only-release",
        action="store_true",
        help=(
            "verify original checkpoint bytes, tensor storages and pickle string metadata "
            "without requiring release-redacted JSON/source copies to match original digests"
        ),
    )
    args = parser.parse_args()

    if args.recover_from is not None:
        recovery = recover_checkpoints(args.recover_from, args.checkpoint_dir)
        if args.write_manifest:
            manifest = build_manifest(
                args.recover_from,
                args.checkpoint_dir,
                args.logs_dir,
                args.source_script,
            )
            _write_json_atomic(args.manifest, manifest)
        elif not args.manifest.is_file():
            raise SystemExit("--write-manifest is required when no manifest exists")
    else:
        recovery = None
        if args.write_manifest:
            raise SystemExit("--write-manifest requires --recover-from for archive provenance")

    verification = verify_curated_set(
        args.manifest,
        args.checkpoint_dir,
        args.logs_dir,
        args.source_script,
        verify_original_sidecars=not args.checkpoint_only_release,
    )
    print(json.dumps({"recovery": recovery, "verification": verification}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
