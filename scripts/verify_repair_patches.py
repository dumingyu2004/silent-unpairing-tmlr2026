"""Verify the curated explanatory repair patches without running training.

Each base is copied to an isolated temporary repository. The script checks and
applies its patch, then reports the actual base and post-patch SHA-256 digests.
It therefore also works on an anonymized release copy, whose base digest is
recorded by PACKAGE_MANIFEST.json rather than by the historical-source record.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPECS = (
    {
        "name": "mwnet_probe_rng_isolation",
        "base": "scripts/moneyshot/moneyshot_mwnet.py",
        "patch": "revision_results/repair_patches/mwnet_probe_rng_isolation.patch",
        "historical_base_sha256": (
            "d8b82d301767f0f2fa4ebf3d954d2ab9781afd18a7f29cd8c4a2e14080d863eb"
        ),
        "historical_original_applied_sha256": (
            "b4513f2c9cffe04863b71e67b7a31ae1329535217f3dcdd7ef0fc25f480fe2e6"
        ),
        "allow_release_base_difference": True,
    },
    {
        "name": "bhn_test_loader_generator",
        "base": "revision_sources/bhn/BHN.py",
        "patch": "revision_results/repair_patches/bhn_test_loader_generator.patch",
        "historical_base_sha256": (
            "e37c95d6b90b32667d48bd4b3f3217c6c460f57efba44f1a2b862fc8e97be8d6"
        ),
        "historical_original_applied_sha256": (
            "b954131157e5193846d247831dce887d5945df4ad343acd82979735dff4e9add"
        ),
        "allow_release_base_difference": False,
    },
    {
        "name": "dividemix_gmm_random_state",
        "base": "revision_sources/dividemix/Train_cifar.py",
        "patch": "revision_results/repair_patches/dividemix_gmm_random_state.patch",
        "historical_base_sha256": (
            "3fa169263c3f3b9cbd6b031017394a66d0d999c7e2f53e5cab01d2d241f5df89"
        ),
        "historical_original_applied_sha256": (
            "ca489f2bbe004eddcb5dd1077b94223d7c72932f73a16450c2475140b40092e2"
        ),
        "allow_release_base_difference": False,
    },
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def verify_one(spec: dict[str, object]) -> dict[str, object]:
    base_rel = Path(str(spec["base"]))
    patch = ROOT / str(spec["patch"])
    base = ROOT / base_rel
    base_digest = sha256(base)
    expected = str(spec["historical_base_sha256"])
    differs = base_digest != expected
    if differs and not bool(spec["allow_release_base_difference"]):
        raise AssertionError(
            f"{spec['name']}: base SHA-256 {base_digest} != expected {expected}"
        )

    with tempfile.TemporaryDirectory(prefix="repair_patch_check_") as raw_tmp:
        tmp = Path(raw_tmp)
        copied_base = tmp / base_rel
        copied_base.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base, copied_base)
        run_git("init", "-q", cwd=tmp)
        run_git("apply", "--check", str(patch), cwd=tmp)
        run_git("apply", str(patch), cwd=tmp)
        applied_digest = sha256(copied_base)

    expected_applied = str(spec["historical_original_applied_sha256"])
    if not differs and applied_digest != expected_applied:
        raise AssertionError(
            f"{spec['name']}: applied SHA-256 {applied_digest} "
            f"!= expected {expected_applied}"
        )

    return {
        "name": spec["name"],
        "base": base_rel.as_posix(),
        "patch": Path(str(spec["patch"])).as_posix(),
        "base_sha256": base_digest,
        "historical_base_sha256": expected,
        "base_differs_from_historical_original": differs,
        "applied_sha256": applied_digest,
        "historical_original_applied_sha256": expected_applied,
        "status": "PASS",
    }


def main() -> None:
    print(json.dumps({"patches": [verify_one(spec) for spec in SPECS]}, indent=2))


if __name__ == "__main__":
    main()
