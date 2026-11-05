"""Frozen serial nanoGPT revision campaign; standard-library controller."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

PLACEMENTS = [200, 500, 900, 1300, 1700]
CAL_SEEDS = list(range(860901, 860917))
VAL_SEEDS = list(range(860917, 860933))
PILOT_SEED = 860899
TOTAL_LIMIT = 72 * 3600
PILOT_LIMIT = 4 * 3600
MAIN_LIMIT = 60 * 3600
EXPECTED_SOURCE = {
    "train.py": "413bd97b40bb400a2a0d01230e2946f056c263ebac1b7281c2367d964a123085",
    "model.py": "7c01703240dbec5d554527dc666e35b3df8391d0b117fddc07afcf325a21d11c",
    "config/train_shakespeare_char.py": "9b41cdfb2c917259d796a8184d835221a349c23cd18a0a63b46da431d95d9c05",
}
EXPECTED_DATA = {
    "train.bin": "6ec305602a99ac2802745a134e1f5e33e2231b4855525b00b9aebb730ac2626f",
    "val.bin": "d37d30cc0c8327c270d493299c3dca54135f6d5f1c9ef60cda78076e311204b1",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def make_jobs(seeds):
    paths = ["baseline"] + [f"shared_{p}" for p in PLACEMENTS] + [f"isolated_{p}" for p in PLACEMENTS]
    return [{"seed": s, "path": p, "repeat": r} for s in seeds for p in paths for r in (1, 2)]


def job_key(job):
    return f"seed{job['seed']}_{job['path']}_r{job['repeat']}"


def feasibility(slowest_seconds):
    projected = 1.25 * slowest_seconds * len(make_jobs(CAL_SEEDS + VAL_SEEDS)) / 3600
    return {"slowest_full_arm_seconds": slowest_seconds,
            "projected_main_hours": projected, "passes": projected <= 60}


def check_seed(rows, seed):
    failures = []
    for path in [j["path"] for j in make_jobs([seed]) if j["repeat"] == 1]:
        keys = [job_key({"seed": seed, "path": path, "repeat": r}) for r in (1, 2)]
        if not all(k in rows and rows[k].get("complete") for k in keys):
            failures.append(f"missing:{path}")
            continue
        a, b = (rows[k] for k in keys)
        for field in ("endpoint", "repeat_trace_sha256", "fixed_val_loss"):
            if a[field] != b[field]:
                failures.append(f"repeat:{path}:{field}")
        if not a["probe_pure"] or not b["probe_pure"]:
            failures.append(f"purity:{path}")
        if path.startswith("isolated"):
            base = rows.get(job_key({"seed": seed, "path": "baseline", "repeat": 1}))
            if base and a["endpoint"] != base["endpoint"]:
                failures.append(f"isolation:{path}")
            if base and a["fixed_val_loss"] != base["fixed_val_loss"]:
                failures.append(f"isolation_metric:{path}")
            if base and a["common_trace_sha256"] != base["common_trace_sha256"]:
                failures.append(f"isolation_trajectory:{path}")
    return {"seed": seed, "passes": not failures, "failures": failures}


def frozen_prediction(rows):
    scales, harvests = [], []
    for seed in CAL_SEEDS:
        b = rows[job_key({"seed": seed, "path": "baseline", "repeat": 1})]["fixed_val_loss"]
        d = [b - rows[job_key({"seed": seed, "path": f"shared_{p}", "repeat": 1})]["fixed_val_loss"] for p in PLACEMENTS]
        avg = sum(d) / len(d)
        scales.append(math.sqrt(sum((v - avg) ** 2 for v in d) / (len(d) - 1)))
        harvests.append(max(d) - avg)
    c5 = 1.1629644736405196
    bias_factor = math.sqrt(2 / 4) * math.gamma(2.5) / math.gamma(2)
    return {"calibration_seeds": CAL_SEEDS, "validation_seeds": VAL_SEEDS,
            "c5": c5, "normal_sample_sd_bias_factor": bias_factor,
            "calibration_mean_sample_sd": sum(scales) / len(scales),
            "predicted_mean_harvest": c5 * sum(scales) / len(scales) / bias_factor,
            "uncorrected_prediction_sensitivity": c5 * sum(scales) / len(scales),
            "calibration_harvest": harvests,
            "scope": "Frozen calibration-only conditional-Gaussian scale prediction; validation scale is never fitted."}


def run(args):
    root, repo = Path(args.root).resolve(), Path(args.repo).resolve()
    root.mkdir(parents=True, exist_ok=True)
    import fcntl  # Remote execution is Linux; refuse concurrent controllers.
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    scripts = Path(__file__).resolve().parent
    for rel, expected in EXPECTED_SOURCE.items():
        if sha(repo / rel) != expected:
            raise RuntimeError(f"Audited source hash mismatch: {rel}")
    for rel, expected in EXPECTED_DATA.items():
        if sha(repo / "data/shakespeare_char" / rel) != expected:
            raise RuntimeError(f"Historical data hash mismatch: {rel}")
    manifest_path = root / "manifest.json"
    manifest = {"protocol": "nanoGPT-independent-20260908-v1", "source_commit": "3adf61e154c3fe3fca428ad6bc3818b27a3b8291",
                "source_hashes": {f: sha(repo / f) for f in ("train.py", "model.py", "config/train_shakespeare_char.py")},
                "code_hashes": {f: sha(scripts / f) for f in ("revision_nanogpt_campaign.py", "revision_nanogpt_arm.py", "revision_nanogpt_analyze.py")},
                "data_hashes": {f: sha(repo / "data/shakespeare_char" / f) for f in ("train.bin", "val.bin", "meta.pkl")},
                "calibration": CAL_SEEDS, "validation": VAL_SEEDS, "pilot_seed": PILOT_SEED,
                "placements": PLACEMENTS, "updates": 2000, "lr_decay_iters": 2000,
                "main_runs": 704, "repeats": 2, "initial_eval": "common original estimate_loss, eval_iters200",
                "probe": "one original estimate_loss after p completed updates, no checkpoint or metric feedback",
                "fixed_metric": "mean loss over8 val batches, isolated CPU generator seed20260908, batch64/block256",
                "determinism": "strict, FP32/noTF32, mathSDPA, fusedAdamWFalse, compileFalse",
                "budget_hours": {"pilot": 4, "main": 60, "reserve": 8, "total": 72}}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("Manifest mismatch: refusing resumed campaign with changed code/data/protocol")
    else:
        atomic(manifest_path, manifest)
    ledger_path = root / "budget.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"total_seconds": 0, "pilot_seconds": 0, "main_seconds": 0, "attempts": [], "active": None}
    if ledger["active"]:
        active = ledger["active"]
        if active.get("pid"):
            try:
                os.kill(active["pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("Recorded training child is still alive; refusing duplicate launch")
        elapsed = max(0, time.time() - active["started_epoch"])
        ledger["total_seconds"] += elapsed
        ledger[active["stage"] + "_seconds"] += elapsed
        ledger["attempts"].append({**active, "charged_seconds": elapsed, "status": "interrupted_conservative_charge"})
        ledger["active"] = None
        atomic(ledger_path, ledger)
    rows = {}
    for file in (root / "results").glob("*.json"):
        result = json.loads(file.read_text())
        if result.get("complete"):
            rows[file.stem] = result
    stages = [("pilot", [PILOT_SEED])]
    if args.main_after_pilot:
        stages += [("main", CAL_SEEDS), ("main", VAL_SEEDS)]
    for stage, seeds in stages:
        if stage == "pilot" and ledger["attempts"]:
            previous = [a["charged_seconds"] for a in ledger["attempts"] if a["stage"] == "pilot" and a.get("returncode") == 0]
            if previous and not feasibility(max(previous))["passes"]:
                atomic(root / "status.json", {"status": "runtime_gate_failed", **feasibility(max(previous))})
                return 4
        if seeds == VAL_SEEDS:
            prediction_path = root / "frozen_calibration_prediction.json"
            prediction = frozen_prediction(rows)
            if prediction_path.exists() and json.loads(prediction_path.read_text()) != prediction:
                raise RuntimeError("Frozen calibration prediction mismatch")
            atomic(prediction_path, prediction)
        for job in make_jobs(seeds):
            key = job_key(job)
            if key in rows:
                continue
            limit = PILOT_LIMIT if stage == "pilot" else MAIN_LIMIT
            remaining = min(TOTAL_LIMIT - ledger["total_seconds"], limit - ledger[stage + "_seconds"])
            if remaining <= 0:
                atomic(root / "status.json", {"status": "budget_exhausted", "stage": stage, "ledger": ledger})
                return 2
            start = time.time()
            ledger["active"] = {"job": key, "stage": stage, "started_epoch": start}
            atomic(ledger_path, ledger)
            result_file = root / "results" / (key + ".json")
            log_file = root / "logs" / (key + ".log")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", str(scripts / "revision_nanogpt_arm.py"), "--repo", str(repo), "--seed", str(job["seed"]), "--path", job["path"], "--out", str(result_file)]
            atomic(root / "status.json", {"status": "running", "stage": stage, "job": key, "completed_runs": len(rows), "budget_seconds": ledger["total_seconds"]})
            print(json.dumps({"event": "start", "job": key, "remaining_seconds": remaining}), flush=True)
            try:
                with log_file.open("w") as log:
                    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    ledger["active"]["pid"] = proc.pid
                    atomic(ledger_path, ledger)
                    try:
                        rc = proc.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                        raise
            except subprocess.TimeoutExpired:
                rc = 124
            elapsed = time.time() - start
            ledger["total_seconds"] += elapsed
            ledger[stage + "_seconds"] += elapsed
            ledger["attempts"].append({**ledger["active"], "charged_seconds": elapsed, "returncode": rc})
            ledger["active"] = None
            atomic(ledger_path, ledger)
            if rc or not result_file.exists():
                atomic(root / "status.json", {"status": "arm_failed", "job": key, "returncode": rc})
                return 3
            rows[key] = json.loads(result_file.read_text())
            if stage == "pilot" and not feasibility(elapsed)["passes"]:
                atomic(root / "status.json", {"status": "runtime_gate_failed", "job": key, **feasibility(elapsed)})
                return 4
            if job["path"] == "isolated_1700" and job["repeat"] == 2:
                checked = check_seed(rows, job["seed"])
                atomic(root / "checks" / f"seed{job['seed']}.json", checked)
                if not checked["passes"]:
                    atomic(root / "status.json", {"status": "floor_or_isolation_failed", **checked})
                    return 5
        if stage == "pilot":
            longest = max(a["charged_seconds"] for a in ledger["attempts"] if a["stage"] == "pilot")
            floor_gate, runtime_gate = check_seed(rows, PILOT_SEED), feasibility(longest)
            gate = {"floor": floor_gate, "runtime": runtime_gate, "passes": floor_gate["passes"] and runtime_gate["passes"]}
            atomic(root / "pilot_gate.json", gate)
            if not gate["passes"]:
                atomic(root / "status.json", {"status": "pilot_gate_failed", **gate})
                return 6
    atomic(root / "status.json", {"status": "complete" if args.main_after_pilot else "pilot_complete", "completed_runs": len(rows), "budget_seconds": ledger["total_seconds"]})
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--main-after-pilot", action="store_true")
    sys.exit(run(parser.parse_args()))
