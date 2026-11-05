"""Instrument the pinned native nanoGPT loop; one process per independent run."""
import argparse
import contextlib
import difflib
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import types


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one pinned source anchor: {old!r}")
    return source.replace(old, new, 1)


def patch_train(source, seed):
    replacements = [
        ("torch.manual_seed(1337 + seed_offset)", f"torch.manual_seed({seed} + seed_offset)"),
        ("torch.backends.cuda.matmul.allow_tf32 = True", "torch.backends.cuda.matmul.allow_tf32 = False"),
        ("torch.backends.cudnn.allow_tf32 = True", "torch.backends.cudnn.allow_tf32 = False"),
        ("ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)", "ctx = nullcontext()  # revision: FP32, no autocast"),
        ("    ix = torch.randint(len(data) - block_size, (batch_size,))", "    ix = torch.randint(len(data) - block_size, (batch_size,))\n    rev.note_draw(split, ix, globals())"),
        ("if iter_num % eval_interval == 0 and master_process:", "if iter_num == 0 and master_process:"),
        ("    optimizer.zero_grad(set_to_none=True)", "    optimizer.zero_grad(set_to_none=True)\n    rev.after_update(globals(), iter_num + 1)"),
        ("if iter_num > max_iters:", "if iter_num >= max_iters:"),
    ]
    for old, new in replacements:
        source = replace_once(source, old, new)
    return source


def main(args):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["OMP_NUM_THREADS"] = "4"
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.cuda.set_per_process_memory_fraction(0.40, 0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    repo = Path(args.repo).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    source = (repo / "train.py").read_text()
    patched = patch_train(source, args.seed)
    model_source = (repo / "model.py").read_text()
    patched_model = replace_once(model_source, "use_fused = fused_available and device_type == 'cuda'", "use_fused = False  # revision: deterministic non-fused AdamW")
    sys.path.insert(0, str(repo))
    module = types.ModuleType("model")
    module.__file__ = str(repo / "model.py")
    sys.modules["model"] = module
    exec(compile(patched_model, module.__file__, "exec"), module.__dict__)

    def digest(value):
        h = hashlib.sha256()
        def visit(v):
            if torch.is_tensor(v):
                t = v.detach().cpu().contiguous()
                h.update(str((str(t.dtype), tuple(t.shape))).encode())
                h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(v, np.ndarray):
                h.update(str((str(v.dtype), v.shape)).encode()); h.update(v.tobytes())
            elif isinstance(v, dict):
                for k in sorted(v, key=lambda k: (type(k).__name__, str(k))):
                    visit(k); visit(v[k])
            elif isinstance(v, (tuple, list)):
                h.update(type(v).__name__.encode())
                for item in v:
                    visit(item)
            else:
                h.update(repr(v).encode())
            h.update(b"\x00")
        visit(value)
        return h.hexdigest()

    class Trace:
        def __init__(self):
            self.completed = 0
            self.in_probe = False
            self.events = []
            self.common = []
            self.pure = True
            self.placements = [200, 500, 900, 1300, 1700]

        def snapshot(self, g):
            torch.cuda.synchronize()
            model = g["raw_model"]
            visible = {"model": model.state_dict(), "optimizer": g["optimizer"].state_dict(),
                       "scaler": g["scaler"].state_dict(), "X": g["X"], "Y": g["Y"],
                       "completed_updates": self.completed, "lr": g["lr"],
                       "module_modes": {n: m.training for n, m in model.named_modules()},
                       "gradients": {n: p.grad for n, p in model.named_parameters()}}
            nonrng = digest(visible)
            rng = digest({"python": random.getstate(), "numpy": np.random.get_state(),
                          "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()})
            return {"weight_sha256": digest(model.state_dict()), "nonrng_sha256": nonrng,
                    "rng_sha256": rng, "full_sha256": digest((nonrng, rng))}

        def note_draw(self, split, ix, g):
            if split == "train" and not self.in_probe and self.completed in self.placements:
                self.events.append({"event": "first_postplacement_training_draw", "after_updates": self.completed,
                                    "indices_sha256": digest(ix), "cpu_rng_sha256": digest(torch.get_rng_state())})

        def after_update(self, g, completed):
            self.completed = completed
            if completed in {1, 2000} | {p + d for p in self.placements for d in (0, 1, 2)}:
                snapshot = self.snapshot(g)
                row = {"event": "after_update", "updates": completed, **snapshot}
                self.events.append(row)
                self.common.append(row)
            if args.path != "baseline" and completed == int(args.path.split("_")[1]):
                before = self.snapshot(g)
                original_mode = g["model"].training
                py_rng, np_rng = random.getstate(), np.random.get_state()
                self.in_probe = True
                try:
                    with torch.random.fork_rng(devices=[0]) if args.path.startswith("isolated") else contextlib.nullcontext():
                        g["estimate_loss"]()
                finally:
                    g["model"].train(original_mode)
                    if args.path.startswith("isolated"):
                        random.setstate(py_rng); np.random.set_state(np_rng)
                    self.in_probe = False
                after = self.snapshot(g)
                self.pure = self.pure and before["nonrng_sha256"] == after["nonrng_sha256"]
                if args.path.startswith("isolated"):
                    self.pure = self.pure and before == after
                self.events.append({"event": "probe", "updates": completed, "before": before, "after": after})
            if completed % 100 == 0:
                print(json.dumps({"completed_updates": completed, "path": args.path}), flush=True)

    trace = Trace()
    start = time.monotonic()
    original_cwd = os.getcwd()
    os.chdir(repo)
    sys.argv = [str(repo / "train.py"), "config/train_shakespeare_char.py", "--max_iters=2000", "--lr_decay_iters=2000",
                "--dtype=float32", "--compile=False", "--wandb_log=False", "--always_save_checkpoint=False",
                "--log_interval=1000000", f"--out_dir={out.parent / 'native_no_checkpoint'}"]
    g = {"__name__": "__main__", "__file__": str(repo / "train.py"), "rev": trace}
    exec(compile(patched, str(repo / "train.py"), "exec"), g)
    if trace.completed != 2000:
        raise RuntimeError("Wrong number of optimizer updates")
    endpoint = trace.snapshot(g)
    # Fixed metric batches: an isolated generator, never native estimate_loss.
    val = np.memmap(repo / "data/shakespeare_char/val.bin", dtype=np.uint16, mode="r")
    fixed_generator = torch.Generator().manual_seed(20260908)
    model = g["raw_model"]
    model.eval()
    losses, indices = [], []
    with torch.no_grad():
        for _ in range(8):
            ix = torch.randint(len(val) - 256, (64,), generator=fixed_generator)
            indices.append(digest(ix))
            x = torch.stack([torch.from_numpy(val[i:i+256].astype(np.int64)) for i in ix]).cuda()
            y = torch.stack([torch.from_numpy(val[i+1:i+257].astype(np.int64)) for i in ix]).cuda()
            losses.append(float(model(x, y)[1]))
    torch.cuda.synchronize()
    result = {"complete": True, "seed": args.seed, "path": args.path, "completed_updates": trace.completed,
              "endpoint": endpoint, "trace": trace.events, "common_trace": trace.common,
              "repeat_trace_sha256": digest(trace.events), "common_trace_sha256": digest(trace.common),
              "probe_pure": trace.pure, "fixed_val_loss": sum(losses) / len(losses),
              "fixed_val_loss_per_batch": losses, "fixed_batch_indices_sha256": indices,
              "wall_seconds": time.monotonic() - start, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
              "train_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "instrumented_source_sha256": hashlib.sha256(patched.encode()).hexdigest(),
              "instrumented_model_sha256": hashlib.sha256(patched_model.encode()).hexdigest(),
              "model_source_diff": "".join(difflib.unified_diff(model_source.splitlines(True), patched_model.splitlines(True), fromfile="model.py", tofile="instrumented_model.py")),
              "source_diff": "".join(difflib.unified_diff(source.splitlines(True), patched.splitlines(True), fromfile="train.py", tofile="instrumented_train.py")),
              "state_hash_scope": "weights+buffers,optimizer,scaler,prefetchedX/Y,completedupdates,lr,modulemodes,grads,Python/NumPy/CPU/CUDA RNG; sparse checkpoints, not every update"}
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False))
    os.replace(tmp, out)
    os.chdir(original_cwd)
    print(json.dumps({"complete": True, "wall_seconds": result["wall_seconds"], "endpoint": endpoint["full_sha256"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--out", required=True)
    main(parser.parse_args())
