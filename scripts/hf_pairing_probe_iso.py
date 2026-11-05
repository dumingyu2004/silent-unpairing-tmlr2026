"""Sixth arm for the HF pairing probe: the CUDA probe inside fork_rng.

Extends scripts/hf_pairing_probe.py (frozen; not modified) with one arm:

  probe_cuda_iso : identical to probe_cuda --- every K steps draw a random
                   val batch via the CUDA global stream and evaluate it ---
                   but the entire probe body runs inside
                   torch.random.fork_rng(devices=[model.device]), which
                   saves and restores both the CPU and the CUDA global RNG
                   state on exit.

Prediction (Proposition 1(i)): the isolated arm is bitwise identical to the
never-probed reference --- max |param diff| vs ref exactly 0.0 --- while the
unisolated probe_cuda arm in the same run diverges (positive control).

Run on the same stack as the original a1b receipts (RTX 5080 laptop,
torch 2.11.0+cu128, transformers 5.12.1).
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from hf_pairing_probe import RandomProbeCallback, flat_params  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,  # noqa: E402
                          Trainer, TrainingArguments,
                          enable_full_determinism)


class IsolatedCudaProbeCallback(RandomProbeCallback):
    """The probe_cuda body, wrapped whole in fork_rng (CPU + CUDA saved)."""

    def on_step_end(self, args, state, control, **kw):
        if state.global_step % self.every != 0:
            return
        dev = self.model.device
        devices = [dev] if dev.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            n = self.val_enc["input_ids"].shape[0]
            idx = torch.randint(0, n, (16,), device=dev).cpu()
            was_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                batch = {k: v[idx].to(dev) for k, v in self.val_enc.items()}
                self.model(**batch)
            if was_training:
                self.model.train()


def train_once(arm, args, train_ds, val_enc):
    enable_full_determinism(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    targs = TrainingArguments(
        output_dir=os.path.join(args.workdir, arm),
        max_steps=args.steps,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        seed=args.seed,
        data_seed=args.seed,
        eval_strategy="no",
        eval_steps=args.probe_every,
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        fp16=False,
        dataloader_num_workers=0,
        disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds)
    if arm == "probe_cuda":
        trainer.add_callback(RandomProbeCallback(model, val_enc, args.probe_every,
                                                 device="cuda"))
    elif arm == "probe_cuda_iso":
        trainer.add_callback(IsolatedCudaProbeCallback(model, val_enc,
                                                       args.probe_every,
                                                       device="cuda"))
    trainer.train()
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        n = val_enc["input_ids"].shape[0]
        for i in range(0, n, 64):
            batch = {k: v[i:i+64].to(model.device) for k, v in val_enc.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            correct += int((logits.argmax(-1) == labels).sum())
            total += int(labels.numel())
    return flat_params(model), correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--task", default="rte", choices=["rte", "mrpc"])
    ap.add_argument("--arms", default="ref,repeat,probe_cuda,probe_cuda_iso")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--probe_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workdir", default="logs_hf_pairing_probe")
    ap.add_argument("--out", default="logs_hf_pairing_probe/a1d_iso_rte_s42.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    raw = load_dataset("nyu-mll/glue", args.task)

    train_ds = raw["train"].select(range(1024)).map(
        lambda ex: tok(ex["sentence1"], ex["sentence2"], truncation=True,
                       padding="max_length", max_length=128), batched=True)
    vd = raw["validation"].select(range(256))
    val_enc = tok(list(vd["sentence1"]), list(vd["sentence2"]), truncation=True,
                  padding="max_length", max_length=128, return_tensors="pt")
    val_enc["labels"] = torch.tensor(list(vd["label"]))

    results = {}
    ref_params = None
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        params, acc = train_once(arm, args, train_ds, val_enc)
        if arm == "ref":
            ref_params = params
            diff = 0.0
        else:
            diff = float((params - ref_params).abs().max())
        results[arm] = {"final_val_acc": acc, "max_param_diff_vs_ref": diff}
        print(f"{arm:15s} acc={acc:.4f}  max|dW| vs ref = {diff:.3e}", flush=True)

    stack = {"torch": torch.__version__,
             "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"args": vars(args), "stack": stack, "results": results},
              open(args.out, "w"), indent=2)

    floor_ok = results.get("repeat", {}).get("max_param_diff_vs_ref") == 0.0
    exposed = results.get("probe_cuda", {}).get("max_param_diff_vs_ref", 0.0) > 0.0
    iso_zero = results.get("probe_cuda_iso", {}).get("max_param_diff_vs_ref") == 0.0
    print(f"\nVERDICT: floor {'PASS' if floor_ok else 'FAIL'} | "
          f"cuda probe diverges {'PASS' if exposed else 'FAIL'} | "
          f"fork_rng isolation bitwise-zero {'PASS' if iso_zero else 'FAIL'}")


if __name__ == "__main__":
    main()
