# -*- coding: utf-8 -*-
"""E1 Val->Test causal-bridge runner: frozen realistic_eval_mwnet.py + two
POST-TRAINING holdout evaluations. DERIVED (per scripts/moneyshot/README.md
copy-to-new-filename discipline) from the frozen provenance script
realistic_eval_mwnet.py @ sha256
704456a431e6dac2106b1815ddfb4c407eb69e5a430adfdfe8c9e121fdda2243
(the exact bytes that produced logs_moneyshot/ext21_re_*.json).

E1 QUESTION (docs/E1_PREREG.md): ext21 proved that best-of-m placement
selection ON THE REPORTED METRIC manufactures ~c_m*sigma. E1 asks the sharper
bridge question: if the practitioner selects the champion by a VALIDATION set
instead, does the manufactured gain TRANSFER to an untouched official test set?

DELTAS vs the frozen script (nothing else changed):
  1. --val1_size (default 4000): carves val1, the post-training SELECTION
     holdout, as the next val1_size examples of the SAME per-seed permutation
     starting at vstart (the monitoring split is its prefix). At the ext21
     protocol (1k meta + 45k train + 2k monitoring val) val1 = perm[46000:50000]
     = the ENTIRE remaining pool (monitoring 2k U never-touched 2k); a 5k val1
     is impossible under the frozen layout. Pure slicing of the already-drawn
     permutation: consumes NO RNG.
  2. Post-training: val1_acc evaluated first, then the official CIFAR-10 test
     split (10k) evaluated exactly once at the very end (the frozen script's
     existing final eval loop, byte-identical, additionally emitted under the
     explicit name official_test_acc).
  3. JSON gains val1_acc, val1_n, official_test_acc; every frozen field is kept.

INVARIANT (the one that makes E1's arms the SAME experiment as ext21): both
added evaluations run strictly AFTER the training loop has ended -- model.eval(),
no_grad, sequential slicing of pre-loaded GPU tensors, no DataLoader, no
shuffling, zero torch/numpy RNG consumption -- so they cannot desynchronize or
displace anything; the training trajectory and state_sha256 are bit-identical
to the frozen script's at equal CLI args (verified bitwise in the E1 CPU smoke:
frozenref vs e1 sha equality).

Original frozen docstring follows.
---------------------------------------------------------------------------
Realistic-eval witness: the mid-training RNG desync arises from an UNMODIFIED,
STANDARD monitoring EVALUATION pass -- NOTHING artificially inserted.

This is the "you contrived the probe" answer to moneyshot_mwnet.py. That runner
inserts an ARTIFICIAL `torch.randperm(1000)` no-op at --probe_step to *model* an
eval-loader shuffle draw. Here the identical desync arises as the INTRINSIC SIDE
EFFECT of a real thing mainstream trainers actually run every eval_steps: a cheap
held-out monitoring eval that iterates a standard `DataLoader(val, shuffle=True)`
for a bounded number of batches and returns a genuine validation accuracy.

WHY THIS IS THE RNG DRAW (concrete, framework-level, not assumed):
  A `DataLoader(..., shuffle=True)` with the default `generator=None` draws from
  the process-GLOBAL torch generator at iteration time:
    * torch/utils/data/dataloader.py:706-710  _BaseDataLoaderIter.__init__:
        self._base_seed = torch.empty((), dtype=torch.int64)
                              .random_(generator=loader.generator).item()   (1 draw)
    * torch/utils/data/sampler.py:163  RandomSampler.__iter__:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())     (1 draw)
  Measured footprint (this env, torch 2.11): EXACTLY 2 x int64 random_() draws
  == 4 uint32 MT19937 words on the global CPU stream per eval pass. The eval-mode
  forward passes draw 0 (no dropout; BN in eval mode reads running stats).
  This is the SAME global CPU stream training consumes every step (per-step meta
  torch.randint + per-epoch torch.randperm shuffle), so the eval displaces it.

  The shuffle is LOAD-BEARING, not a pointless eval-set permutation: the eval is
  early-stopped after --eval_batches batches (the standard cheap-monitoring
  pattern), so shuffle=True is what makes those batches a RANDOM held-out subset
  rather than always the same head-of-file examples. Delete the eval -> both
  draws vanish (G4). The metric is real: bounded held-out val accuracy, computed,
  returned, and stored -- what a trainer logs / early-stops on.

Precedent (docs/MODERN_STACK_RNG_AUDIT_20260707.md): the audit's `exposed` cases
are periodic in-loop eval callbacks sharing the global torch stream with zero
isolation -- SDXL ImageLogger every batch_frequency=1000 steps (main.py:457-459 ->
diffusion.py:246) and TRL generative eval (grpo_trainer.py:3014). Same structural
mechanism; here rendered on a CIFAR-10/ResNet32 substrate via the canonical
PyTorch shuffled-DataLoader idiom whose global-stream draw is the file:line above.

Arms:
  base       : train, NO mid-training eval callback (clean reference).
  evalcb     : identical training; at --probe_step (and/or every --eval_steps) run
               the REAL monitoring eval, which shares the global stream -> desync.
  evalcb_iso : identical to evalcb but the eval is wrapped in
               torch.random.fork_rng(devices=[cuda]) -> CPU+CUDA RNG saved/restored
               -> MUST be bitwise-identical to base.

Everything else (deterministic kernels, CUBLAS_WORKSPACE_CONFIG=:4096:8,
torch.manual_seed/np seed, state_sha256 over fp32 state_dict bytes, MW-Net real
3-phase LR schedule, held-out CIFAR-10 TEST eval for the final number, atomic JSON
writes, per-step timing, MW-Net official resnet.py import) mirrors
moneyshot_mwnet.py EXACTLY. No data augmentation (all data pre-loaded to GPU),
matching that runner; documented deviation that does not affect the CPU-stream
mechanism.
"""
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_DATA = r"/artifact/local-path"


def load_cifar10(root, device, train):
    from torchvision import datasets
    ds = datasets.CIFAR10(root, train=train, download=False)
    x = torch.as_tensor(ds.data).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    return ((x - mean) / std).to(device), torch.as_tensor(ds.targets, dtype=torch.long, device=device)


def state_sha256(module):
    """sha256 over fp32 bytes of every state_dict entry, in named order:
    exact cross-process weight equality without storing weights (ext9)."""
    h = hashlib.sha256()
    for name, t in module.state_dict().items():
        h.update(name.encode("utf-8"))
        h.update(t.detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest()


def lr_at(epoch, base_lr, milestones):
    """MW-Net policy generalised: base_lr x 0.1 per crossed milestone."""
    factor = 1.0
    for m in milestones:
        if epoch >= m:
            factor *= 0.1
    return base_lr * factor


def monitoring_eval(model, vxc, vyc, args, device):
    """A REAL standard cheap-monitoring eval, written the way a trainer writes it:
    model.eval(); no_grad; iterate a STANDARD DataLoader(val, shuffle=True) for a
    bounded number of batches; compute held-out accuracy; restore model.train().

    The shuffled loader's iterator+sampler seed draws (see module docstring) are the
    INTRINSIC global-stream side effect -- no randperm is inserted here. Returns the
    genuine bounded-subset val accuracy (a real metric a trainer logs)."""
    was_training = model.training
    model.eval()
    loader = DataLoader(TensorDataset(vxc, vyc),
                        batch_size=args.eval_batch_size, shuffle=True)
    correct = total = 0
    with torch.no_grad():
        for bi, (xb, yb) in enumerate(loader):
            if bi >= args.eval_batches:
                break
            xb = xb.to(device, non_blocking=False)
            yb = yb.to(device, non_blocking=False)
            correct += int((model(xb).argmax(1) == yb).sum())
            total += int(yb.numel())
    if was_training:
        model.train()
    return correct / max(total, 1)


def run_arm(arm, args, xs, ys, ytr, meta_idx, train_idx, vxc, vyc, xv1, yv1,
            xte, yte, device, ResNet32, VNet):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    # LR milestones scaled from the official 80/100-of-120 policy.
    milestones = [round(args.epochs * 80 / 120), round(args.epochs * 100 / 120)]

    model = ResNet32(10).to(device)
    opt = torch.optim.SGD(model.params(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    vnet = VNet(1, 100, 1).to(device)
    opt_vnet = torch.optim.Adam(vnet.params(), lr=1e-3, weight_decay=1e-4)

    do_eval = arm in ("evalcb", "evalcb_iso")
    isolate = arm == "evalcb_iso"

    spe = len(train_idx) // args.batch_size
    total_steps = spe * args.epochs
    gstep = 0
    eval_fired = 0
    val_metrics = []
    t_train = 0.0
    t_eval = 0.0
    for epoch in range(args.epochs):
        cur_lr = lr_at(epoch, args.lr, milestones)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        t0 = time.perf_counter()
        # GLOBAL CPU RNG for data order -- faithful to DataLoader(shuffle=True);
        # this is the stream the monitoring eval pollutes.
        perm = train_idx[torch.randperm(len(train_idx)).to(device)]
        for bi in range(spe):
            gstep += 1
            # ---- REAL mid-training monitoring eval (the desync source) ----
            fire = do_eval and (gstep == args.probe_step or
                                (args.eval_steps > 0 and gstep % args.eval_steps == 0))
            if fire:
                te0 = time.perf_counter()
                if isolate:
                    # cure arm: SAME real eval, CPU(+CUDA) RNG forked & restored.
                    # If stream displacement is the whole mechanism -> == base bitwise.
                    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                        vacc = monitoring_eval(model, vxc, vyc, args, device)
                else:
                    # shared stream: the eval's loader seed draws displace training.
                    vacc = monitoring_eval(model, vxc, vyc, args, device)
                val_metrics.append([gstep, vacc])
                eval_fired += 1
                t_eval += time.perf_counter() - te0
            # ---- faithful MW-Net meta step ----
            b = perm[bi * args.batch_size:(bi + 1) * args.batch_size]
            x, y = xs[b], ytr[b]
            meta_model = ResNet32(10).to(device)
            meta_model.load_state_dict(model.state_dict())
            cost = F.cross_entropy(meta_model(x), y, reduction="none").reshape(-1, 1)
            v_lambda = vnet(cost.data)
            l_f_meta = torch.sum(cost * v_lambda) / len(cost)
            meta_model.zero_grad()
            grads = torch.autograd.grad(l_f_meta, meta_model.params(), create_graph=True)
            meta_model.update_params(lr_inner=cur_lr, source_params=grads)
            del grads
            mi = torch.randint(0, len(meta_idx), (args.batch_size,)).to(device)  # global CPU stream
            l_g = F.cross_entropy(meta_model(xs[meta_idx[mi]]), ys[meta_idx[mi]])
            opt_vnet.zero_grad(); l_g.backward(); opt_vnet.step()
            cost_w = F.cross_entropy(model(x), y, reduction="none").reshape(-1, 1)
            with torch.no_grad():
                w = vnet(cost_w)
            loss = torch.sum(cost_w * w) / len(cost_w)
            opt.zero_grad(); loss.backward(); opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_train += (time.perf_counter() - t0) - 0.0  # eval time subtracted below
    t_train -= t_eval  # report TRAINING sec/step; eval overhead reported separately

    model.eval()
    # ---- E1 post-training holdout evaluations (the ONLY behavioral delta vs ----
    # ---- the frozen realistic_eval_mwnet.py) ----------------------------------
    # INVARIANT: both evaluations run strictly AFTER the training loop has ended,
    # in eval mode, under no_grad, over sequentially sliced pre-loaded GPU
    # tensors -- no DataLoader, no shuffling, zero torch/numpy RNG consumption of
    # any kind. Running after training, they cannot desynchronize or displace
    # anything: the treatment arms' training trajectories (and state_sha256) are
    # bit-identical to the frozen script's at equal CLI args.
    # (i) val1: the post-training SELECTION set (champion picked on this, never
    #     on the reported metric).
    with torch.no_grad():
        v1_correct = 0
        for i in range(0, len(xv1), 1000):
            v1_correct += int((model(xv1[i:i + 1000]).argmax(1) == yv1[i:i + 1000]).sum())
    val1_acc = v1_correct / len(xv1)
    # (ii) official CIFAR-10 test split (10k): untouched during training and by
    #      val1 (different torchvision split); evaluated exactly once, here, at
    #      the very end. Loop byte-identical to the frozen script's final eval.
    with torch.no_grad():
        correct = 0
        for i in range(0, len(xte), 1000):
            correct += int((model(xte[i:i + 1000]).argmax(1) == yte[i:i + 1000]).sum())
    test_acc = correct / len(xte)

    res = {"final_test_acc": test_acc,
           "official_test_acc": test_acc,  # explicit E1 name; same single final eval
           "val1_acc": val1_acc,
           "val1_n": len(xv1),
           "state_sha256": state_sha256(model),
           "vnet_state_sha256": state_sha256(vnet),
           "total_steps": total_steps,
           "sec_per_step": t_train / max(total_steps, 1),
           "train_wall_sec": t_train,
           "eval_wall_sec": t_eval,
           "eval_fired": eval_fired,
           "eval_rng_int64_draws": 2 * eval_fired,  # measured: 2 int64 / shuffled-loader pass
           "val_metrics": val_metrics,
           "mechanism": "DataLoader(val,shuffle=True) monitoring eval; global-stream seed draws"}
    if do_eval and eval_fired == 0:
        msg = (f"!!! WARNING [e1 seed{args.seed}] eval arm '{arm}' NEVER FIRED "
               f"(probe_step={args.probe_step}, eval_steps={args.eval_steps}, "
               f"total steps={total_steps}) -- silently degenerated into a plain repeat")
        print(msg, flush=True); print(msg, file=sys.stderr, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to cloned MW-Net repo (imports resnet.py)")
    ap.add_argument("--data_root", default=DEFAULT_DATA)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--train_size", type=int, default=45000)
    ap.add_argument("--val_size", type=int, default=2000, help="held-out clean monitoring split")
    ap.add_argument("--val1_size", type=int, default=4000,
                    help="post-training SELECTION holdout val1 = perm[vstart:vstart+val1_size] "
                         "(monitoring split is its prefix); 4000 = entire remaining pool at "
                         "the ext21 layout (1k meta + 45k train), the maximum possible")
    ap.add_argument("--noise", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--probe_step", type=int, default=500, help="single-fire eval step")
    ap.add_argument("--eval_steps", type=int, default=0, help=">0 -> also eval every K steps (periodic monitoring)")
    ap.add_argument("--eval_batch_size", type=int, default=100)
    ap.add_argument("--eval_batches", type=int, default=4, help="bounded #batches per monitoring eval (cheap)")
    ap.add_argument("--arm", required=True, choices=["base", "evalcb", "evalcb_iso"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.repo))
    from resnet import ResNet32, VNet  # their code, unmodified

    device = torch.device(args.device)
    xs, ys = load_cifar10(args.data_root, device, train=True)
    xte, yte = load_cifar10(args.data_root, device, train=False)

    # Split + label noise: IDENTICAL construction to moneyshot_mwnet.py, PLUS a
    # held-out CLEAN monitoring-val split carved from the remaining train pool.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(xs))
    meta_idx = torch.tensor(perm[:1000], device=device)
    train_idx = torch.tensor(perm[1000:1000 + args.train_size], device=device)
    vstart = 1000 + args.train_size
    vend = vstart + args.val_size
    assert vend <= len(xs), (f"val split [{vstart}:{vend}] overflows train pool {len(xs)}; "
                             f"reduce train_size/val_size")
    val_idx = torch.tensor(perm[vstart:vend], device=device)
    # E1: val1, the post-training SELECTION holdout = the next val1_size examples
    # of the SAME per-seed permutation starting at vstart, i.e. the monitoring
    # split is its prefix and its tail is never touched by anything. At the ext21
    # protocol (train_size=45000, val_size=2000, val1_size=4000) val1 =
    # perm[46000:50000] -- the ENTIRE remaining pool (monitoring 2k U untouched
    # 2k); disjoint from meta/train by construction and from the official test by
    # torchvision split. Pure slicing of the already-drawn permutation: consumes
    # NO RNG, so meta_idx/train_idx/label noise/training stay bit-identical to
    # the frozen realistic_eval_mwnet.py.
    v1end = vstart + args.val1_size
    assert v1end <= len(xs), (f"val1 split [{vstart}:{v1end}] overflows train pool {len(xs)}; "
                              f"reduce train_size/val1_size")
    val1_idx = torch.tensor(perm[vstart:v1end], device=device)
    ytr = ys.clone()
    nrng = np.random.default_rng(args.seed * 7919)
    fm = torch.tensor(nrng.random(len(train_idx)) < args.noise, device=device)
    offs = torch.tensor(nrng.integers(1, 10, size=len(train_idx)), device=device)
    ytr[train_idx] = torch.where(fm, (ys[train_idx] + offs) % 10, ys[train_idx])

    # CPU copies for the standard DataLoader (real trainers yield CPU batches).
    vxc = xs[val_idx].detach().cpu().contiguous()
    vyc = ys[val_idx].detach().cpu().contiguous()  # clean labels for monitoring
    # E1: GPU copies for the post-training val1 eval (sequential slicing, no
    # loader, clean labels -- noise is only ever applied to train_idx rows of ytr).
    xv1 = xs[val1_idx]
    yv1 = ys[val1_idx]

    t_wall0 = time.perf_counter()
    res = run_arm(args.arm, args, xs, ys, ytr, meta_idx, train_idx, vxc, vyc,
                  xv1, yv1, xte, yte, device, ResNet32, VNet)
    res["wall_sec"] = time.perf_counter() - t_wall0

    vm = f"  val@fire={res['val_metrics'][0][1]:.3f}" if res["val_metrics"] else ""
    print(f"[e1 seed{args.seed}] {args.arm:12s} p{args.probe_step:<6d} "
          f"val1_acc = {res['val1_acc']:.4f}  official_test_acc = {res['official_test_acc']:.4f}  "
          f"sha={res['state_sha256'][:16]}  "
          f"{res['sec_per_step']*1000:.1f} ms/step  fired={res['eval_fired']} "
          f"draws={res['eval_rng_int64_draws']}  wall={res['wall_sec']:.0f}s{vm}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    json.dump({"args": vars(args), "arm": args.arm, "result": res, "partial": False},
              open(tmp, "w"), indent=1)
    os.replace(tmp, args.out)


if __name__ == "__main__":
    main()
