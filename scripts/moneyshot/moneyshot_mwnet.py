# -*- coding: utf-8 -*-
"""Money-shot runner: shared-stream-probe HARVEST at a believable/long protocol.

Faithful SCALE-UP of scripts/flip_demo_mwnet.py (the validated shortened-protocol
harvest demo). The ONLY changes vs that demo are:
  (1) parametrised, larger protocol: full/large train subset + more epochs;
  (2) a real MW-Net LR schedule (nesterov SGD, x0.1 drops at ~2/3 and ~5/6 of
      training, mirroring the official epoch-80/100-of-120 policy) applied to
      BOTH the outer optimiser and the inner meta-lr (as MW-Net.py does);
  (3) evaluation on the held-out CIFAR-10 TEST set (10k) rather than a train
      slice -- faithful to MW-Net.py and avoids train/val overlap at large
      train_size;
  (4) single-arm invocation + per-step wall-clock timing recorded to JSON, so
      the overnight chain is crash-safe/resumable and self-times.

The RNG-contamination MECHANISM is IDENTICAL to the validated demo:
  * training consumes the global CPU RNG every step (per-step meta-batch
    torch.randint + per-epoch torch.randperm shuffle), exactly as MW-Net.py's
    per-step meta-model rebuild + DataLoader(shuffle=True) topology does;
  * the probe is ONE torch.randperm(1000) on the global CPU stream (an
    eval-loader shuffle draw), inserted at --probe_step; result unused;
  * the isolated arm wraps the SAME probe in torch.random.fork_rng so CPU+CUDA
    states are restored -> must equal the un-probed baseline bitwise.

Deterministic kernels + CUBLAS_WORKSPACE_CONFIG give a bitwise repeat floor
(state_sha256 over fp32 state_dict bytes). No augmentation (all data pre-loaded
to GPU as tensors), matching the demo; documented deviation from the official
augmented pipeline that does NOT affect the CPU-stream harvest mechanism.
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


def run_arm(arm, args, xs, ys, ytr, meta_idx, train_idx, xte, yte, device, ResNet32, VNet):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    # LR milestones scaled from the official 80/100-of-120 policy.
    milestones = [round(args.epochs * 80 / 120), round(args.epochs * 100 / 120)]

    model = ResNet32(10).to(device)
    opt = torch.optim.SGD(model.params(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    use_meta = arm.startswith("mwnet")
    if use_meta:
        vnet = VNet(1, 100, 1).to(device)
        opt_vnet = torch.optim.Adam(vnet.params(), lr=1e-3, weight_decay=1e-4)

    spe = len(train_idx) // args.batch_size
    total_steps = spe * args.epochs
    gstep = 0
    probe_fired = False
    t_train = 0.0
    for epoch in range(args.epochs):
        cur_lr = lr_at(epoch, args.lr, milestones)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        t0 = time.perf_counter()
        # GLOBAL CPU RNG for data order -- faithful to DataLoader(shuffle=True);
        # this is the stream a probe pollutes.
        perm = train_idx[torch.randperm(len(train_idx)).to(device)]
        for bi in range(spe):
            gstep += 1
            if arm == "mwnet_probe" and gstep == args.probe_step:
                torch.randperm(1000)  # ONE probe: an eval-loader shuffle draw. Unused.
                probe_fired = True
            elif arm == "mwnet_probe_isolated" and gstep == args.probe_step:
                # cure arm: SAME probe, CPU(+CUDA) RNG forked & restored. If stream
                # displacement is the whole mechanism, must equal `mwnet` bitwise.
                with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                    torch.randperm(1000)
                probe_fired = True
            b = perm[bi * args.batch_size:(bi + 1) * args.batch_size]
            x, y = xs[b], ytr[b]
            if use_meta:
                # faithful MW-Net step: rebuild meta model, 2nd-order meta grad.
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
                # meta set is CLEAN held-out -> use original labels ys (like the demo)
                l_g = F.cross_entropy(meta_model(xs[meta_idx[mi]]), ys[meta_idx[mi]])
                opt_vnet.zero_grad(); l_g.backward(); opt_vnet.step()
                cost_w = F.cross_entropy(model(x), y, reduction="none").reshape(-1, 1)
                with torch.no_grad():
                    w = vnet(cost_w)
                loss = torch.sum(cost_w * w) / len(cost_w)
            else:
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_train += time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        correct = 0
        for i in range(0, len(xte), 1000):
            correct += int((model(xte[i:i + 1000]).argmax(1) == yte[i:i + 1000]).sum())
    test_acc = correct / len(xte)

    res = {"final_test_acc": test_acc,
           "state_sha256": state_sha256(model),
           "vnet_state_sha256": state_sha256(vnet) if use_meta else None,
           "total_steps": total_steps,
           "sec_per_step": t_train / max(total_steps, 1),
           "train_wall_sec": t_train}
    if "probe" in arm:
        res["probe_fired"] = probe_fired
        if not probe_fired:
            msg = (f"!!! WARNING [ms seed{args.seed}] probe arm '{arm}' NEVER FIRED "
                   f"(probe_step={args.probe_step}, total steps={total_steps}) -- "
                   f"silently degenerated into a plain repeat")
            print(msg, flush=True); print(msg, file=sys.stderr, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to cloned MW-Net repo (imports resnet.py)")
    ap.add_argument("--data_root", default=DEFAULT_DATA)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--train_size", type=int, default=48000)
    ap.add_argument("--noise", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--probe_step", type=int, default=500)
    ap.add_argument("--arm", required=True,
                    choices=["mwnet", "mwnet_rep", "mwnet_probe", "mwnet_probe_isolated", "sgd_baseline"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.repo))
    from resnet import ResNet32, VNet  # their code, unmodified

    device = torch.device(args.device)
    xs, ys = load_cifar10(args.data_root, device, train=True)
    xte, yte = load_cifar10(args.data_root, device, train=False)

    # Split + label noise: IDENTICAL construction to flip_demo_mwnet.py.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(xs))
    meta_idx = torch.tensor(perm[:1000], device=device)
    train_idx = torch.tensor(perm[1000:1000 + args.train_size], device=device)
    ytr = ys.clone()
    nrng = np.random.default_rng(args.seed * 7919)
    fm = torch.tensor(nrng.random(len(train_idx)) < args.noise, device=device)
    offs = torch.tensor(nrng.integers(1, 10, size=len(train_idx)), device=device)
    ytr[train_idx] = torch.where(fm, (ys[train_idx] + offs) % 10, ys[train_idx])

    t_wall0 = time.perf_counter()
    res = run_arm(args.arm, args, xs, ys, ytr, meta_idx, train_idx, xte, yte, device, ResNet32, VNet)
    res["wall_sec"] = time.perf_counter() - t_wall0

    extra = f"  probe_fired={res['probe_fired']}" if "probe_fired" in res else ""
    print(f"[ms seed{args.seed}] {args.arm:20s} p{args.probe_step:<6d} "
          f"final_test_acc = {res['final_test_acc']:.4f}  sha={res['state_sha256'][:16]}  "
          f"{res['sec_per_step']*1000:.1f} ms/step  wall={res['wall_sec']:.0f}s{extra}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    json.dump({"args": vars(args), "arm": args.arm, "result": res, "partial": False},
              open(tmp, "w"), indent=1)
    os.replace(tmp, args.out)


if __name__ == "__main__":
    main()
