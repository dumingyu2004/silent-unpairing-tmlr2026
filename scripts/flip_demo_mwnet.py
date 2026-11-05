"""Conclusion-flip demo (F2 headline experiment).

Claim to demonstrate: in a bilevel codebase whose training consumes the global
CPU RNG every step (MW-Net rebuilds its meta model per step), ONE extra
innocuous mid-training probe (a single torch.randperm, i.e. one eval-loader
construction) changes the final reported "method vs baseline" gap — the number
a paper would print.

Arms (all under deterministic kernels; CUBLAS_WORKSPACE_CONFIG set):
  mwnet        : faithful MW-Net bilevel training (their ResNet32+VNet, unmodified)
  mwnet_rep    : identical repeat                       -> must equal mwnet (floor)
  mwnet_probe  : identical + ONE torch.randperm(1000) on the global CPU RNG at
                 --probe_step (nothing else; result unused)
  mwnet_probe_isolated : the SAME single probe, but wrapped in
                 torch.random.fork_rng (CPU+CUDA RNG states restored) — the
                 "cure" arm (ext9): if RNG-stream displacement is the entire
                 mechanism, this arm must equal mwnet bitwise
  sgd_baseline : plain SGD training, no reweighting (the paper's comparison arm)

Reported: final val accuracy per arm; gap(mwnet - baseline) vs gap(probe - baseline).
Shortened protocol (10k train subset, ~20 epochs) — the demo needs a stable
final metric, not paper-scale accuracy.

ext9 instrumentation (post-training only — existing arms' trajectories are
untouched): per-arm state_sha256 = sha256 over the fp32 bytes of every
state_dict entry in named order (exact weight-level equality checks across
processes without storing weights); optional --dump_state_dir saves each arm's
final state_dict; probe arms record probe_fired and warn loudly if the probe
never executed; JSON is written atomically (tmp + os.replace).
"""
import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F


def load_cifar10(root, device):
    from torchvision import datasets
    ds = datasets.CIFAR10(root, train=True, download=True)
    x = torch.as_tensor(ds.data).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    return ((x - mean) / std).to(device), torch.as_tensor(ds.targets, dtype=torch.long, device=device)


def state_sha256(module):
    """Fingerprint: sha256 over the fp32 bytes of every state_dict entry, in
    named (registration) order. Gives exact-equality checks across processes
    without storing weights (ext9)."""
    h = hashlib.sha256()
    for name, t in module.state_dict().items():
        h.update(name.encode("utf-8"))
        h.update(t.detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest()


def run_arm(arm, args, xs, ys, splits, device, ResNet32, VNet):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    train_idx, meta_idx, val_idx, ytr, is_noisy = splits

    model = ResNet32(10).to(device)
    opt = torch.optim.SGD(model.params(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    use_meta = arm.startswith("mwnet")
    if use_meta:
        vnet = VNet(1, 100, 1).to(device)
        opt_vnet = torch.optim.Adam(vnet.params(), lr=1e-3, weight_decay=1e-4)

    spe = len(train_idx) // args.batch_size
    gstep = 0
    probe_fired = False
    for epoch in range(args.epochs):
        # GLOBAL CPU RNG for data order — faithful to their DataLoader(shuffle=True)
        # topology; this is the stream a probe pollutes.
        perm = train_idx[torch.randperm(len(train_idx)).to(device)]
        for bi in range(spe):
            gstep += 1
            if arm == "mwnet_probe" and gstep == args.probe_step:
                torch.randperm(1000)  # ONE probe: an eval-loader's shuffle draw. Result unused.
                probe_fired = True
            elif arm == "mwnet_probe_isolated" and gstep == args.probe_step:
                # ext9 cure arm: the SAME probe, but with CPU (and CUDA, when
                # applicable) RNG states forked and restored around it — if
                # stream displacement is the whole mechanism, this arm must
                # equal `mwnet` bitwise.
                with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                    torch.randperm(1000)  # identical probe body. Result unused.
                probe_fired = True
            b = perm[bi * args.batch_size:(bi + 1) * args.batch_size]
            x, y = xs[b], ytr[b]
            if use_meta:
                # faithful MW-Net step: rebuild meta model (consumes global CPU RNG)
                meta_model = ResNet32(10).to(device)
                meta_model.load_state_dict(model.state_dict())
                cost = F.cross_entropy(meta_model(x), y, reduction="none").reshape(-1, 1)
                v_lambda = vnet(cost.data)
                l_f_meta = torch.sum(cost * v_lambda) / len(cost)
                meta_model.zero_grad()
                grads = torch.autograd.grad(l_f_meta, meta_model.params(), create_graph=True)
                meta_model.update_params(lr_inner=0.1, source_params=grads)
                del grads
                mi = torch.randint(0, len(meta_idx), (args.batch_size,)).to(device)  # global CPU stream, like their meta loader
                l_g = F.cross_entropy(meta_model(xs[meta_idx[mi]]), ys[meta_idx[mi]])
                opt_vnet.zero_grad(); l_g.backward(); opt_vnet.step()
                cost_w = F.cross_entropy(model(x), y, reduction="none").reshape(-1, 1)
                with torch.no_grad():
                    w = vnet(cost_w)
                loss = torch.sum(cost_w * w) / len(cost_w)
            else:
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        correct = 0
        for i in range(0, len(val_idx), 1000):
            vb = val_idx[i:i + 1000]
            correct += int((model(xs[vb]).argmax(1) == ys[vb]).sum())

    # --- ext9 instrumentation: everything below runs AFTER training+eval and
    # consumes no RNG, so existing arms' trajectories are untouched. ---
    res = {"final_val_acc": correct / len(val_idx),
           "state_sha256": state_sha256(model),
           "vnet_state_sha256": state_sha256(vnet) if use_meta else None}
    if "probe" in arm:
        res["probe_fired"] = probe_fired
        if not probe_fired:
            msg = (f"!!! WARNING [flip seed{args.seed}] probe arm '{arm}' NEVER FIRED "
                   f"(probe_step={args.probe_step}, total steps={args.epochs * spe}) — "
                   f"it silently degenerated into a plain repeat")
            print(msg, flush=True)
            print(msg, file=sys.stderr, flush=True)
    if args.dump_state_dir:
        os.makedirs(args.dump_state_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.out))[0]
        torch.save({"model": model.state_dict(),
                    "vnet": vnet.state_dict() if use_meta else None},
                   os.path.join(args.dump_state_dir, f"{stem}_{arm}.pt"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data_root", default=str(os.path.join(os.path.dirname(__file__), "..", "vision_data")))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--train_size", type=int, default=10000)
    ap.add_argument("--noise", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--probe_step", type=int, default=500)
    ap.add_argument("--arms", default="mwnet,mwnet_rep,mwnet_probe,sgd_baseline")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dump_state_dir", default=None,
                    help="ext9: optionally save each arm's final state_dict here")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    known = {"mwnet", "mwnet_rep", "mwnet_probe", "mwnet_probe_isolated", "sgd_baseline"}
    unknown = sorted(set(arms) - known)
    if unknown:
        sys.exit(f"unknown arms: {unknown} (known: {sorted(known)})")

    sys.path.insert(0, os.path.abspath(args.repo))
    from resnet import ResNet32, VNet  # their code, unmodified

    device = torch.device(args.device)
    xs, ys = load_cifar10(args.data_root, device)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(xs))
    meta_idx = torch.tensor(perm[:1000], device=device)
    train_idx = torch.tensor(perm[1000:1000 + args.train_size], device=device)
    val_idx = torch.tensor(perm[41000:51000][:5000], device=device)
    ytr = ys.clone()
    nrng = np.random.default_rng(args.seed * 7919)
    fm = torch.tensor(nrng.random(len(train_idx)) < args.noise, device=device)
    offs = torch.tensor(nrng.integers(1, 10, size=len(train_idx)), device=device)
    ytr[train_idx] = torch.where(fm, (ys[train_idx] + offs) % 10, ys[train_idx])
    is_noisy = torch.zeros(len(xs), dtype=torch.bool, device=device)
    splits = (train_idx, meta_idx, val_idx, ytr, is_noisy)

    results = {}
    for arm in arms:
        res = run_arm(arm, args, xs, ys, splits, device, ResNet32, VNet)
        results[arm] = res
        extra = f"  probe_fired={res['probe_fired']}" if "probe_fired" in res else ""
        print(f"[flip seed{args.seed}] {arm:13s} final_val_acc = {res['final_val_acc']:.4f}  "
              f"sha={res['state_sha256'][:16]}{extra}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"  # atomic write (mirrors vision_topo_ema.py's dump)
    json.dump({"args": vars(args), "results": results, "partial": False}, open(tmp, "w"), indent=1)
    os.replace(tmp, args.out)

    def acc_of(a):
        return results[a]["final_val_acc"]

    if all(k in results for k in ("mwnet", "mwnet_rep")):
        sha_eq = results["mwnet"]["state_sha256"] == results["mwnet_rep"]["state_sha256"]
        print(f"\nfloor |mwnet - rep| = {abs(acc_of('mwnet')-acc_of('mwnet_rep'))*100:.3f}pt (must be 0); "
              f"state_sha256 equal: {sha_eq}")
    if all(k in results for k in ("mwnet", "mwnet_probe_isolated")):
        sha_eq = results["mwnet"]["state_sha256"] == results["mwnet_probe_isolated"]["state_sha256"]
        print(f"cure arm: mwnet_probe_isolated state_sha256 == mwnet: {sha_eq} "
              f"(fork_rng must cancel the probe)")
    if all(k in results for k in ("mwnet", "mwnet_probe", "sgd_baseline")):
        g1 = (acc_of("mwnet") - acc_of("sgd_baseline")) * 100
        g2 = (acc_of("mwnet_probe") - acc_of("sgd_baseline")) * 100
        print(f"reported gap (mwnet - baseline)      = {g1:+.2f}pt")
        print(f"reported gap (+1 probe)              = {g2:+.2f}pt")
        print(f"conclusion shift caused by ONE probe = {abs(g1-g2):.2f}pt")


if __name__ == "__main__":
    main()
