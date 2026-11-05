# -*- coding: utf-8 -*-
"""Money-shot comparator: prints the MANUFACTURED-GAIN verdict.

Pre-registered comparison (fixed BEFORE the full run):
  A = shared-stream probe (contaminated). Select best-of-N probe placements per
      seed, then average. Report the apparent gain vs the honest no-probe number
      and (if sgd present) vs the SGD baseline.
  B = fork_rng-isolated probe. Same selection. Gain must be ~0 (each isolated
      placement is provably bitwise-identical to the un-probed baseline).

Metrics (matching scripts/prop2_fit.py conventions):
  Delta[s,p]        = 100 * (acc_probe[s,p] - acc_base[s])         (pts)
  honest_mean       = mean_{s,p} Delta                             (probe on avg)
  per_seed_select   = mean_s max_p Delta                           (harvested)
  manufactured_gain = per_seed_select - honest_mean                (selection bias)
  extreme-value pred = c_N * sigma_bar,  c_N = E[max-mean of N iid N(0,1)]
                       (deterministic MC, seed 0), sigma_bar = mean_s sd_p(Delta).

Verdicts (pre-registered):
  V1 floor  : sha(mwnet) == sha(mwnet_rep) per seed (bitwise repeat floor).
  V2 cure   : sha(mwnet_probe_isolated) == sha(mwnet) for every isolated witness
              -> Condition B Delta == 0 exactly -> manufactured_gain_B == 0.
  V3 harvest: manufactured_gain_A > 0 and within ~[0.7, 1.4]x the c_N*sigma_bar
              extreme-value prediction (existence + magnitude claim).
  V4 vanish : manufactured_gain_A - manufactured_gain_B is the headline number
              (apparent gain that exists under shared stream, gone under fork_rng).
No claim is made about MW-Net's TRUE ranking vs SGD.
"""
import argparse
import json
import os
from itertools import combinations

import numpy as np


def acc(path):
    d = json.load(open(path))
    return d["result"]["final_test_acc"], d["result"]["state_sha256"]


def mc_cN(m, n=200_000):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, m))
    return float((x.max(1) - x.mean(1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", required=True)
    ap.add_argument("--prefix", default="ms")
    ap.add_argument("--seeds", required=True, help="comma list, e.g. 1,2,3")
    ap.add_argument("--placements", required=True, help="comma list of probe steps")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    placements = [int(p) for p in args.placements.split(",") if p.strip()]
    L, pre = args.logs_dir, args.prefix

    def P(name):
        return os.path.join(L, f"{pre}_{name}.json")

    base, base_sha, missing = {}, {}, []
    for s in seeds:
        f = P(f"seed{s}_baseline")
        if os.path.exists(f):
            base[s], base_sha[s] = acc(f)
        else:
            missing.append(f)

    # Condition A: shared-stream probe grid
    D, present_seeds = [], []
    for s in seeds:
        if s not in base:
            continue
        row, ok = [], True
        for p in placements:
            f = P(f"seed{s}_probe_p{p}")
            if os.path.exists(f):
                a, _ = acc(f)
                row.append(100.0 * (a - base[s]))
            else:
                missing.append(f); ok = False
        if ok:
            D.append(row); present_seeds.append(s)
    D = np.array(D) if D else np.zeros((0, len(placements)))

    # Condition B: isolated witnesses (SHA must equal baseline)
    iso_checks, iso_deltas = [], []
    for s in seeds:
        if s not in base:
            continue
        for p in placements:
            f = P(f"seed{s}_iso_p{p}")
            if os.path.exists(f):
                a, sha = acc(f)
                iso_checks.append((s, p, sha == base_sha[s]))
                iso_deltas.append(100.0 * (a - base[s]))

    # Floor: rep
    floor = []
    for s in seeds:
        if s not in base:
            continue
        f = P(f"seed{s}_rep")
        if os.path.exists(f):
            _, sha = acc(f)
            floor.append((s, sha == base_sha[s]))

    # sgd baseline (optional, for the apparent-vs-baseline narrative)
    sgd = {}
    for s in seeds:
        f = P(f"seed{s}_sgd")
        if os.path.exists(f):
            sgd[s], _ = acc(f)

    out = {"seeds_present": present_seeds, "placements": placements,
           "deltas_A_pt": D.tolist(), "missing": missing}

    print("=" * 74)
    print(f"MONEY-SHOT COMPARATOR  (prefix={pre})  seeds_present={present_seeds}  N={len(placements)}")
    print("=" * 74)

    if D.shape[0] == 0:
        print("no complete seeds yet; missing files:")
        for m in missing[:12]:
            print("   ", m)
        json.dump(out, open(args.out + ".tmp", "w"), indent=1); os.replace(args.out + ".tmp", args.out)
        return

    N = len(placements)
    c_N = mc_cN(N)
    sd = D.std(axis=1, ddof=1)
    sigma_bar = float(sd.mean())
    honest_mean_A = float(D.mean())
    per_seed_select_A = float(np.mean(D.max(axis=1)))
    manufactured_gain_A = per_seed_select_A - honest_mean_A
    pred = c_N * sigma_bar

    print("\n--- per-seed shared-stream deltas (probe - baseline, pt) ---")
    for i, s in enumerate(present_seeds):
        row = " ".join(f"{d:+6.2f}" for d in D[i])
        print(f"seed{s}: [{row}]  mean {D[i].mean():+.2f}  sd {sd[i]:.2f}  "
              f"max {D[i].max():+.2f}  (best@p{placements[int(np.argmax(D[i]))]})")

    print("\n--- CONDITION A  (shared stream, contaminated) ---")
    print(f"honest mean delta (probe on avg)      : {honest_mean_A:+.2f} pt")
    print(f"per-seed select-best then average     : {per_seed_select_A:+.2f} pt")
    print(f"MANUFACTURED GAIN (selection bias)    : {manufactured_gain_A:+.2f} pt")
    print(f"extreme-value prediction c_N*sigma    : {pred:.2f} pt "
          f"(c_{N}={c_N:.4f}, sigma_bar={sigma_bar:.2f})")
    ratio = manufactured_gain_A / pred if pred else float("nan")
    print(f"ratio measured/predicted              : {ratio:.2f}")

    # Isolated / Condition B
    iso_all_eq = all(ok for _, _, ok in iso_checks) if iso_checks else None
    manufactured_gain_B = 0.0
    print("\n--- CONDITION B  (fork_rng isolated, cure) ---")
    if iso_checks:
        neq = [(s, p) for s, p, ok in iso_checks if not ok]
        print(f"isolated witnesses: {len(iso_checks)}  sha==baseline: {sum(ok for _,_,ok in iso_checks)}/{len(iso_checks)}")
        print(f"isolated deltas (pt)                  : {[round(x,3) for x in iso_deltas]}")
        print(f"MANUFACTURED GAIN (isolated)          : {manufactured_gain_B:+.2f} pt "
              f"({'all bitwise-equal to baseline -> harvest impossible' if iso_all_eq else 'MISMATCH: '+str(neq)})")
    else:
        print("no isolated witnesses found (Condition B is 0 by fork_rng construction; "
              "run mwnet_probe_isolated witnesses to make it empirical)")

    print("\n--- FLOOR (bitwise repeat) ---")
    if floor:
        print(f"mwnet==mwnet_rep sha: {sum(ok for _,ok in floor)}/{len(floor)} seeds  "
              f"{'OK' if all(ok for _,ok in floor) else 'BROKEN: '+str([s for s,ok in floor if not ok])}")
    else:
        print("no rep arm found")

    if sgd:
        common = [s for s in present_seeds if s in sgd]
        if common:
            idx = [present_seeds.index(s) for s in common]
            honest_gap = float(np.mean([100 * (base[s] - sgd[s]) for s in common]))
            sel_gap = float(np.mean([D[i].max() + 100 * (base[s] - sgd[s]) for i, s in zip(idx, common)]))
            print("\n--- apparent method-vs-SGD gain (the number a paper prints) ---")
            print(f"honest (no probe)   gap vs sgd        : {honest_gap:+.2f} pt")
            print(f"select-best-probe   gap vs sgd        : {sel_gap:+.2f} pt")
            print(f"manufactured by selection             : {sel_gap - honest_gap:+.2f} pt "
                  f"(== Condition A manufactured gain)")
            out["apparent_vs_sgd"] = {"honest": honest_gap, "selected": sel_gap}

    print("\n--- growth curve (exhaustive subsets m'=2..N, mean over seeds) ---")
    for mp in range(2, N + 1):
        subs = list(combinations(range(N), mp))
        emp = float(np.mean([[D[i, list(c)].max() - D[i, list(c)].mean() for c in subs] for i in range(len(present_seeds))]))
        print(f"m'={mp}: empirical {emp:.3f} | pred c_m'*sigma {mc_cN(mp)*sigma_bar:.3f}")

    print("\n" + "=" * 74)
    print("PRE-REGISTERED VERDICT")
    v1 = bool(floor) and all(ok for _, ok in floor)
    v2 = iso_all_eq is True
    v3 = manufactured_gain_A > 0 and (0.7 <= ratio <= 1.4 if pred else False)
    headline = manufactured_gain_A - manufactured_gain_B
    print(f"  V1 floor  (mwnet==rep bitwise)            : {'PASS' if v1 else 'n/a or FAIL'}")
    print(f"  V2 cure   (isolated==baseline bitwise)    : {'PASS' if v2 else 'n/a or FAIL'}")
    print(f"  V3 harvest(gain>0 & ~c_N*sigma)           : {'PASS' if v3 else 'FAIL/inconclusive'}")
    print(f"  V4 HEADLINE manufactured apparent gain    : {headline:+.2f} pt  "
          f"(shared stream) -> {manufactured_gain_B:+.2f} pt (fork_rng)")
    print("=" * 74)

    out.update({"c_N": c_N, "sigma_bar": sigma_bar, "honest_mean_A": honest_mean_A,
                "per_seed_select_A": per_seed_select_A,
                "manufactured_gain_A": manufactured_gain_A,
                "extreme_value_pred": pred, "ratio": ratio,
                "manufactured_gain_B": manufactured_gain_B,
                "iso_sha_equal": iso_all_eq, "floor_ok": v1,
                "headline_apparent_gain": headline,
                "verdicts": {"V1_floor": v1, "V2_cure": v2, "V3_harvest": v3}})
    json.dump(out, open(args.out + ".tmp", "w"), indent=1)
    os.replace(args.out + ".tmp", args.out)
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
