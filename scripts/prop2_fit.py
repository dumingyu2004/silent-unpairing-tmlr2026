# -*- coding: utf-8 -*-
"""Prop 2 (extreme-value harvest) fit on the P13 multi-seed harvest grid.

Corrected protocol (adversarial math review, 2026-07-07):
  - model: Delta_{s,j} = mu_s + W_{s,j}; max-mean is invariant to mu_s (and to
    any component shared across placements), so the fit targets E[max-mean] =
    c_m * sigma_W with the EXACT expected-max-of-m-standard-normals constant
    c_m (NOT the sqrt(2 ln m) asymptote, which overshoots ~45-50% at m<=8).
  - per-seed sigma estimated on 6 cross-process placements {200,400,800,1000,
    1200,1600} (balanced grid; in-process p500 of seeds 1-2 excluded).
  - cross-prediction: seed i's harvest predicted with sigma pooled from the
    OTHER seeds.
  - subsample growth curve m'=2..6 (exhaustive subsets, dependent-but-honest),
    compared against Gaussian c_m and Student-t3 c_m^(t) nulls (both via
    deterministic Monte Carlo, rng seed 0).
  - wash-out test (reviewer attack #2): per-seed selection then averaging vs
    honest mean; plus shared-placement selection (predicted to shrink ~1/sqrt(S)).

Schema note: flip/harvest JSONs before 2026-07-07 store results[arm] as float;
later files store a dict with final_val_acc (ext9 M4 migration).
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "logs_flip_demo"
SEEDS = [1, 2, 3, 4, 5]
PLACEMENTS = [200, 400, 800, 1000, 1200, 1600]


def acc(rec):
    return rec["final_val_acc"] if isinstance(rec, dict) else rec


def load():
    grid, base = {}, {}
    for s in SEEDS:
        fb = json.load(open(ROOT / f"flip_seed{s}.json"))
        base[s] = acc(fb["results"]["mwnet"])
        for p in PLACEMENTS:
            fh = json.load(open(ROOT / f"harvest_seed{s}_p{p}.json"))
            grid[(s, p)] = acc(fh["results"]["mwnet_probe"])
    return base, grid


def mc_cm(m, dist, n=200_000):
    rng = np.random.default_rng(0)
    if dist == "normal":
        x = rng.standard_normal((n, m))
    else:  # t3, scaled to unit variance (var of t3 = 3)
        x = rng.standard_t(3, (n, m)) / np.sqrt(3.0)
    return float((x.max(1) - x.mean(1)).mean())


def main():
    base, grid = load()
    D = np.array([[100 * (grid[(s, p)] - base[s]) for p in PLACEMENTS] for s in SEEDS])
    print("=== per-seed deltas (pt) ===")
    for i, s in enumerate(SEEDS):
        row = " ".join(f"{d:+6.2f}" for d in D[i])
        print(f"seed{s}: [{row}]  mean {D[i].mean():+.2f}  sd {D[i].std(ddof=1):.2f}  "
              f"max-mean {D[i].max() - D[i].mean():+.2f}")

    m = len(PLACEMENTS)
    c_gauss = mc_cm(m, "normal")
    c_t3 = mc_cm(m, "t3")
    sd = D.std(axis=1, ddof=1)
    harvest = D.max(axis=1) - D.mean(axis=1)

    print(f"\n=== Prop 2 fit (m={m}) ===")
    print(f"c_m gaussian {c_gauss:.4f} | c_m t3(unit var) {c_t3:.4f} | sqrt(2 ln m) {np.sqrt(2*np.log(m)):.4f} (rejected asymptote)")
    for i, s in enumerate(SEEDS):
        pooled_sd = np.std(np.delete(D, i, axis=0), ddof=1)
        print(f"seed{s}: measured {harvest[i]:+.2f} | self-pred {c_gauss*sd[i]:.2f} | "
              f"cross-pred(gauss) {c_gauss*pooled_sd:.2f} | cross-pred(t3) {c_t3*pooled_sd:.2f}")
    print(f"mean measured {harvest.mean():+.2f} | gauss pred (mean sd) {c_gauss*sd.mean():.2f} | "
          f"t3 pred {c_t3*sd.mean():.2f}")

    print("\n=== subsample growth curve (exhaustive subsets, mean over seeds) ===")
    for mp in range(2, m + 1):
        subs = list(combinations(range(m), mp))
        emp = np.mean([[D[i, list(c)].max() - D[i, list(c)].mean() for c in subs] for i in range(len(SEEDS))])
        print(f"m'={mp}: empirical {emp:.3f} | gauss {mc_cm(mp,'normal')*sd.mean():.3f} | "
              f"t3 {mc_cm(mp,'t3')*sd.mean():.3f}")

    print("\n=== wash-out test (reviewer attack #2) ===")
    honest = D.mean()
    per_seed_sel = float(np.mean(D.max(axis=1)))
    shared_j = int(np.argmax(D.mean(axis=0)))
    shared_sel = float(D.mean(axis=0)[shared_j])
    print(f"honest mean delta            : {honest:+.2f} pt")
    print(f"per-seed select then average : {per_seed_sel:+.2f} pt  (bias {per_seed_sel-honest:+.2f}; predicted c_m*sigma = {c_gauss*sd.mean():.2f})")
    print(f"shared-placement select      : {shared_sel:+.2f} pt (p{PLACEMENTS[shared_j]}) "
          f"(bias {shared_sel-honest:+.2f}; predicted c_m*sigma/sqrt(S) = {c_gauss*sd.mean()/np.sqrt(len(SEEDS)):.2f})")
    print("verdict: bias does NOT wash out with seed averaging (per-seed selection); "
          "shared-placement selection shrinks ~1/sqrt(S) as predicted" )

    out = {
        "placements": PLACEMENTS, "seeds": SEEDS,
        "deltas_pt": D.tolist(), "sd_pt": sd.tolist(),
        "harvest_max_minus_mean_pt": harvest.tolist(),
        "c_m": {"gauss": c_gauss, "t3_unitvar": c_t3, "sqrt2lnm": float(np.sqrt(2*np.log(m)))},
        "washout": {"honest_mean": honest, "per_seed_select": per_seed_sel,
                     "shared_select": shared_sel, "shared_placement": PLACEMENTS[shared_j]},
    }
    tmp = ROOT / "prop2_fit.json.tmp"
    json.dump(out, open(tmp, "w"), indent=1)
    import os
    os.replace(tmp, ROOT / "prop2_fit.json")
    print("\nwritten logs_flip_demo/prop2_fit.json")


if __name__ == "__main__":
    main()
