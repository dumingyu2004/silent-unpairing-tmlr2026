# -*- coding: utf-8 -*-
"""E1 comparator: does VAL-driven selection transfer to an untouched test set?

Consumes JSONs produced by e1_valselect_mwnet.py (frozen ext21 runner + two
post-training holdouts: val1 selection set, official CIFAR-10 test). File naming
mirrors the ext21 runners: {prefix}_seed{s}_baseline.json and
{prefix}_seed{s}_evalcb_p{p}.json.

Estimands (FROZEN in docs/E1_PREREG.md before any GPU run; do not change after
results are read):

  PRIMARY   mean over seeds of  Delta_s = Test(champion_by_Val1) - Test(base),
            champion = argmax val1_acc over {base + all placements} PER SEED
            (ties broken toward the earliest candidate in the order
            [base, placements ascending as given] -> exact ties favor base).
            Seeds where champion == base contribute exactly 0 BY CONSTRUCTION --
            that is part of the estimand, they are NOT excluded.
            Reported with a t-based 95% CI over seeds.

  SECONDARY transfer coefficient rho = Pearson correlation over ALL
            (seed, placement) pairs of (dVal1_j, dTest_j), deltas vs the
            same-seed base, with Fisher-z 95% CI. sigma_Test = mean over seeds
            of the per-seed std (ddof=1) of test deltas. c_m from the exact
            constant table (E[max of m iid N(0,1)]; 1.1630 for m=5). Derived
            check printed next to the primary: predicted mean test harvest
            ~= rho * c_m * sigma_Test.

  Also: selection rate (fraction of seeds with champion != base), per-seed
  champion table, honest mean of test deltas (no selection).

DETERMINISTIC: no RNG anywhere in this analysis (extreme-value constants come
from the exact table, not Monte Carlo; t quantiles from a fixed table).
All deltas reported in percentage points (pt = 100 * accuracy delta).
Only complete seeds (base + every placement present) enter the estimands;
missing files are listed, matching realistic_eval_compare.py conventions.
"""
import argparse
import json
import math
import os

import numpy as np

# E[max of m iid N(0,1)] -- exact-table constants (no Monte Carlo).
C_M = {2: 0.5641895835, 3: 0.8462843753, 4: 1.0293753730, 5: 1.1629644736,
       6: 1.2672063606, 7: 1.3521783756, 8: 1.4236003060, 9: 1.4850131622,
       10: 1.5387527308}

# Two-sided 95% Student-t quantiles t_{0.975, df}. Exact for df 1..30; for
# df > 30 the value at the largest tabulated df <= df is used (conservative:
# t decreases in df, so the CI can only widen).
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
        40: 2.021, 60: 2.000, 120: 1.980}


def t975(df):
    if df in T975:
        return T975[df]
    return T975[max(k for k in T975 if k <= df)] if df > 30 else T975[min(T975)]


def load(path):
    d = json.load(open(path))
    r = d["result"]
    if "val1_acc" not in r or "official_test_acc" not in r:
        raise KeyError(f"{path}: missing val1_acc/official_test_acc -- "
                       f"produced by realistic_eval_mwnet.py instead of e1_valselect_mwnet.py?")
    return float(r["val1_acc"]), float(r["official_test_acc"]), r.get("val1_n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", required=True)
    ap.add_argument("--prefix", default="e1")
    ap.add_argument("--seeds", required=True, help="comma list, e.g. 101,102")
    ap.add_argument("--placements", required=True, help="comma list of eval steps")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    placements = [int(p) for p in args.placements.split(",") if p.strip()]
    L, pre = args.logs_dir, args.prefix

    def P(name):
        return os.path.join(L, f"{pre}_{name}.json")

    missing, val1_ns = [], set()
    rows = {}  # seed -> {"base": (v,t), p: (v,t)}
    for s in seeds:
        f = P(f"seed{s}_baseline")
        if not os.path.exists(f):
            missing.append(f)
            continue
        v, t, n = load(f); val1_ns.add(n)
        row, ok = {"base": (v, t)}, True
        for p in placements:
            f = P(f"seed{s}_evalcb_p{p}")
            if os.path.exists(f):
                v, t, n = load(f); val1_ns.add(n)
                row[p] = (v, t)
            else:
                missing.append(f); ok = False
        if ok:
            rows[s] = row

    present = sorted(rows)
    m = len(placements)
    out = {"seeds_present": present, "placements": placements, "missing": missing}

    print("=" * 74)
    print(f"E1 VAL->TEST BRIDGE COMPARATOR  (prefix={pre})  "
          f"seeds_present={present}  m={m} placements")
    print("=" * 74)
    if len(val1_ns) > 1:
        print(f"!!! WARNING: inconsistent val1_n across runs: {sorted(val1_ns)}")
    if not present:
        print("no complete seeds yet; missing files:")
        for f in missing[:12]:
            print("   ", f)
        json.dump(out, open(args.out + ".tmp", "w"), indent=1)
        os.replace(args.out + ".tmp", args.out)
        return

    # ---- per-seed champion selection (on val1 ONLY; test never consulted) ----
    cand_names = ["base"] + [f"p{p}" for p in placements]
    champs, prim, sel_flags = [], [], []
    dv_pairs, dt_pairs = [], []          # all (seed,placement) deltas, pt
    per_seed_dt = []                     # rows of test deltas for sigma_Test
    print("\n--- per-seed champion table (deltas vs same-seed base, pt) ---")
    for s in present:
        vb, tb = rows[s]["base"]
        dv = [100.0 * (rows[s][p][0] - vb) for p in placements]
        dt = [100.0 * (rows[s][p][1] - tb) for p in placements]
        dv_pairs += dv; dt_pairs += dt; per_seed_dt.append(dt)
        # champion over {base} U placements by val1; ties -> earliest candidate
        # in cand_names order, i.e. exact ties favor base (np.argmax = first max).
        v1s = np.array([vb] + [rows[s][p][0] for p in placements])
        ci = int(np.argmax(v1s))
        champ = cand_names[ci]
        d_s = 0.0 if ci == 0 else 100.0 * (rows[s][placements[ci - 1]][1] - tb)
        champs.append(champ); prim.append(d_s); sel_flags.append(ci != 0)
        print(f"seed{s}: champion={champ:6s}  dTest(champ)={d_s:+6.3f}  "
              f"dVal1=[{' '.join(f'{x:+6.3f}' for x in dv)}]  "
              f"dTest=[{' '.join(f'{x:+6.3f}' for x in dt)}]")

    prim = np.array(prim)
    n = len(prim)
    sel_rate = float(np.mean(sel_flags))
    honest_mean = float(np.mean(dt_pairs))

    # ---- PRIMARY: paired mean + t-based 95% CI over seeds ----
    mean_p = float(prim.mean())
    if n >= 2:
        sd_p = float(prim.std(ddof=1))
        half = t975(n - 1) * sd_p / math.sqrt(n)
        lo, hi = mean_p - half, mean_p + half
    else:
        sd_p, lo, hi = float("nan"), float("nan"), float("nan")

    print("\n--- PRIMARY  Test(champion_by_Val1) - Test(base), over seeds ---")
    print(f"n seeds                               : {n}")
    print(f"mean harvested test delta             : {mean_p:+.3f} pt")
    print(f"sd over seeds                         : {sd_p:.3f} pt")
    print(f"95% CI (t, df={n-1:d})                    : [{lo:+.3f}, {hi:+.3f}] pt")
    print(f"selection rate (champion != base)     : {sel_rate:.2f} "
          f"({int(round(sel_rate*n))}/{n} seeds)")
    print(f"honest mean test delta (no selection) : {honest_mean:+.3f} pt")

    # ---- SECONDARY: transfer coefficient rho over all (seed,placement) pairs ----
    dv_a, dt_a = np.array(dv_pairs), np.array(dt_pairs)
    npairs = dv_a.size
    rho = rlo = rhi = float("nan")
    if npairs >= 3 and dv_a.std() > 0 and dt_a.std() > 0:
        rho = float(np.corrcoef(dv_a, dt_a)[0, 1])
        if npairs > 3 and abs(rho) < 1.0:
            z = math.atanh(rho)
            hz = 1.96 / math.sqrt(npairs - 3)
            rlo, rhi = math.tanh(z - hz), math.tanh(z + hz)
    sigma_test = (float(np.mean([np.std(r, ddof=1) for r in per_seed_dt]))
                  if m >= 2 else float("nan"))
    c_m = C_M.get(m)
    pred = rho * c_m * sigma_test if (c_m is not None and not math.isnan(rho)
                                      and not math.isnan(sigma_test)) else float("nan")

    print("\n--- SECONDARY  transfer coefficient rho (dVal1_j vs dTest_j) ---")
    print(f"pairs (seed,placement)                : {npairs}")
    print(f"rho (Pearson)                         : {rho:+.3f}")
    print(f"95% CI (Fisher z)                     : [{rlo:+.3f}, {rhi:+.3f}]")
    print(f"sigma_Test (mean per-seed sd, pt)     : {sigma_test:.3f}")
    if c_m is None:
        print(f"c_m: m={m} not in exact table {sorted(C_M)} -> derived check skipped")
    else:
        print(f"c_{m} (exact table)                    : {c_m:.4f}")
    print(f"DERIVED predicted mean test harvest   : rho*c_m*sigma_Test = {pred:+.3f} pt"
          f"   (vs PRIMARY measured {mean_p:+.3f} pt)")

    print("\n" + "=" * 74)
    print("E1 PRE-REGISTERED READOUT (ladder frozen in docs/E1_PREREG.md)")
    print(f"  PRIMARY   mean {mean_p:+.3f} pt, 95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  SECONDARY rho {rho:+.3f}, 95% CI [{rlo:+.3f}, {rhi:+.3f}]"
          f"   derived {pred:+.3f} pt")
    print(f"  selection rate {sel_rate:.2f}   honest mean {honest_mean:+.3f} pt")
    # Resolved rung: ladder evaluated top-down exactly as prereg'd. Rung 1:
    # primary CI entirely > 0; rung 2: primary CI contains 0 AND rho CI
    # entirely > 0; rung 3: otherwise (including any all-negative CI, and any
    # undefined CI -- NaN comparisons are False, so they fall through to 3).
    if lo > 0:
        rung, label = 1, ("val-driven selection manufactures held-out test "
                          "gains at this protocol")
    elif lo <= 0 <= hi and rlo > 0:
        rung, label = 2, "partial transfer; underpowered mean at this sigma"
    else:
        rung, label = 3, "no detectable transfer at this protocol; report honestly"
    print(f"  rung: {rung} -- {label}")
    print(f"  analysis set: n={n} complete seeds (df={n-1})")
    if n < 20:
        print(f"  incomplete/missing seeds: {[s for s in seeds if s not in rows]}")
    print("=" * 74)

    out.update({"n_seeds": n, "champions": dict(zip(map(str, present), champs)),
                "primary_mean_pt": mean_p, "primary_sd_pt": sd_p,
                "primary_ci95_pt": [lo, hi],
                "selection_rate": sel_rate, "honest_mean_pt": honest_mean,
                "n_pairs": npairs, "rho": rho, "rho_ci95": [rlo, rhi],
                "sigma_test_pt": sigma_test, "c_m": c_m,
                "predicted_harvest_pt": pred,
                "deltas_val1_pt": [list(r) for r in
                                   np.array(dv_pairs).reshape(n, m).tolist()],
                "deltas_test_pt": [list(r) for r in
                                   np.array(dt_pairs).reshape(n, m).tolist()]})
    json.dump(out, open(args.out + ".tmp", "w"), indent=1)
    os.replace(args.out + ".tmp", args.out)
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
