#!/usr/bin/env python3
"""Reviewer-requested reanalysis; never modifies historical inputs/comparators.

Run: python scripts/revision_reanalysis.py --draws 100000 --seed 20260908
Only revision_results/ is written. Units are percentage points throughout.
"""
import argparse
import csv
from functools import lru_cache
import hashlib
from itertools import product
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import norm, t


@lru_cache(None)
def normal_cm(m):
    """E max of m independent standard normals, numerical quadrature."""
    return quad(lambda z: m * z * norm.pdf(z) * norm.cdf(z) ** (m - 1),
                -np.inf, np.inf, epsabs=1e-11, epsrel=1e-11)[0]


def normal_c4(m):
    """E sample SD / sigma for m IID normal observations (ddof=1)."""
    return float(np.sqrt(2 / (m - 1)) * np.exp(gammaln(m / 2) - gammaln((m - 1) / 2)))


def finite_correlation(x, y):
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def prediction_metrics(predicted, observed):
    e = predicted - observed
    return dict(predicted_pt=predicted.tolist(), observed_pt=observed.tolist(),
                errors_pt=e.tolist(), mae_pt=float(abs(e).mean()),
                rmse_pt=float(np.sqrt(np.mean(e ** 2))), bias_pt=float(e.mean()),
                pearson_r=finite_correlation(predicted, observed))


def harvest_summary(d):
    d = np.asarray(d, dtype=float)
    if d.ndim != 2 or min(d.shape) < 2 or not np.isfinite(d).all():
        raise ValueError('finite seed-by-placement matrix with >=2 rows/columns required')
    n, m = d.shape
    cm, c4 = normal_cm(m), normal_c4(m)
    h, sd = d.max(1) - d.mean(1), d.std(1, ddof=1)
    if np.any(sd <= 0):
        raise ValueError('ratio bootstrap requires positive within-row SD for every seed')
    idx = np.array(list(product(range(n), repeat=n)))
    bh, bs = h[idx].mean(1), sd[idx].mean(1)
    ratio = bh / (cm * bs)
    pred_mean = np.array([cm * np.delete(sd, i).mean() for i in range(n)])
    pred_flat = np.array([cm * np.delete(d, i, axis=0).std(ddof=1) for i in range(n)])
    # Contrast covariance removes each seed's arbitrary common offset, but
    # five seed rows do not identify/validate an unrestricted covariance law.
    centered = d - d.mean(1, keepdims=True)
    cov = np.cov(d, rowvar=False, ddof=1)
    contrast_cov = np.cov(centered, rowvar=False, ddof=1)
    means = d.mean(0)
    return dict(n_seeds=n, n_placements=m, deltas_pt=d.tolist(),
                normal_cm=cm, normal_c4=c4, harvest_pt=h.tolist(),
                within_row_sd_pt=sd.tolist(), harvest_mean_pt=float(h.mean()),
                scale_mean_pt=float(sd.mean()), predicted_pt=float(cm * sd.mean()),
                predicted_c4_corrected_pt=float(cm * sd.mean() / c4),
                ratio_of_means=float(h.mean() / (cm * sd.mean())),
                ratio_c4_corrected=float(c4 * h.mean() / (cm * sd.mean())),
                bootstrap=dict(method='exhaustive ordinary whole-seed percentile; quantile linear',
                    resamples=n ** n, harvest_ci95_pt=np.quantile(bh, [.025, .975]).tolist(),
                    prediction_ci95_pt=np.quantile(cm * bs, [.025, .975]).tolist(),
                    ratio_ci95=np.quantile(ratio, [.025, .975]).tolist(),
                    ratio_c4_ci95=np.quantile(c4 * ratio, [.025, .975]).tolist()),
                loso=dict(mean_other_within_sd=prediction_metrics(pred_mean, h),
                    mean_other_within_sd_c4=prediction_metrics(pred_mean / c4, h),
                    flatten_other_rows_sd=prediction_metrics(pred_flat, h),
                    own_within_sd_in_sample=prediction_metrics(cm * sd, h)),
                placement_means_pt=means.tolist(), placement_centered_means_pt=(means - means.mean()).tolist(),
                placement_covariance_pt2=cov.tolist(),
                contrast_covariance_pt2=contrast_cov.tolist(),
                placement_covariance_rank=int(np.linalg.matrix_rank(cov)),
                contrast_covariance_rank=int(np.linalg.matrix_rank(contrast_cov)),
                covariance_scope='descriptive sample covariance only; rank <= n_seeds-1; not validated model')


def plugin_simulation(n, m, draws=100000, seed=20260908, seed_scales=None):
    """Simulate full n-seed x m-placement statistic, re-estimating every SD.

    This is a model-reference sensitivity distribution, not new training
    evidence or a goodness-of-fit test. c4 correction is normal-specific.
    """
    rng = np.random.default_rng(seed)
    scales = np.ones(n) if seed_scales is None else np.asarray(seed_scales, dtype=float)
    if scales.shape != (n,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('one positive finite scale per seed required')
    result = {}
    for dist in ['normal', 't3_unit_variance']:
        x = (rng.standard_normal((draws, n, m)) if dist == 'normal'
             else rng.standard_t(3, (draws, n, m)) / np.sqrt(3))
        x *= scales[None, :, None]
        h = (x.max(2) - x.mean(2)).mean(1)
        scale = x.std(2, ddof=1).mean(1)
        ratio = h / (normal_cm(m) * scale)
        c4_ratio = normal_c4(m) * ratio
        result[dist] = dict(draws=draws, seed=seed, n_seeds=n, n_placements=m, seed_scales=scales.tolist(),
            harvest_mean=float(h.mean()), harvest_mcse=float(h.std(ddof=1) / np.sqrt(draws)),
            scale_mean=float(scale.mean()),
            ratio_mean=float(ratio.mean()), ratio_mcse=float(ratio.std(ddof=1) / np.sqrt(draws)),
            ratio_reference95=np.quantile(ratio, [.025, .975]).tolist(),
            normal_c4_corrected_ratio_mean=float(c4_ratio.mean()),
            normal_c4_corrected_ratio_reference95=np.quantile(c4_ratio, [.025, .975]).tolist(),
            ratio_of_population_monte_carlo_means=float(h.mean() / (normal_cm(m) * scale.mean())),
            interpretation='simulation sensitivity only; normal c4 is not an unbiased t3 scale correction')
    return result


def select_champions(val, test):
    """Column order must be baseline, then ascending placements; first max wins."""
    val, test = np.asarray(val), np.asarray(test)
    if val.shape != test.shape or val.ndim != 2 or not np.isfinite(val).all() or not np.isfinite(test).all():
        raise ValueError('matched finite matrices required')
    index = np.argmax(val, axis=1)
    delta = test[np.arange(len(test)), index] - test[:, 0]
    return index, delta


def covariance_decomposition(x, y):
    """Exact empirical uniform-pair law of total covariance (ddof=0).

    The between term contains seed means AND all shared-baseline effects;
    it does not identify an independent causal baseline contribution.
    """
    x, y = np.asarray(x), np.asarray(y)
    xc, yc = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    xm, ym = x.mean(1), y.mean(1)
    within = np.mean(xc * yc)
    between = np.mean((xm - xm.mean()) * (ym - ym.mean()))
    total = np.mean((x - x.mean()) * (y - y.mean()))
    assert np.isclose(total, within + between, atol=1e-12)
    return dict(total_covariance=float(total), within_covariance=float(within),
                between_covariance=float(between),
                total_variance_x=float(x.var()), total_variance_y=float(y.var()),
                within_variance_x=float(np.mean(xc ** 2)), within_variance_y=float(np.mean(yc ** 2)),
                between_variance_x=float(xm.var()), between_variance_y=float(ym.var()),
                pooled_r=finite_correlation(x, y), centered_r=finite_correlation(xc, yc),
                normalization='uniform empirical seed-placement distribution; ddof=0')


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def e1_analysis(root, draws=100000, seed=20260908):
    placements = [400, 1400, 2700, 4000, 5200]
    seeds = list(range(101, 121))
    inputs, val, test = [], [], []
    for s in seeds:
        vr, tr = [], []
        for name in ['baseline'] + [f'evalcb_p{p}' for p in placements]:
            path = root / 'logs_moneyshot' / f'e1_seed{s}_{name}.json'
            rec = load_json(path)
            if rec.get('partial') is not False or rec['args']['seed'] != s:
                raise ValueError(f'incomplete or wrong-seed input: {path}')
            r = rec['result']
            if r['val1_n'] != 4000 or r['total_steps'] != 5400:
                raise ValueError(f'wrong E1 protocol: {path}')
            vr.append(100 * r['val1_acc']); tr.append(100 * r['official_test_acc'])
            inputs.append(path)
        val.append(vr); test.append(tr)
    val, test = np.array(val), np.array(test)
    champs, primary = select_champions(val, test)
    dv, dt = val[:, 1:] - val[:, :1], test[:, 1:] - test[:, :1]
    centered_v, centered_t = dv - dv.mean(1, keepdims=True), dt - dt.mean(1, keepdims=True)
    mean, sd, n = primary.mean(), primary.std(ddof=1), len(seeds)
    half_hist = 2.093 * sd / np.sqrt(n)
    half_exact = t.ppf(.975, n - 1) * sd / np.sqrt(n)
    rho = finite_correlation(dv, dt)
    fisher = np.tanh(np.arctanh(rho) + np.array([-1, 1]) * 1.96 / np.sqrt(dv.size - 3))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(draws, n))
    bp = primary[idx].mean(1)
    # Sufficient sums produce the EXACT recomputed pooled and centered r
    # for each whole-seed draw without creating large 3D temporary arrays.
    def boot_corr(x, y):
        sx, sy = x.sum(1)[idx].sum(1), y.sum(1)[idx].sum(1)
        xx, yy = (x*x).sum(1)[idx].sum(1), (y*y).sum(1)[idx].sum(1)
        xy = (x*y).sum(1)[idx].sum(1)
        count = x.size
        return (xy - sx*sy/count) / np.sqrt((xx - sx*sx/count)*(yy - sy*sy/count))
    pr, cr = boot_corr(dv, dt), boot_corr(centered_v, centered_t)
    if not np.isfinite(pr).all() or not np.isfinite(cr).all():
        raise ValueError('degenerate correlation bootstrap; do not silently drop draws')
    historical = load_json(root / 'logs_moneyshot' / 'e1_verdict.json')
    for new, old in [(dv, historical['deltas_val1_pt']), (dt, historical['deltas_test_pt']),
                     (mean, historical['primary_mean_pt']), ([mean-half_hist, mean+half_hist], historical['primary_ci95_pt'])]:
        np.testing.assert_allclose(new, old, atol=1e-11, rtol=0)
    labels = ['base'] + [f'p{p}' for p in placements]
    return dict(n_seeds=n, seeds=seeds, placements=placements, n_pairs=dv.size,
        deltas_val1_pt=dv.tolist(), deltas_test_pt=dt.tolist(),
        primary_delta_pt=primary.tolist(), champions=[labels[i] for i in champs],
        selected_count=int(np.sum(champs != 0)), baseline_count=int(np.sum(champs == 0)),
        primary_mean_pt=float(mean), primary_sd_pt=float(sd),
        historical_primary_t_ci95_pt=[float(mean-half_hist), float(mean+half_hist)],
        exact_quantile_primary_t_ci95_pt=[float(mean-half_exact), float(mean+half_exact)],
        historical_pooled_r=rho, historical_fisher_ci95=fisher.tolist(),
        historical_rung=2, historical_interpretation='partial transfer; primary mean interval includes zero',
        historical_derived_rho_c5_sigma_pt=float(rho * normal_cm(5) * dt.std(1, ddof=1).mean()),
        derived_formula_scope='historical heuristic only: positive Pearson correlation does not establish IID bivariate Gaussian candidates; baseline plus five is not five exchangeable candidates',
        honest_test_delta_pt=float(dt.mean()), covariance_decomposition=covariance_decomposition(dv, dt),
        reviewer_bootstrap=dict(draws=draws, seed=seed,
            method='whole-seed paired percentile; recompute each correlation; quantile linear',
            primary_ci95_pt=np.quantile(bp, [.025, .975]).tolist(),
            pooled_r_ci95=np.quantile(pr, [.025, .975]).tolist(),
            centered_r_ci95=np.quantile(cr, [.025, .975]).tolist()),
        interval_provenance=dict(appendix_minus0041_plus0290='reproduced frozen comparator t CI, critical value 2.093',
            main_minus0019_plus0279='historical source text reports seed-cluster CI but contains no executable seed/draw provenance; new deterministic bootstrap separately reported; do not relabel as t CI')), inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--draws', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=20260908)
    args = ap.parse_args()
    root = args.root.resolve()
    sources = [root/'logs_flip_demo/prop2_fit.json', root/'logs_moneyshot/ext21_re_verdict5.json', root/'logs_fullrecipe/analysis_summary.json']
    a, b, c = map(load_json, sources)
    matrices = {'shortened': a['deltas_pt'], 'realistic_eval': b['deltas_A_pt'],
                'full_best': c['harvest']['best']['deltas_pt'], 'full_final': c['harvest']['final']['deltas_pt']}
    campaigns = {key: harvest_summary(d) for key, d in matrices.items()}
    for key in campaigns:
        campaigns[key]['simulation'] = plugin_simulation(campaigns[key]['n_seeds'], campaigns[key]['n_placements'], args.draws, args.seed)
        campaigns[key]['simulation_fitted_seed_scales'] = plugin_simulation(
            campaigns[key]['n_seeds'], campaigns[key]['n_placements'], args.draws, args.seed,
            np.asarray(campaigns[key]['within_row_sd_pt']) / campaigns[key]['normal_c4'])
    # Independently reconcile shortened aggregate against all 35 raw arm logs.
    raw = []
    def accuracy(rec):
        return rec['final_val_acc'] if isinstance(rec, dict) else rec
    for i, s in enumerate(a['seeds']):
        path = root/'logs_flip_demo'/f'flip_seed{s}.json'
        base = accuracy(load_json(path)['results']['mwnet']); sources.append(path)
        row = []
        for p in a['placements']:
            path = root/'logs_flip_demo'/f'harvest_seed{s}_p{p}.json'
            row.append(100*(accuracy(load_json(path)['results']['mwnet_probe'])-base)); sources.append(path)
        raw.append(row)
    np.testing.assert_allclose(raw, matrices['shortened'], atol=1e-12, rtol=0)
    e1, e1_paths = e1_analysis(root, args.draws, args.seed)
    sources += e1_paths + [root/'logs_moneyshot/e1_verdict.json', root/'docs/E1_PREREG.md',
                           root/'scripts/moneyshot/e1_compare.py', root/'scripts/prop2_fit.py',
                           root/'figs/make_fig_r1_crossseed.py']
    historical_cm = dict(shortened=a['c_m']['gauss'], realistic_eval=b['c_N'],
                         full_best=c['harvest']['c_m']['gauss'], full_final=c['harvest']['c_m']['gauss'])
    for key, val in historical_cm.items():
        campaigns[key]['historical_cm_monte_carlo'] = val
        campaigns[key]['historical_ratio'] = campaigns[key]['harvest_mean_pt'] / (val * campaigns[key]['scale_mean_pt'])
    report = dict(schema='silent-unpairing-revision-statistics-v1', seed=args.seed, draws=args.draws,
        analysis_code=dict(path='scripts/revision_reanalysis.py', sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        campaigns=campaigns, e1=e1,
        lineage=[dict(path=p.relative_to(root).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size) for p in sources],
        cautions=['Seed bootstrap with three/five seeds is discrete and cannot establish nominal population coverage.',
                  'Paired numerator/denominator resampling is essential; placements are not independent bootstrap units.',
                  'Gaussian c4 corrects normal sample-SD bias only; it does not validate Gaussian/exchangeable residuals.',
                  'Placement means and contrast covariances are descriptive, not independent model validation.',
                  'LOSO variants are historically different and neither is evidence of per-seed scale transport.'])
    output = root/'revision_results'
    output.mkdir(exist_ok=True)
    (output/'statistics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with (output/'harvest_summary.csv').open('w', newline='', encoding='utf-8') as f:
        fields=['campaign','n_seeds','n_placements','harvest_mean_pt','predicted_pt','predicted_c4_corrected_pt','ratio_of_means','ratio_c4_corrected','ratio_lo','ratio_hi','ratio_c4_lo','ratio_c4_hi']
        w=csv.DictWriter(f, fields); w.writeheader()
        for key,s in campaigns.items():
            row={k:s[k] for k in fields[1:8]}; row['campaign']=key
            row.update(zip(fields[8:], s['bootstrap']['ratio_ci95']+s['bootstrap']['ratio_c4_ci95'])); w.writerow(row)
    with (output/'e1_seed_readout.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['seed','champion','selected_test_delta_pt'])
        w.writerows(zip(e1['seeds'],e1['champions'],e1['primary_delta_pt']))
    with (output/'harvest_seed_readout.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['campaign','seed','harvest_pt','sd_ddof1_pt','loso_mean_sd_prediction_pt','loso_flat_prediction_pt'])
        for key,s in campaigns.items():
            for i in range(s['n_seeds']):
                w.writerow([key,i+1,s['harvest_pt'][i],s['within_row_sd_pt'][i],
                            s['loso']['mean_other_within_sd']['predicted_pt'][i],
                            s['loso']['flatten_other_rows_sd']['predicted_pt'][i]])
    with (output/'lineage.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,['path','sha256','bytes']);w.writeheader();w.writerows(report['lineage'])
    print(json.dumps({'campaigns':{k:{j:s[j] for j in ['harvest_mean_pt','predicted_pt','predicted_c4_corrected_pt','ratio_of_means','ratio_c4_corrected','bootstrap','loso']} for k,s in campaigns.items()},'e1':{k:v for k,v in e1.items() if k not in ['deltas_val1_pt','deltas_test_pt']}},indent=2))


if __name__ == '__main__':
    main()
