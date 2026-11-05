"""Post-readout diagnostics; never changes the frozen primary readout or training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def ci(x):
    return np.percentile(x, [2.5, 97.5], axis=0).tolist()


def correlation(cov):
    denom = np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    return [[float(cov[i,j]/denom[i,j]) if denom[i,j] else None
             for j in range(len(cov))] for i in range(len(cov))]


def placement_diagnostics(x, ix):
    n,k = x.shape
    means = x.mean(axis=0)
    raw = np.cov(x, rowvar=False, ddof=1)
    contrasts = x-x.mean(axis=1,keepdims=True)
    ccov = np.cov(contrasts,rowvar=False,ddof=1)
    # Empirical expectation (1/n), while covariance uses 1/(n-1).
    marginal = (n-1)/n * np.trace(raw)/k
    offdiag = (n-1)/n * (raw.sum()-np.trace(raw))/(k*(k-1))
    mean_component = np.sum((means-means.mean())**2)/(k-1)
    lhs = x.var(axis=1,ddof=1).mean()
    rhs = marginal-offdiag+mean_component
    np.testing.assert_allclose(lhs,rhs,rtol=1e-12,atol=1e-18)
    return {'means':means.tolist(), 'mean_ci95':np.asarray(ci(x[ix].mean(axis=1))).T.tolist(),
            'sd':x.std(axis=0,ddof=1).tolist(), 'raw_covariance':raw.tolist(),
            'raw_correlation':correlation(raw),'contrast_covariance':ccov.tolist(),
            'contrast_correlation':correlation(ccov),
            'contrast_warning':'Row centering forces each seed row to sum to zero and the contrast covariance to be singular. These correlations are not an independence test.',
            'interval_scope':'Exploratory marginal percentile mean intervals, not simultaneous; whole seed vectors resampled.',
            'variance_identity':{'mean_within_seed_sample_variance':float(lhs),
                'mean_marginal_variance_component':float(marginal),
                'mean_offdiagonal_covariance_component_subtracted':float(offdiag),
                'between_placement_mean_component':float(mean_component),
                'sum_components':float(rhs),
                'formula':'mean_s S_s^2 = ((n-1)/n)*(trace(C)/k - sum_{i!=j} C_ij/[k(k-1)]) + sum_j (mean_j-grand_mean)^2/(k-1)',
                'ddof':'within-seed sample variance: k-1; across-seed C: n-1; empirical expectation: n'}}


def summary(x, ix):
    selected, unselected = x.max(axis=1), x.mean(axis=1)
    values = {'selected':selected,'unselected':unselected,'harvest':selected-unselected}
    return {key:{'estimate':float(value.mean()),'ci95':ci(value[ix].mean(axis=1)),
                 'per_seed':value.tolist()} for key,value in values.items()}


def build(root):
    root = Path(root)
    paths = {name:root/name for name in ['analysis.json','frozen_calibration_prediction.json','manifest.json']}
    # Operational budget is optional provenance, not an analysis input.
    if (root/'budget.json').exists():
        paths['budget.json'] = root/'budget.json'
    hashes = {name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in paths.items()}
    a = json.loads(paths['analysis.json'].read_text())
    f = json.loads(paths['frozen_calibration_prediction.json'].read_text())
    cal,val = [np.asarray(a[name+'_shared_deltas']) for name in ['calibration','validation']]
    assert cal.shape == val.shape == (16,5)
    assert np.isfinite(cal).all() and np.isfinite(val).all()
    assert a['bootstrap']['repetitions']==20000 and a['bootstrap']['seed']==2026090801
    rng = np.random.default_rng(2026090801)
    cal_ix = rng.integers(0,16,size=(20000,16))
    val_ix = rng.integers(0,16,size=(20000,16))
    sd = cal.std(axis=1,ddof=1)
    h = val.max(axis=1)-val.mean(axis=1)
    coef = f['c5']/f['normal_sample_sd_bias_factor']
    prediction = coef*sd.mean()
    bp = coef*sd[cal_ix].mean(axis=1)
    bh = h[val_ix].mean(axis=1)
    reproduced = {'predicted_mean_harvest':float(prediction),
                  'observed_validation_mean_harvest':float(h.mean()),
                  'signed_prediction_discrepancy':float(h.mean()-prediction),
                  'relative_prediction_discrepancy':float((h.mean()-prediction)/prediction),
                  'calibration_scale_prediction_ci95':ci(bp),
                  'validation_mean_harvest_ci95':ci(bh),'discrepancy_ci95':ci(bh-bp)}
    for key,value in reproduced.items():
        np.testing.assert_allclose(value,a[key],rtol=1e-13,atol=1e-16,err_msg=key)
    np.testing.assert_allclose(prediction,f['predicted_mean_harvest'],rtol=1e-13)
    up = f['c5']*sd.mean()
    ubp = f['c5']*sd[cal_ix].mean(axis=1)
    np.testing.assert_allclose(up,f['uncorrected_prediction_sensitivity'],rtol=1e-13)
    cal_loo = (sd.sum()-sd)/(len(sd)-1)*coef
    val_loo = (h.sum()-h)/(len(h)-1)
    influence = {'stage':'exploratory_post_readout','scope':'All seeds retained. Leave-one-out sensitivity only; no exclusion or refitting on validation.'}
    influence['calibration'] = {'prediction_range':[float(cal_loo.min()),float(cal_loo.max())],
        'per_seed':[{'omitted_seed':seed,'predicted_mean_harvest':float(p),'change_from_full':float(p-prediction)}
                    for seed,p in zip(f['calibration_seeds'],cal_loo)]}
    influence['validation'] = {'observed_mean_harvest_range':[float(val_loo.min()),float(val_loo.max())],
        'discrepancy_range':[float(val_loo.min()-prediction),float(val_loo.max()-prediction)],
        'fixed_prediction':float(prediction),
        'per_seed':[{'omitted_seed':seed,'observed_mean_harvest':float(v),'signed_prediction_discrepancy':float(v-prediction),
                     'change_from_full_observed':float(v-h.mean())} for seed,v in zip(f['validation_seeds'],val_loo)]}
    analyzer = Path(__file__).parent/'revision_nanogpt_analyze.py'
    if not analyzer.exists():
        analyzer = root/'revision_nanogpt_analyze.py'
    result = {'schema_version':1,'units':'fixed validation loss; D=baseline_loss-shared_loss; H=max_placement D-mean_placement D',
        'scope':'Supplement only; frozen corrected primary unchanged. No p-values, significance claim, cross-domain transfer inference, seed selection, or model selection.',
        'placements':a['placements'],'calibration_seeds':f['calibration_seeds'],'validation_seeds':f['validation_seeds'],
        'bootstrap':dict(a['bootstrap'],draw_order='calibration indices then validation indices; same frozen draws',
            joint_resampling='Selected, unselected, H and placement diagnostics use the same whole-seed vectors within each split.'),
        'provenance':{'input_sha256':hashes,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'frozen_analyzer_sha256':hashlib.sha256(analyzer.read_bytes()).hexdigest()},
        'primary_reproduction':{'stage':'frozen_primary_verification','all_checks_passed':True,'tolerance':{'rtol':1e-13,'atol':1e-16},'reproduced':reproduced},
        'uncorrected_sensitivity':{'stage':'prespecified_sensitivity','formula':'c5 * mean_calibration_sample_SD (no normal-SD bias correction)',
            'predicted_mean_harvest':float(up),'observed_validation_mean_harvest':float(h.mean()),
            'signed_prediction_discrepancy':float(h.mean()-up),'relative_prediction_discrepancy':float((h.mean()-up)/up),
            'calibration_scale_prediction_ci95':ci(ubp),'validation_mean_harvest_ci95':ci(bh),'discrepancy_ci95':ci(bh-ubp)},
        'selected_unselected_harvest':{'stage':'exploratory_post_readout','calibration':summary(cal,cal_ix),'validation':summary(val,val_ix)},
        'placement_diagnostics':{'stage':'exploratory_post_readout','calibration':placement_diagnostics(cal,cal_ix),'validation':placement_diagnostics(val,val_ix)},
        'influence':influence}
    assert hashes == {name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in paths.items()}
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',help='Optional output JSON path; refuses to overwrite any existing file.')
    args=parser.parse_args()
    serialized = json.dumps(build(args.root),indent=2,allow_nan=False)
    if args.output:
        with Path(args.output).open('x',encoding='utf-8') as handle:
            handle.write(serialized+'\n')
    print(serialized)
