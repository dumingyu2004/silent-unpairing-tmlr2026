"""Exploratory comparison of two calibration-only constant mean forecasts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def build(root):
    root=Path(root)
    paths={name:root/name for name in ['analysis.json','frozen_calibration_prediction.json','manifest.json']}
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}
    a=json.loads(paths['analysis.json'].read_text())
    f=json.loads(paths['frozen_calibration_prediction.json'].read_text())
    cal=np.asarray(a['calibration_shared_deltas'],dtype=float)
    val=np.asarray(a['validation_shared_deltas'],dtype=float)
    assert cal.shape==val.shape==(16,5)
    assert np.isfinite(cal).all() and np.isfinite(val).all()
    assert len(f['calibration_seeds'])==len(f['validation_seeds'])==16
    assert a['bootstrap']['repetitions']==20000 and a['bootstrap']['seed']==2026090801
    ch=cal.max(axis=1)-cal.mean(axis=1)
    h=val.max(axis=1)-val.mean(axis=1)
    sd=cal.std(axis=1,ddof=1)
    coefficient=f['c5']/f['normal_sample_sd_bias_factor']
    gaussian=float(f['predicted_mean_harvest'])
    empirical=float(ch.mean())
    np.testing.assert_allclose(gaussian,coefficient*sd.mean(),rtol=1e-13,atol=1e-16)
    np.testing.assert_allclose(gaussian,a['predicted_mean_harvest'],rtol=1e-13,atol=1e-16)
    np.testing.assert_allclose(h,a['validation_harvest_per_seed'],rtol=1e-13,atol=1e-16)
    np.testing.assert_allclose(ch,f['calibration_harvest'],rtol=1e-13,atol=1e-16)
    rng=np.random.default_rng(2026090801)
    cal_ix=rng.integers(0,16,(20000,16))
    val_ix=rng.integers(0,16,(20000,16))
    gboot=coefficient*sd[cal_ix].mean(axis=1)
    eboot=ch[cal_ix].mean(axis=1)
    ge=(h-gaussian)**2
    ee=(h-empirical)**2
    delta=ge-ee
    fixed=delta[val_ix].mean(axis=1)
    full=((h[val_ix]-gboot[:,None])**2-(h[val_ix]-eboot[:,None])**2).mean(axis=1)
    predictors={}
    for name,pred,se in [('gaussian',gaussian,ge),('empirical',empirical,ee)]:
        predictors[name]={'prediction':pred,'signed_mean_discrepancy':float(h.mean()-pred),
            'absolute_mean_discrepancy':float(abs(h.mean()-pred)),
            'mse':float(se.mean()),'mae_descriptive':float(abs(h-pred).mean())}
    result={'schema_version':1,'stage':'exploratory_post_readout',
        'scope':'Entire benchmark chosen after readout. Two calibration-only constant mean forecasts, not model selection. Frozen primary unchanged; all seeds retained. No significance, superiority, or transfer claim.',
        'units':{'harvest':'fixed validation loss; H=max_placement D-mean_placement D; D=baseline_loss-shared_loss',
                 'squared_error':'squared loss units'},
        'estimand':'Difference in heldout per-seed mean squared prediction error; negative favors Gaussian descriptively.',
        'mean_discrepancy_scope':'Observed heldout mean minus calibration-only prediction estimates the population-mean discrepancy; the population mean is not known.',
        'mae_scope':'Descriptive only. Squared error is the benchmark score for mean forecasts; MAE is not the primary benchmark score.',
        'calibration_n':16,'validation_n':16,'placements':a['placements'],
        'calibration_seeds':f['calibration_seeds'],'validation_seeds':f['validation_seeds'],
        'observed_validation_mean_harvest':float(h.mean()),
        'predictor_formulas':{'gaussian':'frozen c5 / normal_sample_sd_bias_factor * calibration mean within-seed sample SD (ddof=1)',
                              'empirical':'calibration mean H'},
        'predictors':predictors,
        'contrast':{'mse_gaussian_minus_empirical':float(delta.mean()),
            'fixed_predictor_ci95':np.percentile(fixed,[2.5,97.5]).tolist(),
            'full_procedure_ci95':np.percentile(full,[2.5,97.5]).tolist(),
            'mse_ratio':float(ge.mean()/ee.mean()) if ee.mean()!=0 else None,
            'ratio_definition':'Gaussian MSE / empirical MSE; null if empirical MSE is zero'},
        'bootstrap':{'repetitions':20000,'seed':2026090801,'method':'approximate percentile 95% intervals',
            'unit':'whole seed vector','draw_order':'calibration then validation; same draws as frozen analyzer',
            'fixed_predictor_ci95':'Conditional on original calibration data: paired per-heldout-seed squared-error difference, predictors held fixed.',
            'full_procedure_ci95':'Calibration and validation independently resampled; both predictors recomputed from the same resampled calibration vectors only, then scored on the same resampled validation seeds.'},
        'per_seed':[{'seed':seed,'harvest':float(hh),'gaussian_squared_error':float(g),
                     'empirical_squared_error':float(e),'squared_error_difference':float(d)}
                    for seed,hh,g,e,d in zip(f['validation_seeds'],h,ge,ee,delta)],
        'primary_checks':{'calibration_gaussian_prediction_reproduced':True,'primary_point_prediction_unchanged':True,
                          'harvest_vectors_reproduced':True},
        'provenance':{'input_sha256':hashes,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    assert hashes=={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',help='Optional new JSON path; refuses to overwrite any existing file.')
    args=parser.parse_args()
    serialized=json.dumps(build(args.root),indent=2,allow_nan=False)
    if args.output:
        with Path(args.output).open('x',encoding='utf-8') as handle:
            handle.write(serialized+'\n')
    print(serialized)
