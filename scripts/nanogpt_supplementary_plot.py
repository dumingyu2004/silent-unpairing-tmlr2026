"""Per-seed display; never changes the frozen inference or selects seeds."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def build_figure(data, frozen):
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.25), layout='constrained')
    x = np.arange(1, 17)
    for ax, key, title, color in zip(axes[:2],
            ['calibration_shared_deltas', 'validation_shared_deltas'],
            ['(a) Calibration: descriptive', '(b) Held-out: frozen mean prediction'],
            ['#0072B2', '#009E73']):
        grid = np.asarray(data[key])
        h = grid.max(axis=1) - grid.mean(axis=1)
        ax.scatter(x, h*1000, color=color, s=23, zorder=3, label='Per-seed harvest')
        ax.axhline(frozen['predicted_mean_harvest']*1000, color='#D55E00', lw=1.3,
                   label='Frozen corrected prediction')
        ax.axhline(frozen['uncorrected_prediction_sensitivity']*1000, color='#555555',
                   lw=1, ls='--', label='Uncorrected sensitivity')
        ax.set(title=title, xlabel='Seed index (prespecified order)',
               ylabel=r'Harvest $H$ (loss $\times 10^{-3}$)', ylim=(0, 8), xlim=(0, 17))
        ax.set_xticks([1, 4, 8, 12, 16])
        ax.grid(axis='y', alpha=.18)
    axes[0].legend(loc='upper right', fontsize=6.7, frameon=False)
    grid = np.asarray(data['validation_shared_deltas'])
    mean, best = grid.mean(axis=1)*1000, grid.max(axis=1)*1000
    ax = axes[2]
    for seed, lo, hi in zip(x, mean, best):
        ax.plot([seed, seed], [lo, hi], color='#AAAAAA', lw=.8, zorder=1)
    ax.scatter(x, mean, s=23, facecolors='white', edgecolors='#0072B2',
               label='Placement mean', zorder=3)
    ax.scatter(x, best, s=23, color='#D55E00', marker='^', label='Selected maximum', zorder=3)
    ax.axhline(0, color='black', lw=.8)
    ax.set(title='(c) Held-out reported score: exploratory', xlabel='Seed index (prespecified order)',
           ylabel=r'$D=L_{base}-L_{shared}$ (loss $\times 10^{-3}$)', xlim=(0, 17))
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.legend(loc='lower right', fontsize=7, frameon=False)
    ax.grid(axis='y', alpha=.18)
    return fig


def main():
    folder = ROOT/'revision_results/nanogpt_final'
    data = json.loads((folder/'analysis.json').read_text())
    frozen = json.loads((folder/'frozen_calibration_prediction.json').read_text())
    fig = build_figure(data, frozen)
    fig.savefig(folder/'nanogpt_per_seed.png', dpi=180)
    fig.savefig(folder/'nanogpt_per_seed.pdf')
    fig.savefig(ROOT/'paper_tmlr/figs/nanogpt_per_seed.pdf')
    fig.savefig(ROOT/'paper/figs/nanogpt_per_seed.pdf')
    plt.close(fig)
    print('Saved per-seed nanoGPT plots; original analysis unchanged')


if __name__ == '__main__':
    main()
