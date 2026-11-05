"""Generate manuscript numbers and figures from the canonical reanalysis only."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def main():
    r=json.loads((ROOT/'revision_results/statistics.json').read_text())
    e=r['e1']; b=e['reviewer_bootstrap']; s=r['campaigns']['shortened']
    values={'RevEOneMean':e['primary_mean_pt'],
            'RevEOneTLow':e['historical_primary_t_ci95_pt'][0],
            'RevEOneTHigh':e['historical_primary_t_ci95_pt'][1],
            'RevEOneBLow':b['primary_ci95_pt'][0],
            'RevEOneBHigh':b['primary_ci95_pt'][1],
            'RevEOneR':e['covariance_decomposition']['centered_r'],
            'RevHarvest':s['harvest_mean_pt'],
            'RevPrediction':s['predicted_pt'],
            'RevCorrectedPrediction':s['predicted_c4_corrected_pt']}
    text='% GENERATED from revision_results/statistics.json.\n'
    text+=''.join('\\newcommand{\\'+k+'}{'+f'{v:.4f}'+'}\n' for k,v in values.items())
    (ROOT/'revision_results/statistics_macros.tex').write_text(text,encoding='utf-8')
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(3.05,2.8),layout='constrained')
    keys=['shortened','realistic_eval','full_best','full_final']
    for i,k in enumerate(keys):
        a=r['campaigns'][k]
        for shift,col,field,ci in [(-.12,'#0072B2','ratio_of_means','ratio_ci95'),(.12,'#D55E00','ratio_c4_corrected','ratio_c4_ci95')]:
            mid=a[field]; lo,hi=a['bootstrap'][ci]
            ax.errorbar(mid,i+shift,xerr=[[mid-lo],[hi-mid]],fmt='o',color=col,capsize=3,
                        label=('Plug-in' if shift<0 else 'Normal c4') if i==0 else None)
    ax.axvline(1,color='gray',ls='--',lw=1)
    ax.set_yticks(range(4),['Shortened\n5 seeds','Real eval.\n5 seeds','Full: best\n3 seeds','Full: final\nsame 3 seeds'])
    ax.invert_yaxis()
    ax.set_xlabel('Observed / fitted harvest')
    ax.legend(frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.25,1.00),fontsize=7)
    for folder in ['figs','paper_tmlr/figs']:
        fig.savefig(ROOT/folder/'fig_r2_ratios_ci.pdf')
    fig.savefig(ROOT/'revision_results/ratios.png',dpi=140)
    plt.close(fig)
    # The historical LOSO figure is retained as a historical readout;
    # canonical numbers and both implementations live in the reanalysis.
    print('Generated canonical TeX macros and ratio figure')

if __name__=='__main__':
    main()
