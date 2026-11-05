"""Rebuild source audit and evidence ledgers without running external training code.

The audit judgments below are manual source-reading results. File hashes bind the
citations to recovered snapshots; they are NOT training-state hashes.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# name, reviewed verdict, source anchors, scope/correction
AUDIT = [
('l2rw_tf','exposed','mnist/mnist_train.py:270-272,372-373,396-397','Evaluation advances the same training iterator; structural cursor channel, not a pure RNG probe.'),
('l2rw_torch','exposed','data_loader.py:105-111','Unconditional shuffle=True override shares global CPU sampling. Model reinitialization is an additional method-side consumer, not assumed state-pure.'),
('mwnet','exposed','MW-Net.py:196-218','Meta iterator creation/recreation and model initialization consume shared CPU randomness; witness repair is separately assessed.'),
('dividemix','exposed','Train_cifar.py:87,186-187','Unpinned GMM initialization shares global NumPy with mixup. Pinning changes initialization as well as stream position.'),
('coteaching','exposed','model.py:45,54','Functional dropout omits training flag, so model.eval does not disable it; evaluation and training share device RNG.'),
('maml','exposed','miniimagenet_train.py:67,78-86','Shuffled mid-training test loader and train loader share global CPU generator.'),
('nanogpt','exposed','train.py:116-123,216-222,264,303','Evaluation get_batch and training get_batch share CPU randint; no repaired arm implied by static verdict.'),
('rent','exposed','RENT/library/method_utils.py:122-123','On-device multinomial is a method-specific training consumer. Exposure does not establish a no-op intervention or incorrect published result.'),
('bhn','exposed','BHN.py:202-210,756,781,810,1112-1120','Unshuffled test iterator construction consumes CPU base seed; next training iterator/augmentation shares that state.'),
('topofilter','exposed','train.py:103-110,236-269,348-350','Feature extraction reuses shuffled trainloader; selection additionally changes training data.'),
('crust','exposed','robust_cifar_train.py:176-184,237,320-335','Validation iterator has no dedicated generator; shuffled training shares CPU state. Coreset changes are structural method effects.'),
('selcl','exposed','CIFAR/train_Sel-CL.py:189-198;utils/utils_noise.py:350-359','WeightedRandomSampler shares CPU RNG with pair_selection torch.rand allocation.'),
('mlc','exposed','CIFAR/data_helper_cifar.py:74-79','Training and validation shuffled loaders have no generator isolation.'),
('promix','exposed','Train_promix.py:208,419-427,494,514-515;dataloader_cifarn.py:273-303','Selection iterates nonpersistent eval loader with shuffle=False and no generator; base seed shares CPU RNG with randperm/training.'),
('unicon','exposed','Train_cifar.py:148,291-305,387,402-413;dataloader_cifar.py:314-344','JSD evaluation iterator consumes CPU base seed, despite shuffle=False; training randperm/loaders share CPU generator.'),
('disc','exposed','algorithms/DISC.py:308-310,406-420;main.py:183-189','Per-epoch evaluation loader construction/iteration shares CPU stream with training data loading. Mixup is an additional method consumer.'),
('ddpm','exposed','denoising_diffusion_pytorch/denoising_diffusion_pytorch.py:678-686,791,838,1078-1084','Periodic EMA sampling and training diffusion noise use shared device RNG.'),
('timm','exposed','timm/data/loader.py:237,346-363;train.py:1095-1110,1392-1415','Nondistributed shuffled path: first eval iterator construction advances global CPU RNG. Persistent workers reuse iterators: do not claim one fresh base seed every epoch.'),
('cleanrl','exposed','cleanrl/qdagger_dqn_atari_impalacnn.py:277,328,362;cleanrl_utils/evals/dqn_eval.py:28','Periodic evaluation and subsequent QDagger training share Python random. Specific QDagger path, not all CleanRL algorithms.'),
('litgpt','exposed','litgpt/finetune/lora.py:58,343-348,465-490;litgpt/lora.py:80-84;litgpt/generate/base.py:31-36','Default LoRA dropout 0.05 and example generation share torch device stream. Zero-dropout or disabled generation requires separate classification.'),
('infobatch','exposed','research/cifar_example.py:51-54,70-77,145-150,174-176;research/infobatch.py:79-88;examples/cifar_example.py:106-108,138-143,271','CORRECTED: counter-reseed guards NumPy pruning only. Unshuffled eval iterator takes global torch CPU base seed; train transforms use CPU randomness with num_workers=0. No dedicated train-loader generator in this path.'),
('dcgan','guarded','dcgan/main.py:133-145,211,240,264-271','RNG channel only: visualization reuses fixed_noise. Train-mode BatchNorm buffers are mutated by netG(fixed_noise), so this is NOT a state-pure or full-pairing certificate.'),
('sb3','guarded','stable_baselines3/common/callbacks.py:372-403,465-471;stable_baselines3/common/evaluation.py:92-99','Default deterministic policy evaluation with separate environment using its own RNG. Stochastic actions, custom global-RNG environment or feature extractor invalidate this limited guard; separate object alone is insufficient.'),
('trl','conditional','trl/trainer/grpo_trainer.py:1352-1368,1542-1614;trl/trainer/utils.py:712-719,831','Transformers sampling backend exposed when eval enabled. vLLM engine topology needs explicit scope; train-only shuffle is not an eval draw. No blanket immunity for arbitrary engines/reward functions.'),
('olmo','guarded','src/olmo_core/data/data_loader.py:544-564,670-688;src/olmo_core/train/train_module/transformer/train_module.py:497,649-654','Audited evaluator disables training-only forward draws and uses private data-order PCG generator. Torch generator at544 is MOCK batch generation, not generic eval DataLoader construction. No universal k=0 claim.'),
('sdxl','exposed','main.py:457-459;sgm/models/diffusion.py:246-251;sgm/modules/diffusionmodules/loss.py:72-80','Image logging samples with global torch and stochastic denoising; training uses shared torch noise stream. Trigger must fire mid-training.'),
('mmselfsup','guarded','configs/selfsup/_base_/datasets/imagenet_deepcluster.py:33-55;mmselfsup/datasets/samplers/deepcluster_sampler.py:47-48;mmselfsup/engine/hooks/deepcluster_hook.py:73-123','Shipped DeepCluster path: private sampler, worker-seeded augmentation, persistent workers. First extract iterator can draw global CPU base seed; shuffle=False is NOT zero consumption. Head initialization/label updates are method effects; no state-purity claim.'),
('diffusers','conditional','examples/text_to_image/train_text_to_image.py:161-174,972-982,1104','Audited validation pipeline uses private generator if seed supplied, global generator otherwise. Applies to this sampling channel, not arbitrary pipeline/scheduler/custom callback.'),
]

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def build_audit():
    manifest = {x['name']:x for x in json.loads((ROOT/'revision_sources/recovery_manifest.json').read_text())}
    rows=[]
    for name, verdict, anchors, note in AUDIT:
        citations=[]
        for anchor in anchors.split(';'):
            filename, lines=anchor.split(':')
            path=ROOT/'revision_sources'/name/filename
            if not path.is_file():
                index = json.loads((ROOT/'revision_sources/external_source_index.json').read_text())
                item = index[path.relative_to(ROOT).as_posix()]
                citations.append(item['citation'])
                continue
            line_count=len(path.read_text(encoding='utf-8').splitlines())
            for span in lines.split(','):
                if max(map(int,span.split('-'))) > line_count:
                    raise ValueError(anchor)
            citations.append({'path':path.relative_to(ROOT).as_posix(),'lines':lines,'source_file_sha256':digest(path)})
        rows.append({'name':name,'pinned_commit':manifest[name]['commit'],'before':manifest[name]['original_verdict'],'after':verdict,'verification':'static_source_reading','citations':citations,'conditions':note})
    dependency=ROOT/'revision_sources/mmengine/mmengine/dataset/utils.py'
    return {'schema':1,'scope':'Deliberately selected pinned code paths; RNG/cursor exposure only. Not population prevalence, full state purity, dynamic divergence, or published-result validity.', 'before_counts':dict(Counter(x['before'] for x in rows)), 'after_counts':dict(Counter(x['after'] for x in rows)), 'loader_rule':'New PyTorch iterator construction may draw base_seed even with shuffle=False and num_workers=0. Persistent iterator reuse is not new construction. Worker_init_fn can override downstream worker seeds.', 'mmengine_dependency':{'commit':'a2e410bd48890c54f16c527d67c371ed8562facd','path':dependency.relative_to(ROOT).as_posix(),'lines':'34-39','sha256':digest(dependency)},'repositories':rows}

def node(p):
    d=json.loads(p.read_text())
    return d.get('result',d)

def build_evidence():
    cells=[]
    for family in ['logs_nlp_gateb','logs_nlp_payout_det']:
        for p in sorted((ROOT/family).rglob('topo_sel_log.jsonl')):
            lines=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            if not lines: continue
            last=lines[-1]
            selected=last.get('val2_selected_primary'); raw=last.get('val2_raw_primary')
            cells.append({'campaign':family,'cell':p.parent.relative_to(ROOT/family).as_posix(),'source':p.relative_to(ROOT).as_posix(),'source_sha256':digest(p),'champion':last.get('champion'),'selected_metric':selected,'within_arm_raw_metric':raw,'selected_equals_within_arm_raw_metric':selected is not None and selected==raw,'cross_arm_weight_equality_verified':False,'full_state_equality_verified':False,'independent_repeat_floor_verified':False,'certificate_eligible':False,'reason':'Selection/metric log alone is not independent baseline weight evidence or within-configuration repeat evidence.'})
    pairs=[]
    for family,pattern,base in [('logs_fullrecipe','probe_isolated_p*_s*.json','mwnet_s{seed}.json'),('logs_fullrecipe','mwnet_rep_s*.json','mwnet_s{seed}.json'),('logs_moneyshot','ext21_re_seed*_iso_p1400.json','ext21_re_seed{seed}_rep.json')]:
        for p in sorted((ROOT/family).glob(pattern)):
            seed=p.stem.split('seed')[1].split('_')[0] if 'seed' in p.stem else p.stem.rsplit('_s',1)[1]
            q=p.parent/base.format(seed=seed)
            a,b=node(p),node(q)
            keys=['state_sha256','vnet_state_sha256']
            equal=all(isinstance(a.get(k),str) and len(a[k])==64 and a[k]==b.get(k) for k in keys)
            pairs.append({'campaign':family,'comparison_kind':'independent_repeat_floor' if p.stem.startswith('mwnet_rep') else 'cross_arm_isolated_vs_baseline','arm':p.relative_to(ROOT).as_posix(),'baseline':q.relative_to(ROOT).as_posix(),'hash_fields':keys,'recorded_hashes':{k:a.get(k) for k in keys},'equal':equal,'resolution':'model and meta-network state_dict weights/buffers; not complete optimizer/RNG state','full_state_equality_verified':False})
    return {'schema':1,'interpretation':'Source-file SHA checks provenance only. Hash values inside run records concern only their explicitly named training components. Missing evidence is unknown, not observed inequality.', 'nlp_cells':cells,'nlp_summary':{'cells_with_selection_logs':len(cells),'direct_weight_verified':0,'full_state_verified':0,'independent_floor_verified':0,'certificate_eligible':0},'individually_checked_weight_pairs':pairs,'historical_guard_8_hash_verified_claim':'Withdrawn: original figure script derived 8 from table text, not a per-cell hash registry. Historical metric outcomes retained.', 'unverified_campaigns':['vision tuned guard per-cell weight/floor registry not established by this ledger','shortened MW-Net witness handled by separate run records; not inferred from guard counts']}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--check',action='store_true'); args=parser.parse_args()
    for name,value in [('audit_ledger.json',build_audit()),('evidence_ledger.json',build_evidence())]:
        p=ROOT/'revision_results'/name
        if args.check:
            assert json.loads(p.read_text())==value, f'Stale ledger: {p}'
        else:
            p.parent.mkdir(exist_ok=True); p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(name, 'checked' if args.check else 'rebuilt')

if __name__=='__main__': main()
