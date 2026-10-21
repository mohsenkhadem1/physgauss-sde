"""Scientific report from sealed evaluation artifacts, without model updates."""
import argparse,hashlib,html,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FLOW=('25d-rgb','25d-rgbd','25d-prior','chart-rgb','chart-rgbd','chart-prior')
MODELS=('ridge-rgb','ridge-rgbd',*FLOW)
LABELS={'ridge-rgb':'Ridge RGB','ridge-rgbd':'Ridge RGBD','25d-rgb':'25D RGB','25d-rgbd':'25D RGBD',
        '25d-prior':'25D prior','chart-rgb':'Chart RGB','chart-rgbd':'Chart RGBD','chart-prior':'Chart prior'}
COLORS={'ridge-rgb':'#a77448','ridge-rgbd':'#447f83','25d-rgb':'#c97132','25d-rgbd':'#26769e',
        '25d-prior':'#909090','chart-rgb':'#e6a064','chart-rgbd':'#6ba6be','chart-prior':'#b7b7b7'}
DATASETS={'fresh64':'gaussian-boundary-neural-v01','old12':'isaac-boundary-orbit-v0.1'}
TASKS={'full':('gap_coordinate_rmse_m','static_gap_coordinate_rmse_m','Withheld 1–2 s gap · ten views'),
       'prefix':('late_2to3_coordinate_rmse_m','static_late_2to3_coordinate_rmse_m','Future 2–3 s · inputs ≤1 s')}


def read(path):return json.loads(Path(path).read_text())


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def bootstrap_ratio(numerator,denominator,groups,repeats=4000,seed=20261111):
    numerator=np.asarray(numerator);denominator=np.asarray(denominator);unique=sorted(set(groups))
    a=np.array([numerator[np.asarray(groups)==g].sum() for g in unique]);b=np.array([denominator[np.asarray(groups)==g].sum() for g in unique])
    selected=np.random.default_rng(seed).integers(len(unique),size=(repeats,len(unique)))
    gains=100*(1-a[selected].sum(1)/b[selected].sum(1))
    return {'gain_percent':float(100*(1-a.sum()/b.sum())),
            'group_bootstrap95_percent':np.quantile(gains,[.025,.975]).tolist(),'independent_groups':len(unique)}


def subset(rows,dataset):return [r for r in rows if r['key'].startswith(DATASETS[dataset]+'/')]


def load_study():
    seal=read('runs/gaussian-final-passive-seal.json');digest=seal['protocol_sha256'];data={};parameters={};paths=[];groups={}
    for root in ('runs/gaussian-boundary-neural-v01','runs/isaac-boundary-orbit-v01'):
        plan=read(Path(root)/'plan.json')
        for case in plan['episodes']:
            if case['split']=='test':groups[plan['dataset']+'/'+case['episode_id']]=plan['dataset']+'/'+case['group_id']
    for model in MODELS:
        data[model]={}
        for task in TASKS:
            data[model][task]={}
            if model.startswith('ridge'):
                path=f'runs/gaussian-final-test-ridge-{model.split("-")[1]}-{task}-v01/results.json';raw=read(path);paths.append(path)
                assert raw['final_test_protocol_sha256']==digest
                for dataset in DATASETS:data[model][task][dataset]=subset(raw['episodes'],dataset)
            else:
                for dataset in DATASETS:
                    path=f'runs/gaussian-final-test-{dataset}-{model}-{task}-v01/results.json';raw=read(path);paths.append(path)
                    assert raw['final_test_protocol_sha256']==digest and raw['samples']==128
                    data[model][task][dataset]=raw['episodes']
            for dataset,count in [('fresh64',64),('old12',12)]:assert len(data[model][task][dataset])==count
    for model in FLOW:
        parameters[model]={}
        for task in TASKS:
            path=f'runs/gaussian-final-test-parameters-{model}-{task}-v01/results.json';raw=read(path);paths.append(path)
            assert raw['final_test_protocol_sha256']==digest and raw['samples']==128
            parameters[model][task]={dataset:subset(raw['episodes'],dataset) for dataset in DATASETS}
    views={}
    for model in ('25d-rgb','25d-rgbd','chart-rgb','chart-rgbd'):
        path=f'runs/gaussian-final-active-view-{model}-test-v01/results.json';raw=read(path);paths.append(path)
        assert raw['protocol']['split']=='test' and len(raw['episodes'])==16
        assert raw['protocol']['samples']==128 and raw['protocol']['observation_prefix_s']==1.
        views[model]=raw
    return seal,data,parameters,views,groups,sorted(set(paths))


def metrics_summary(data,parameters,views,groups):
    summary={'passive':{},'physical':{},'active_views':{}}
    for dataset in DATASETS:
        summary['passive'][dataset]={}
        for model in MODELS:
            summary['passive'][dataset][model]={}
            for task,(metric,baseline,_) in TASKS.items():
                rows=data[model][task][dataset];value={
                    'mean_coordinate_rmse_mm':float(np.mean([r[metric] for r in rows])*1000),
                    'static_mean_coordinate_rmse_mm':float(np.mean([r[baseline] for r in rows])*1000),
                    'proper_energy_score_6times64nodes_mm':float(np.mean([r['predictive_energy_score_6times64nodes_m'] for r in rows])*1000),
                    **bootstrap_ratio([r[metric] for r in rows],[r[baseline] for r in rows],[groups[r['key']] for r in rows])}
                if 'pointwise_coordinate90_coverage' in rows[0]:
                    value['pointwise_coordinate90_coverage']=float(np.mean([r['pointwise_coordinate90_coverage'] for r in rows]))
                    value['pointwise_coordinate90_width_mm']=float(np.mean([r['pointwise_coordinate90_mean_width_m'] for r in rows])*1000)
                if model in FLOW and 'prior' not in model:
                    ridge='ridge-rgbd' if model.endswith('rgbd') else 'ridge-rgb'
                    reference={r['key']:r[metric] for r in data[ridge][task][dataset]}
                    value['versus_matched_ridge']=bootstrap_ratio([r[metric] for r in rows],[reference[r['key']] for r in rows],[groups[r['key']] for r in rows])
                summary['passive'][dataset][model][task]=value
        summary['physical'][dataset]={}
        for model in FLOW:
            summary['physical'][dataset][model]={}
            for task in TASKS:
                rows=parameters[model][task][dataset]
                material={name:{stat:float(np.mean([r['material'][name][stat] for r in rows])) for stat in ('covered90','absolute_error','width90')} for name in rows[0]['material']}
                cats={name:{stat:float(np.mean([r[name][stat] for r in rows])) for stat in ('correct','brier','negative_log_probability')} for name in ('boundary_type','force_type','joint_type','support_face','load0_face','load1_face') if name in rows[0]}
                summary['physical'][dataset][model][task]={'material':material,'categorical':cats}
    for model,raw in views.items():
        summary['active_views'][model]={'sensor_units':raw['protocol']['sensor_units'],'aggregate':raw['aggregate'],'comparisons':{}}
        for metric in ('independent_view_sensor_rmse','future_coordinate_rmse_m'):
            summary['active_views'][model]['comparisons'][metric]={}
            for baseline in ('passive','random_expected','nearest_evaluation_camera','max_observation_variance'):
                summary['active_views'][model]['comparisons'][metric][baseline]=bootstrap_ratio(
                    [r['policies']['predictive_risk'][metric] for r in raw['episodes']],
                    [r['policies'][baseline][metric] for r in raw['episodes']],
                    [r['episode_id'] for r in raw['episodes']])
    return summary


def save(fig,output,name):
    fig.tight_layout();fig.savefig(output/(name+'.png'),dpi=180);fig.savefig(output/(name+'.pdf'));plt.close(fig)


def plots(summary,output):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8.8),sharey=True)
    for row,dataset in enumerate(DATASETS):
        for col,task in enumerate(TASKS):
            ax=axes[row,col]
            for i,model in enumerate(MODELS):
                s=summary['passive'][dataset][model][task];ci=s['group_bootstrap95_percent']
                ax.plot(ci,[i,i],color=COLORS[model],lw=2);ax.plot(s['gain_percent'],i,'o',color=COLORS[model],ms=6)
            ax.axvline(0,color='#555',ls='--',lw=1);ax.grid(axis='x',alpha=.15)
            ax.set_title(('Fresh 64 independent layouts' if row==0 else 'Original 12 episodes / 4 layouts')+'\n'+TASKS[task][2]);ax.set_xlabel('Improvement over static (%)')
    axes[0,0].set_yticks(range(len(MODELS)),[LABELS[m] for m in MODELS]);axes[0,0].invert_yaxis();save(fig,output,'sealed-passive-gain')
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
    for row,dataset in enumerate(DATASETS):
        for col,task in enumerate(TASKS):
            ax=axes[row,col];values=[100*summary['physical'][dataset][m][task]['material']['youngs_modulus_Pa']['covered90'] for m in FLOW]
            ax.barh(range(len(FLOW)),values,color=[COLORS[m] for m in FLOW]);ax.axvline(90,color='#a22',ls='--',lw=1)
            ax.set_xlim(0,100);ax.set_xlabel('Empirical Young’s modulus 90% coverage (%)');ax.set_title(('Fresh 64' if row==0 else 'Original 12 / 4 layouts')+' · '+('ten views' if task=='full' else 'prefix ≤1 s'));ax.grid(axis='x',alpha=.15)
    axes[0,0].set_yticks(range(len(FLOW)),[LABELS[m] for m in FLOW]);axes[0,0].invert_yaxis();save(fig,output,'sealed-material-coverage')
    modes=('25d-rgb','25d-rgbd','chart-rgb','chart-rgbd');fig,axes=plt.subplots(2,2,figsize=(12,6.7),sharey=True)
    for row,metric in enumerate(('independent_view_sensor_rmse','future_coordinate_rmse_m')):
        for col,baseline in enumerate(('random_expected','nearest_evaluation_camera')):
            ax=axes[row,col]
            for i,model in enumerate(modes):
                s=summary['active_views'][model]['comparisons'][metric][baseline]
                ax.plot(s['group_bootstrap95_percent'],[i,i],color=COLORS[model],lw=2);ax.plot(s['gain_percent'],i,'o',color=COLORS[model])
            ax.axvline(0,color='#555',ls='--',lw=1);ax.grid(axis='x',alpha=.15);ax.set_xlabel('Predictive-risk improvement (%)')
            ax.set_title(('Independent new-view sensor' if row==0 else 'Future 2–3 s motion')+'\nvs '+('exact uniform-random expectation' if col==0 else 'nearest evaluation camera'))
    axes[0,0].set_yticks(range(4),[LABELS[m] for m in modes]);axes[0,0].invert_yaxis();save(fig,output,'sealed-active-view-gain')


def probe_section(raws,development,output):
    metric='withheld_control_coordinate_rmse_m';policies=('passive','predictive_risk','random_expected','fixed0','fixed_last','max_observation_variance')
    rows=[];summary={}
    for model,raw in raws.items():
        rows.append([LABELS[model]]+[f'{raw["aggregate"][p][metric]*1000:.3f}' for p in policies])
        summary[model]={'aggregate':raw['aggregate'],'cases':[r['episode_id'] for r in raw['episodes']],'comparisons':{}}
        for baseline in ('random_expected','fixed_last'):
            summary[model]['comparisons'][baseline]=bootstrap_ratio([r['policies']['predictive_risk'][metric] for r in raw['episodes']],
                [r['policies'][baseline][metric] for r in raw['episodes']],[r['episode_id'] for r in raw['episodes']])
    if development:
        caption='<h2>Development-only calibrated-poke diagnostic</h2><p>Four original VAL layouts (e0002,e0005,e0008,e0015) were used for this diagnostic. These results do not count as held-out TEST evidence. The target is a separate withheld-control trajectory after one independent reset probe, using the explicitly controlled neural teacher.</p>'
    else:
        caption='<h2>Sealed TEST calibrated poking</h2><p>Eight held-out layouts compare one uncertainty-selected reset probe with exact uniform-random expectation, fixed first/last actions and maximum observation variance. The target is an independent withheld-control trajectory. A separately trained controlled teacher is used only for controlled queries; it does not replace the passive decoder.</p>'
        fig,axes=plt.subplots(1,2,figsize=(11,3.8),sharey=True)
        for ax,baseline in zip(axes,('random_expected','fixed_last')):
            for i,model in enumerate(raws):
                s=summary[model]['comparisons'][baseline];ax.plot(s['group_bootstrap95_percent'],[i,i],color=COLORS[model],lw=2);ax.plot(s['gain_percent'],i,'o',color=COLORS[model])
            ax.axvline(0,color='#555',ls='--',lw=1);ax.grid(axis='x',alpha=.15);ax.set_xlabel('Predictive-risk improvement (%)');ax.set_title('Withheld-control response\nvs '+('uniform-random expectation' if baseline=='random_expected' else 'fixed last action'))
        axes[0].set_yticks(range(len(raws)),[LABELS[m] for m in raws]);axes[0].invert_yaxis();save(fig,output,'sealed-active-probe-gain')
        caption+='<img src="sealed-active-probe-gain.png" alt="Sealed probe policy gains versus random expectation and fixed last action">'
    caption+=table(['Model','No probe','Predictive risk','Uniform random expected','Fixed first','Fixed last','Max variance'],rows)
    caption+='<p class="note">Entries are mean withheld-control coordinate RMSE in mm. Force interventions repeat the same supports and original hardware after reset. Conservative working likelihood parameters were frozen on reserved TRAIN intervention cases, including censored visibility and range returns; they do not guarantee physical posterior calibration.</p>'
    if not development:
        physical_rows=[]
        for model,raw in raws.items():
            summary[model]['physical_after_probe']={}
            for policy in ('passive','predictive_risk','random_expected','fixed_last'):
                values=[]
                for episode in raw['episodes']:
                    choices=[episode['policies'][policy]] if policy!='random_expected' else [v for k,v in episode['policies'].items() if k.startswith('random_trial')]
                    measures=[p['physical_uncertainty_evaluation_only']['youngs_modulus_Pa'] for p in choices]
                    values.append([np.mean([v['covered90'] for v in measures]),np.mean([v['width90'] for v in measures]),np.mean([v['absolute_median_error'] for v in measures])])
                coverage,width,error=np.mean(values,axis=0);summary[model]['physical_after_probe'][policy]={'coverage90':float(coverage),'width90_Pa':float(width),'absolute_median_error_Pa':float(error)}
                physical_rows.append([LABELS[model],policy,f'{coverage*100:.1f}%',f'{width/1000:.2f}',f'{error/1000:.2f}'])
        caption+='<h3>Material uncertainty before and after poking: only eight layouts</h3>'+table(['Model','Policy','E90% coverage','E interval width(kPa)','E median MAE(kPa)'],physical_rows)
        caption+='<p class="note">Weighted quantiles use the inverse empirical CDF, preserving zero-mass gaps. Random expectation averages coverage/width over the four possible actions, not quantiles of an invented mixed posterior. Eight layouts provide very limited calibration evidence; narrowing an interval is not evidence of correct uncertainty.</p>'
        caption+='<p>For 25D RGBD, one selected poke reduces Young’s-modulus 90% coverage from 87.5% to 62.5% and raises median-estimate MAE from 4.33 to 4.89 kPa while mean interval width shrinks from 19.31 to 15.88 kPa. This is a failed physical-uncertainty update despite lower response-prediction error.</p>'
        caption+='<p>The prior-plus-poke variant improves mean control-response prediction by about18% over its passive prior, but the fixed-last action nearly ties it. The25D RGBD policy also nearly ties fixed-last. Chart RGBD becomes worse than its passive prediction. These results do not establish a novel active-selection advantage.</p>'
    return caption,summary


def image_section(raw,output):
    methods=raw['models'];labels={'PIDG_inspired_RGB':'PIDG-inspired RGB','deformation_only_RGB':'Deformation-only RGB',
        'ridge_RGB':'Ridge RGB','ridge_RGBD_secondary':'Ridge RGBD','flow25_RGB':'25D RGB','flow25_RGBD_secondary':'25D RGBD','flowchart_RGB':'Chart RGB','flowchart_RGBD_secondary':'Chart RGBD'}
    methods={k:v for k,v in methods.items() if k in labels};rows=[]
    fig,axes=plt.subplots(1,3,figsize=(12,5.3),sharey=True)
    for i,(name,value) in enumerate(methods.items()):
        a=value['aggregate'];orbit=a['withheld_orbit']['prediction'];view=a['independent_same_state_view']['prediction']
        gap=value.get('gap_coordinate_rmse_m')
        rows.append([labels[name],f'{value["coordinate_rmse_m"]*1000:.3f}',f'{gap*1000:.3f}' if gap is not None else '—',f'{orbit["psnr_db"]:.3f}',f'{view["psnr_db"]:.3f}',f'{value["minimum_volume_ratio"]:.3f}'])
        color='#866345' if name.startswith(('PIDG','deformation')) else '#4986a4'
        for ax,score in zip(axes,(value['coordinate_rmse_m']*1000,orbit['psnr_db'],view['psnr_db'])):ax.barh(i,score,color=color)
    first=next(iter(methods.values()))['aggregate']
    axes[1].axvline(first['withheld_orbit']['static']['psnr_db'],ls='--',color='#555',lw=1)
    axes[2].axvline(first['independent_same_state_view']['static']['psnr_db'],ls='--',color='#555',lw=1)
    axes[0].set_yticks(range(len(methods)),[labels[m] for m in methods]);axes[0].invert_yaxis()
    for ax,title in zip(axes,('Full 61 coordinate RMSE(mm)↓','Withheld-orbit PSNR(dB)↑','Independent-pose PSNR(dB)↑')):ax.set_title(title);ax.grid(axis='x',alpha=.15)
    save(fig,output,'sealed-image-baselines')
    text='<h2>Matched four-case images and scene-specific baselines</h2><p>Cases z0000,z0005,z0010,z0015 use identical known-rest crops, cameras, frozen TRAIN-only appearance calibration and rendering. Independent PIDG-inspired and deformation-only fits use ten public RGB observations, a fixed200-step budget and fixed seed; this is not an official PIDG reproduction or a claim of fully optimized published-method performance.</p><img src="sealed-image-baselines.png" alt="Matched geometry and image scores for eight methods">'
    text+=table(['Model','Full61 qRMSE(mm)','Strict gap qRMSE(mm)','Orbit PSNR(dB)','Independent-pose PSNR(dB)','Minimum tet volume ratio'],rows)
    text+='<p>Optimizing image error improves orbit PSNR while worsening mechanical predictions. PIDG-inspired RGB reaches29.529dB but1.860mm full61 error;25D RGB reaches28.678dB and1.076mm. On this four-case strict gap,25D RGB and ridge RGB are approximately tied with static, while RGBD25D is worse than static. These subset outcomes do not replace the primary64-layout test.</p><p class="note">The separately stored privileged true-motion render has29.189dB orbit and28.421dB independent-pose PSNR. PIDG-inspired orbit PSNR exceeds that shader diagnostic despite incorrect geometry, illustrating the simulator/Gaussian appearance mismatch. True motion is excluded from model fitting and the eight-method comparison.</p>'
    return text


def build(output,update=False):
    output=Path(output)
    if output.exists() and not update:raise FileExistsError('Choose a fresh report directory, or explicitly update the unarchived draft')
    seal,data,parameters,views,groups,paths=load_study();summary=metrics_summary(data,parameters,views,groups)
    probe_seal=read('runs/gaussian-final-probe-seal.json');paths.extend(['runs/gaussian-final-passive-seal.json','runs/gaussian-final-probe-seal.json'])
    output.mkdir(parents=True,exist_ok=update);plots(summary,output);(output/'source.py').write_bytes(Path(__file__).read_bytes())
    probe_models=('25d-rgbd','chart-rgbd','chart-prior')
    development_version='v02' if all(Path(f'runs/gaussian-final-active-probe-{m}-validation-v02/results.json').exists() for m in probe_models) else 'v01'
    development={m:read(f'runs/gaussian-final-active-probe-{m}-validation-{development_version}/results.json') for m in probe_models}
    development_html,summary['development_probes']=probe_section(development,True,output)
    for m in development:paths.append(f'runs/gaussian-final-active-probe-{m}-validation-{development_version}/results.json')
    test_paths={m:Path(f'runs/gaussian-final-active-probe-{m}-test-v01/results.json') for m in development}
    if all(p.exists() for p in test_paths.values()):
        probes={m:read(p) for m,p in test_paths.items()}
        for raw in probes.values():assert len(raw['episodes'])==8 and raw['protocol']['split']=='test'
        probe_html,summary['test_probes']=probe_section(probes,False,output);paths.extend(str(p) for p in test_paths.values())
    else:probe_html='<h2>Calibrated-poke TEST status</h2><p>The predeclared eight-layout TEST capture/evaluation is still pending. Development diagnostics below do not substitute for that test.</p>'
    calibration_path='runs/gaussian-probe-censored-calibration-v01/frozen-calibration.json';calibration=read(calibration_path);paths.append(calibration_path)
    c=calibration['covariance_parameters'];marginal_std=np.sqrt(sum(c[k]**2 for k in ('independent_std_m','frame_common_std_m','sequence_common_std_m')))*1000
    development_html+=f'<p class="note">The working covariance used four reserved TRAIN cases / sixteen sequences, marginal range standard deviation {marginal_std:.2f}mm and {100*calibration["calibration_diagnostics"]["nominal_gaussian_90_coverage_joint_hits"]:.2f}% empirical coverage of jointly valid TRAIN-calibration returns. This is conservative likelihood calibration, not a TEST or physical-parameter coverage guarantee.</p>'
    image_path=Path('runs/gaussian-baseline-images-test-v01/summary.json')
    if image_path.exists():
        image_data=read(image_path);assert image_data['protocol_sha256']==seal['protocol_sha256']
        image_html=image_section(image_data,output);summary['matched_image_baselines']=image_data;paths.append(str(image_path))
    else:image_html='<p>Matched fixed-budget scene-specific image comparison is pending.</p>'
    video_html='<h2>Actual Isaac simulation videos</h2>'
    for name,title in [('isaac-test-scenarios','Held-out scenarios and moving-camera orbit'),('isaac-test-calibrated-probe','Actual calibrated probe response')]:
        path=Path('examples/gaussian-neural-boundary')/(name+'.mp4');manifest=path.with_suffix('.manifest.json')
        if path.exists() and manifest.exists():
            paths.extend([str(path),str(manifest)])
            video_html+=f'<h3>{title}</h3><video controls preload="metadata" style="width:100%" poster="../../examples/gaussian-neural-boundary/{name}.poster.png"><source src="../../examples/gaussian-neural-boundary/{name}.mp4" type="video/mp4"></video><p><a href="../../examples/gaussian-neural-boundary/{name}.mp4">Download MP4</a> · <a href="../../examples/gaussian-neural-boundary/{name}.manifest.json">Source timestamps and SHA manifest</a></p>'
    summary.update(protocol_sha256=seal['protocol_sha256'],controlled_probe_protocol_sha256=probe_seal['protocol_sha256'],no_model_updates=True,
                   energy_score_scope={'times_s':[.5,1,1.5,2,2.5,3],'nodes':64,'qualification':'Fixed-grid all-time score; includes timestamps observed by ten-view inputs, not gap-only.'},
                   source_artifacts={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
                   confidence_interval_unit='Independent TEST layout groups; original three material variants remain together. Paired unadjusted95% bootstrap, not simultaneous multiple-comparison intervals.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(output/'seal.json').write_text(json.dumps(seal,indent=2)+'\n')
    (output/'passive-seal.json').write_text(json.dumps(seal,indent=2)+'\n');(output/'controlled-probe-seal.json').write_text(json.dumps(probe_seal,indent=2)+'\n')
    sections=[]
    for dataset in DATASETS:
        rows=[]
        first=summary['passive'][dataset][MODELS[0]]
        rows.append(['Static',f'{first["full"]["static_mean_coordinate_rmse_mm"]:.3f}',f'{first["prefix"]["static_mean_coordinate_rmse_mm"]:.3f}','—','—'])
        for model in MODELS:
            s=summary['passive'][dataset][model]
            rows.append([LABELS[model],f'{s["full"]["mean_coordinate_rmse_mm"]:.3f}',f'{s["prefix"]["mean_coordinate_rmse_mm"]:.3f}',f'{s["full"]["gain_percent"]:+.1f}%',f'{s["prefix"]["gain_percent"]:+.1f}%'])
        title='Primary TEST: 64 fresh independent layouts' if dataset=='fresh64' else 'Secondary TEST: 12 original episodes / 4 layout groups'
        sections.append('<h2>'+title+'</h2>'+table(['Model','Withheld gap RMSE(mm)','Future RMSE(mm)','Gap vs static','Future vs static'],rows))
    predictive_rows=[]
    for model in MODELS:
        s=summary['passive']['fresh64'][model];f=s['full'];p=s['prefix']
        predictive_rows.append([LABELS[model],f'{f["proper_energy_score_6times64nodes_mm"]:.3f}',f'{p["proper_energy_score_6times64nodes_mm"]:.3f}',
            f'{100*f["pointwise_coordinate90_coverage"]:.1f}%' if 'pointwise_coordinate90_coverage' in f else 'point only',
            f'{100*p["pointwise_coordinate90_coverage"]:.1f}%' if 'pointwise_coordinate90_coverage' in p else 'point only'])
    predictive_html='<h2>Distribution scores and coordinate intervals</h2>'+table(['Model','Ten-view energy score(mm)↓','Prefix energy score(mm)↓','Ten-view coordinate90% coverage','Prefix coordinate90% coverage'],predictive_rows)
    predictive_html+='<p class="note">Energy score is a proper distribution score on the fixed six-time/64-node normalized coordinate vector at 0.5,1,1.5,2,2.5,3 s, averaged by episode. This is a fixed-grid all-time score, including timestamps observed by the ten-view model, not a gap-only score. Ridge is treated as a point mass. Pointwise coverage uses all nondeterministic times and correlated coordinates; it does not establish physical calibration or simultaneous trajectory coverage.</p>'
    physical=[]
    for dataset in DATASETS:
        rows=[]
        for task in TASKS:
            for model in FLOW:
                p=summary['physical'][dataset][model][task];e=p['material']['youngs_modulus_Pa'];c=p['categorical']
                rows.append([LABELS[model],('ten views' if task=='full' else 'prefix'),f'{100*e["covered90"]:.1f}%',f'{e["absolute_error"]/1000:.2f}',f'{e["width90"]/1000:.2f}',f'{100*c["boundary_type"]["correct"]:.1f}%',f'{100*c["force_type"]["correct"]:.1f}%',f'{c["force_type"]["brier"]:.3f}'])
        physical.append('<h3>'+('Fresh 64' if dataset=='fresh64' else 'Original 12 / 4 groups')+'</h3>'+table(['Model','Inputs','E90% coverage','E MAE(kPa)','E width(kPa)','BC accuracy','Force accuracy','Force Brier↓'],rows))
    view_tables=[]
    for metric,title in [('independent_view_sensor_rmse','Independent new-view sensor'),('future_coordinate_rmse_m','Future motion after acquisition')]:
        rows=[]
        for model,raw in views.items():
            units=raw['protocol']['sensor_units'] if metric=='independent_view_sensor_rmse' else 'm'
            scale=1000 if units=='m' else 1;unit='mm' if units=='m' else 'pixels'
            rows.append([LABELS[model],unit]+[f'{raw["aggregate"][p][metric]*scale:.3f}' for p in ('passive','predictive_risk','random_expected','fixed0','nearest_evaluation_camera','max_observation_variance')])
        view_tables.append('<h3>'+title+'</h3>'+table(['Model','Unit','No new view','Predictive risk','Uniform random expected','Fixed0','Nearest camera','Max variance'],rows))
    contrast=[];ridgeval=read('runs/gaussian-ridge-final-validation-summary.json')['runs']
    for model in MODELS:
        if model.startswith('ridge'):
            mode=model.split('-')[1];vf=ridgeval[f'gaussian-ridge-final-{mode}-full-v01']['aggregate']['gap_coordinate_rmse_m'];vp=ridgeval[f'gaussian-ridge-final-{mode}-prefix-v01']['aggregate']['forecast_coordinate_rmse_m']
        else:
            vf=read(f'runs/gaussian-final-val-{model}-full-v01/results.json')['aggregate']['gap_coordinate_rmse_m'];vp=read(f'runs/gaussian-final-val-{model}-prefix-v01/results.json')['aggregate']['late_2to3_coordinate_rmse_m']
        t=summary['passive']['fresh64'][model]
        contrast.append([LABELS[model],f'{vf*1000:.3f}',f'{t["full"]["mean_coordinate_rmse_mm"]:.3f}',f'{vp*1000:.3f}',f'{t["prefix"]["mean_coordinate_rmse_mm"]:.3f}'])
    body='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sealed Gaussian deformation study</title>
<style>body{font:16px/1.55 system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 24px;color:#202c36}h1{font-size:36px;line-height:1.15}h2{margin-top:2.1em}h3{margin-top:1.7em}img{width:100%;height:auto}.callout{border-left:5px solid #b96936;background:#faf3ec;padding:16px 22px}.note{font-size:14px;color:#56646e}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:right;padding:9px 10px;border-bottom:1px solid #dce2e7}th:first-child,td:first-child{text-align:left}th{background:#eef2f4}code{overflow-wrap:anywhere;font-size:12px}a{color:#216b8a}.equation{font:22px Georgia,serif;text-align:center;padding:18px;background:#f1f5f7}</style>
<h1>Gaussian deformation prediction and active sensing</h1><p>Sealed TEST evidence · 152 TRAIN / 44 VAL / 64 primary TEST layouts / 12 secondary TEST episodes</p>
<div class="callout"><strong>Deformation prediction improved; reliable physical uncertainty and active-selection superiority remain unresolved.</strong><p>On64 fresh TEST layouts, the25D RGB and RGBD baselines reduce withheld-gap error by9.8% and13.8%. With a strict≤1s observation prefix, RGBD improves future prediction; RGB-only does not beat static. The structured chart makes physical hypotheses admissible but does not dominate the25D or ridge baselines and does not calibrate material/force uncertainty. Predictive-risk camera selection does not beat the nearest-camera baseline consistently.</p></div>
<h2>What was frozen and measured</h2><p>Checkpoint identities,128 posterior samples, input timestamps, comparison models and test episodes were frozen before each corresponding task's predictive TEST evaluation. Passive prediction/new-view inference and controlled probing have separate checkpoint seals; the controlled-response teacher was frozen separately before probe TEST scoring. No model or temperature was tuned on these results. Every episode has61 simulation frames and a shared known cuboid reference with455 nodes. The primary64 fresh layouts are separate from the original12 episodes, which contain only4 layout groups×3 material variants. All272 dataset cases passed sensor/physics QA before test execution.</p>
<p>Interpolation uses ten public images at0,.25,.5,.75,1,2,2.25,2.5,2.75 and3s, then predicts the withheld1–2s gap. Forecasting physically loads only images at or before1s and evaluates2–3s. Both use the same frozen neural Green decoder. RGBD explicitly uses measured depth; it is a separate track.</p>
<h2>Passive prediction against strong baselines</h2><img src="sealed-passive-gain.png" alt="Sealed prediction gains with independent-layout bootstrap intervals"><p class="note">Coordinate RMSE averages the three coordinate errors before the square root, then averages episodes. Error bars are paired, unadjusted95% layout-bootstrap intervals. Original material variants stay together; frames and nodes are not independent samples. Positive gains favor the model. Priors ignore video. Ridge is a separately sealed point predictor with modality/conditioning matched and no calibrated uncertainty.</p>
'''+''.join(sections)+predictive_html+image_html+'''
<h2>Physical uncertainty failed calibration</h2><img src="sealed-material-coverage.png" alt="Sealed Young modulus coverage by dataset and observation condition"><p>Valid surface centers and force directions do not establish recovered mechanics. On primary TEST, the chart RGBD model has81.25% Young’s-modulus coverage for nominal90% intervals with ten views, BC classification48.44%, and force classification26.56%. Its force Brier score0.763 remains worse than uniform4-class0.750. Secondary endpoint materials are substantially worse. Pointwise deformation coverage near90% must not be substituted for physical-parameter calibration.</p>
'''+''.join(physical)+'''
<h2>Why the ambiguity matters</h2><div class="equation">u ≈ (F / E) G(ν, ρ, boundary geometry)</div><p>In a quasi-static regime, an unknown force magnitude and stiffness can compensate for one another. Dynamics and known controlled excitation can add information, but motion agreement alone cannot verify the inferred material or supports. This experiment retains both plausible multimodality and substantial model error; it does not demonstrate a calibrated Bayesian inverse solution.</p>
<p>The chart uses16 intrinsic coefficients and discrete BC/load/face categories. It restricts centers to known cuboid faces, force families to their declared directions, and amplitudes/materials to declared ranges. Inactive coordinates are nuisance variables; finite-face clipping is noninvertible. Its density is intrinsic, not a25D physical Lebesgue density. It is specific to this cuboid and does not establish arbitrary COLMAP geometry or generalized3D boundary fields.</p>
<h2>One actual new view: compare with random and geometry</h2><p>Each of16 held-out layouts supplies a≤1s prefix, one acquired view at1.5s, and an independent evaluation camera. Candidate poses render the same object state. RGB uses silhouette support measurements in pixels; RGBD uses range measurements in metres. The policy is committed before its outcome is revealed. Uniform-random expectation evaluates all four possible one-view choices with equal weight; nearest camera and maximum observation variance are strong controls.</p><img src="sealed-active-view-gain.png" alt="Predictive risk versus uniform random expectation and nearest camera"><p class="note">Paired95% bootstrap intervals use16 independent layouts. Improvement against passive alone cannot establish a superior acquisition policy. RGB and RGBD sensor values have different units and are never pooled.</p>
'''+''.join(view_tables)+'''
<p>Extra observations often reduce independent-view error. Predictive-risk selection improves on uniform-random expectation only for the25D variants in this experiment, and neither establishes an advantage over nearest-evaluation-camera selection. The chart variants are worse than random expectation on the primary sensor target. Future-motion results also do not demonstrate consistent policy superiority.</p>
'''+probe_html+development_html+'''<h2>Validation and sealed-test contrast</h2>'''+table(['Model','VAL gap(mm)','Fresh TEST gap(mm)','VAL future(mm)','Fresh TEST future(mm)'],contrast)+'''
<p class="note">VAL contains32 fresh layouts plus12 original episodes; its material mixture differs from primary TEST. It selected checkpoints. The chart additionally trains with50% exact≤1s prefixes;25D retains original random-frame-dropout training. Differences cannot be assigned solely to the chart parameterization. Texture-feature and CE-weight sweeps were preserved as mixed/negative development results and were not tuned after test sealing.</p>
<h2>Scope, artifacts and remaining evidence</h2><p>Runtime inference uses an observation encoder, conditional flow and neural trajectory operator; no FEM or per-video optimization is used by these amortized models. TRAIN/VAL mechanical labels supervise learning; runtime observations use public RGB/masks/cameras, known reference geometry, and depth only in RGBD. Oracle geometry was used in declared TRAIN/VAL appearance calibration and renderer-ceiling diagnostics, and in a separately stored privileged four-case TEST true-motion shader diagnostic, which is not a ceiling because fitted incorrect geometry can score higher. It was never an input to the real predictive methods.</p>
<p>This study uses a supplied cuboid mesh, cameras and foreground masks, a rest initial state with zero velocity, and a three-second trajectory horizon. Material uncertainty concerns four spatially homogeneous scalar parameters: Young’s modulus, density, Poisson ratio and drag; no heterogeneous material field is inferred. Probe hardware inserts common additional constraints/load handles and is represented through the separate controlled-response model, rather than a fully general boundary-conditioning architecture.</p>
<p>The matched fixed-budget scene-specific image comparison is included above. Its optimizer budget does not establish published PIDG convergence. Controlled-query results are not silently replaced by the passive neural teacher or by development-only results. No official PIDG reproduction claim is made.</p>
<p><a href="summary.json">Complete machine-readable metrics and artifact hashes</a> · <a href="passive-seal.json">Passive/new-view checkpoint seal</a> · <a href="controlled-probe-seal.json">Separate controlled-probe checkpoint seal</a> · <a href="sealed-passive-gain.pdf">Passive plot PDF</a> · <a href="sealed-material-coverage.pdf">Material plot PDF</a> · <a href="sealed-active-view-gain.pdf">Active-view plot PDF</a></p>
<p class="note">Protocol SHA <code>'''+html.escape(seal['protocol_sha256'])+'''</code>. Controlled-probe protocol SHA <code>'''+html.escape(probe_seal['protocol_sha256'])+'''</code>. This report is a result summary, not a claim that the research objective of calibrated physical inference and superior active probing has been solved.</p></html>'''
    # Normalize scientific prose spacing without touching numerical artifacts.
    for old,new in [('On64','On 64'),('the25D','the 25D'),('by9.8%','by 9.8%'),('and13.8%','and 13.8%'),('strict≤1s','strict ≤1 s'),('before1s','before 1 s'),('evaluate2–3s','evaluate 2–3 s'),('withheld1–2s','withheld 1–2 s'),('at0,','at 0,'),('and3s','and 3 s'),('has61','has 61'),('with455','with 455'),('primary64','primary 64'),('original12','original 12'),('only4','only 4'),('groups×3','groups × 3'),('All272','All 272'),('unadjusted95%','unadjusted 95%'),('has81.25%','has 81.25%'),('nominal90%','nominal 90%'),('classification48.44%','classification 48.44%'),('classification26.56%','classification 26.56%'),('score0.763','score 0.763'),('uniform4-class0.750','uniform 4-class 0.750'),('near90%','near 90%'),('uses16','uses 16'),('a25D','a 25D'),('generalized3D','generalized 3D'),('of16','of 16'),('a≤1s','a ≤1 s'),('at1.5s','at 1.5 s'),('use16','use 16'),('contains32','contains 32'),('plus12','plus 12'),('with50%','with 50%'),('exact≤1s','exact ≤1 s')]:body=body.replace(old,new)
    body=body.replace('<h2>Physical uncertainty failed calibration</h2>','<p class="note">The mean future gains have bootstrap intervals spanning zero: 25D RGBD 5.25% [−1.58,11.54]%, chart RGBD 1.70% [−2.99,4.82]%. They do not establish robust forecasting improvement.</p><h2>Physical uncertainty failed calibration</h2>')
    body=body.replace('<h2>Scope, artifacts and remaining evidence</h2>',video_html+'<h2>Scope, artifacts and remaining evidence</h2>')
    (output/'index.html').write_text(body)
    return {'report':str(output/'index.html'),'protocol_sha256':seal['protocol_sha256'],'models_updated':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--update',action='store_true');a=p.parse_args();print(json.dumps(build(a.output,a.update),indent=2))
