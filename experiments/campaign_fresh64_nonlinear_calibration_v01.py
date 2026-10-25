"""Pure prospective fixed-load fresh64 calibration helpers; no import-time RNG.

Actual parameter/noise draws belong behind the runner's frozen root release.
No adaptive policy or quadrature-atom posterior ranks are used.
"""
import numpy as np
from experiments.campaign_cluster_calibration_v01 import primary_calibration,paired_cluster_intervals
OBJECTS=64
PARAMETER_SEED_BASE=2026102600
NOISE_SEED_BASE=2026102800
BOOTSTRAP_SEED=2026102999
PREFIX='normal_x180'
FAMILIES=('normal','axial')
PROBES=('normal_x180','paired_axial_x180')
TARGETS=('normal_x135','normal_x225','paired_axial_x135','paired_axial_x225')
TRUTH_LABELS=(PREFIX,'paired_axial_x180')+TARGETS
MODELS=('linear','nonlinear')
PHASES=('prefix_only','prefix_plus_independent_centralnormal','prefix_plus_independent_centralaxial')
CALIBRATION_LABELS=tuple(phase+'_'+parameter for phase in PHASES for parameter in ('E','K'))


def draw_fresh_object(index):
    if type(index)is not int or not 0<=index<64:raise ValueError('Exactly64prospective independent objects')
    unit=np.random.default_rng(PARAMETER_SEED_BASE+index).uniform(size=2)
    generator=np.random.default_rng(NOISE_SEED_BASE+index)
    prefix_noise=generator.normal(0,20e-6,size=24)
    probe_noise=generator.normal(0,20e-6,size=(2,3,24))
    return dict(unit=unit,physical_parameters=np.exp(np.log([2000.,5.])+unit*np.log([4.,6.])),prefix_noise=prefix_noise,probe_noise=probe_noise)


def paired_observations(truth_responses,prefix_noise,probe_noise,*,family=None,reset=None):
    response=np.asarray(truth_responses,float);pn=np.asarray(prefix_noise,float);qn=np.asarray(probe_noise,float)
    if response.shape!=(6,24)or pn.shape!=(24,)or qn.shape!=(2,3,24)or not all(np.isfinite(v).all()for v in(response,pn,qn)):raise ValueError('Six direct responses and fixed paired independent noise arrays')
    result=[(PREFIX,response[0]+pn)]
    if family is not None:
        if family not in(0,1)or reset not in(0,1,2):raise ValueError('Two fixed alternative probes and3resets')
        label=PROBES[family];result.append((label,response[TRUTH_LABELS.index(label)]+qn[family,reset]))
    elif reset is not None:raise ValueError('Prefix has no artificial reset copies')
    return result


def exact_density_calibration(reference,final,true_unit,*,outer,inner):
    u=np.asarray(true_unit,float);q=np.asarray(final['unit_quantiles'],float)
    if u.shape!=(2,)or q.shape!=(2,3)or not np.isfinite(u).all()or not np.isfinite(q).all()or np.any((u<0)|(u>1))or np.any(np.diff(q,axis=1)<0):raise ValueError('True intrinsic parameters and finite central posterior quantiles')
    pit=np.array([reference.cdf(d,float(u[d]),outer=outer,inner=inner,log_evidence=final['log_evidence'])for d in(0,1)])
    if not np.isfinite(pit).all()or np.any((pit<0)|(pit>1)):raise FloatingPointError('Invalid continuous posterior PIT')
    return dict(covered=(u>=q[:,0])&(u<=q[:,2]),PIT=pit,intrinsic_interval_width=q[:,2]-q[:,0],intrinsic_signed_bias=np.asarray(final['unit_mean'])-u)


def primary_arrays(records,model):
    """Explicit64objects×(phase-major E,K) primary arrays; never drop failures."""
    if model not in MODELS or len(records)!=64 or [r['index']for r in records]!=list(range(64)):raise ValueError('All64orderedobject records retained')
    covered=np.full((64,6),np.nan);pit=covered.copy()
    for row in records:
        for phase,key in enumerate(('prefix','normal-r0','axial-r0')):
            record=row.get('posteriors',{}).get(model,{}).get(key,{})
            if record.get('qualified')and record.get('physical_valid'):
                c=np.asarray(record['covered'],float);p=np.asarray(record['PIT'],float)
                if c.shape!=(2,)or p.shape!=(2,):raise ValueError('Two material parameters')
                covered[row['index'],2*phase:2*phase+2]=c;pit[row['index'],2*phase:2*phase+2]=p
    return covered,pit


def summarize_calibration(records):
    result={}
    for model in MODELS:
        c,p=primary_arrays(records,model)
        result[model]=primary_calibration(c,p,labels=CALIBRATION_LABELS)
    result['nonlinear_primary_family']=list(CALIBRATION_LABELS)
    result['linear_role']='Comparator diagnostics only; nonlinear six-entry family is predeclared primary'
    result['comparison_blocked']=not all(result[m]['complete']for m in MODELS)
    return result


def paired_prediction_metrics(records):
    """All primary+secondary paired differences in ONE shared-index bootstrap."""
    if len(records)!=64 or [r['index']for r in records]!=list(range(64)):raise ValueError('All64objects, no filters')
    names=[];columns=[]
    def add(name,values):names.append(name);columns.append(np.asarray(values,float))
    def posterior(row,model,key):return row.get('posteriors',{}).get(model,{}).get(key,{})
    def metric(model,key):return np.array([posterior(r,model,key).get('target_RMS_m',np.nan)if posterior(r,model,key).get('qualified')and posterior(r,model,key).get('physical_valid')else np.nan for r in records])
    for model in MODELS:
        prefix=metric(model,'prefix');add(model+'_prefix_RMS_m',prefix)
        for family in FAMILIES:
            values=np.stack([metric(model,family+f'-r{r}')for r in range(3)],axis=1)
            add(model+'_'+family+'_reset0_RMS_m',values[:,0]);add(model+'_'+family+'_mean3_RMS_m',values.mean(axis=1))
            add(model+'_prefix_minus_'+family+'_reset0_RMS_m',prefix-values[:,0]);add(model+'_prefix_minus_'+family+'_mean3_RMS_m',prefix-values.mean(axis=1))
    for phase in('prefix','normal-r0','axial-r0'):add('nonlinear_minus_linear_'+phase+'_RMS_m',metric('nonlinear',phase)-metric('linear',phase))
    def attribute(model,key,attribute,parameter):
        return np.array([posterior(r,model,key).get(attribute,[np.nan,np.nan])[parameter]if posterior(r,model,key).get('qualified')and posterior(r,model,key).get('physical_valid')else np.nan for r in records],float)
    for model in MODELS:
        for parameter,name in enumerate(('E','K')):
            for key in('prefix','normal-r0','axial-r0'):
                for attr in('intrinsic_interval_width','intrinsic_signed_bias'):
                    add(model+'_'+key+'_'+name+'_'+attr,attribute(model,key,attr,parameter))
            for family in FAMILIES:
                for attr in('covered','intrinsic_interval_width','intrinsic_signed_bias'):
                    values=np.stack([attribute(model,family+f'-r{r}',attr,parameter)for r in range(3)],axis=1)
                    add(model+'_'+family+'_'+name+'_mean3_'+attr,values.mean(axis=1))
        add(model+'_knownparameter_oracle_RMS_m',[r.get('oracle_RMS_m',{}).get(model,np.nan)for r in records])
    matrix=np.column_stack(columns)
    if not np.isfinite(matrix).all():return dict(complete=False,comparison_blocked=True,objects=64,unresolved_entries=int((~np.isfinite(matrix)).sum()),metric_labels=names)
    return dict(complete=True,comparison_blocked=False,objects=64,shared_bootstrap=paired_cluster_intervals(matrix,labels=names,seed=BOOTSTRAP_SEED,resamples=10000))
