"""No-refit public reset using exact fitted graph, avoiding KD-tree reorder on translation."""
import importlib.util,json,pickle,os
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17.native_phystwin_fit import ROOT,sha,write
BASE=ROOT/'runs/compositional-dynamics-v17/native-phystwin-calibration'
OUT=ROOT/'runs/compositional-dynamics-v17/native-phystwin-graph-recovery'
PLAN=ROOT/'reports/compositional-dynamics-v17/native-phystwin-graph-recovery-plan.json'
def replace_initial_vertices(native_tuple,object_points,controller_points):
    import torch
    _,edges,lengths,masses,count=native_tuple
    vertices=object_points if controller_points is None else torch.cat([object_points,controller_points],dim=0)
    if vertices.shape!=(len(masses),3):raise ValueError('Reset vertex census')
    return vertices,edges,lengths,masses,count

def guard():
    p=json.loads(PLAN.read_text())
    for name,digest in p['sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Changed recovery evidence '+name)
    return p

def main():
    guard()
    if OUT.exists():raise ValueError('Preserve recovery output')
    OUT.mkdir()
    # Reuse immutable fit artifacts; no old predictions or old failure records overwritten.
    for name in ['fit-complete.json','optimal_params.pkl','fitted-spring-endpoints.npy']:(OUT/name).symlink_to(BASE/name)
    source=ROOT/'scripts/compositional_dynamics_v17/native_phystwin_fit.py'
    spec=importlib.util.spec_from_file_location('_native_fixed_graph_recovery',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    original=m.runtime_classes
    with (BASE/'prefix.pkl').open('rb') as f:prefix=pickle.load(f)
    fixed=np.concatenate([prefix['object_points'][0],prefix['surface_points'],prefix['interior_points']],axis=0)
    def classes(masses,alpha):
        import torch
        CMA,Trainer=original(masses,alpha)
        class FixedGraph(Trainer):
            def _init_start(self,object_points,controller_points,*args,**kwargs):
                old=torch.as_tensor(fixed,dtype=object_points.dtype,device=object_points.device)
                graph=super()._init_start(old,controller_points,*args,**kwargs)
                if not np.array_equal(graph[1].detach().cpu().numpy(),np.load(BASE/'fitted-spring-endpoints.npy')):raise ValueError('Exact fitted graph reconstruction failed')
                return replace_initial_vertices(graph,object_points,controller_points)
        return CMA,FixedGraph
    m.OUT=OUT;m.runtime_classes=classes
    m.run('predict');guard()
    write(OUT/'recovery-complete.json',dict(completed=True,recovery_plan_sha256=sha(PLAN),prediction_commit_sha256=sha(OUT/'predict-complete.json'),forecast_sha256=sha(OUT/'forecast.npz'),scope='Identical fitted graph/rest lengths/weights; public initial vertex reset only. Old reordered-graph failure retained. No refit or private forecast target access.'))
if __name__=='__main__':main()
