"""Development-only force-isolation experiment; never a benchmark selector.
Requires scene-fem-development.json with source/start/stop manifest.
Run from repository root with PYTHONPATH=. and CPU thread limit 2.
"""
import json,time,numpy as np
from pathlib import Path
from scripts.contact_benchmark_v12.data import load
from scripts.contact_benchmark_v12.scene import load_scene,StaticScene
from scripts.contact_benchmark_v12.scene_fem import rollout,SceneFEM
from scripts.external_benchmark.evaluate import score,expand_prediction
def main():
    path=Path('runs/contact-benchmark-v12/scene-fem-development.json');old=json.loads(path.read_text());data=load(dict(subset='spatial-soft',task=0,episode=0));source=Path(old['source']);base=np.load(source/'prediction.npz')['predicted'];theta=np.load(source/'best.npz')['theta'];scene=load_scene(data);start=old['start'];stop=old['stop'];state=base[start-1];velocity=(base[start-1]-base[start-2])/(data['times'][start-1]-data['times'][start-2]);report=dict(name=data['name'],start=start,stop=stop,theta=theta.tolist(),source=str(source),variants={})
    mass=SceneFEM(data,scene).mass.cpu().numpy()
    for name,geometry,friction in [('no-field',scene,.3),('no-field-no-friction',scene,0.),('freeflight',StaticScene(floor=-10.),0.)]:
     report['variants'][name]=[];previous=None
     for steps in [20,80,160]:
      tic=time.monotonic();pred,meta=rollout(data,base[:stop],theta,geometry,start=start,substeps=(steps,),initial_state=state,initial_velocity=velocity,friction=friction,disable_field_on_release=True)
      np.savez_compressed(f'runs/contact-benchmark-v12/scene-fem-{name}-{steps}.npz',predicted=pred)
      row=dict(steps=steps,elapsed=time.monotonic()-tic,meta=meta,metrics=score(data,expand_prediction(data,pred),start,stop)['summary'])
      if previous is not None:row['previous_difference_per_frame_mm']=(np.sqrt(np.mean((previous[start:stop]-pred[start:stop])**2,axis=(1,2)))*1000).tolist()
      if name=='freeflight':
       t=data['times'][start:stop]-data['times'][start-1];centroid=np.einsum('n,nc->c',mass,state)[None]+t[:,None]*np.einsum('n,nc->c',mass,velocity)[None];centroid[:,2]-=.5*9.81*t*t
       row['ballistic_centroid_error_per_frame_mm']=(np.linalg.norm(np.einsum('n,tnc->tc',mass,pred[start:stop])-centroid,axis=1)*1000).tolist()
      report['variants'][name].append(row);previous=pred;print(name,steps,'time',row['elapsed'],'rmse',row['metrics'].get('coordinate_rmse_mm'),'fail',meta['failed'],flush=True)
      Path('runs/contact-benchmark-v12/scene-fem-isolation.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
