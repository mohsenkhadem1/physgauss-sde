"""CPU-only preparation from existing strict prefix archives; no target loader."""
import json
from pathlib import Path
import numpy as np
import torch
from scripts.gaussian_boundary_flow_v02.export_inputs import reduce_geometry,write_package,load_package,sha
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import validate_predictor,predictor_digest
from scripts.gaussian_boundary_flow_v02.affine_observation import affine_observation_matrix
from scripts.gaussian_boundary_flow_v02.object_backend_geometry import GeometryAwareGaussianBoundaryBackend

ROOT=Path('reports/gaussian-boundary-flow-v02');OUT=Path('runs/gaussian-boundary-flow-v05-phystwin-inputs')
CASES=['single_lift_dinosor','double_lift_sloth','double_stretch_sloth','double_stretch_zebra']

def write(path,value):
 temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
 plans={case:Path('runs/real-transfer-v19/inputs/incoming-v22')/(case+'-240')for case in CASES}
 plan={'cases':CASES,'source':'Existing incoming-v22 strict prefix240 archives only; no raw pickle/evaluator access','geometry_policy':'Initial-only filtered64, then120ONLY if64constructor rejects disconnected/unused/nonpositive-mass nodes. Other errors remain terminal. No alternatecase or convex fallback.','device':'cpu','new_motion_fitting':False,'suffix_target_access':False,
       'source_prefix_manifest_sha256':{case:sha(path/'manifest.json')for case,path in plans.items()},
       'source_code_sha256':{str(path):sha(path)for path in [Path(__file__),Path('scripts/gaussian_boundary_flow_v02/object_backend_geometry.py'),Path('scripts/external_benchmark/fem.py'),Path('scripts/gaussian_boundary_flow_v02/export_inputs.py'),Path('scripts/gaussian_boundary_flow_v02/affine_observation.py')]}}
 planpath=ROOT/'phystwin-v05-geometry-plan.json'
 if planpath.exists():raise FileExistsError('Preparation plan already exists; do not silently rerun')
 write(planpath,plan);rows=[]
 for case,source in plans.items():
  meta=json.loads((source/'manifest.json').read_text())
  if meta['mask_version']!='incoming-v22' or meta['evaluation_geometry_exported']is not False or meta['force_labels_exported']is not False:raise ValueError('Invalid strict prefix manifest')
  if sha(source/'prefix.npz')!=meta['prefix_sha256']:raise ValueError('Changed strict prefix archive')
  with np.load(source/'prefix.npz',allow_pickle=False)as f:data={key:f[key].copy()for key in f.files}
  data.update(name=case,dataset='PhysTwin',train_end=meta['train_end'],floor=meta['floor'],mask_version=meta['mask_version'],
              source_prefix_archive=str((source/'prefix.npz').resolve()),source_prefix_archive_sha256=meta['prefix_sha256'],
              source_prefix_manifest_sha256=sha(source/'manifest.json'),historical_exposure='Historically exposed PhysTwin plush; retrospective development',
              observation_contract='Estimated prefix3Dtracks and supplied initial full geometry; no new evaluator or force labels',
              units='SI metres/seconds; complete known controller position schedules')
  validate_predictor(data);row={'case':case,'source_nodes':len(data['nodes']),'prefix_frames':data['train_end'],'total_command_frames':len(data['times']),'observed_points':data['observed'].shape[1],'controller_groups':data['controller_targets'].shape[1],'source_prefix_sha256':meta['prefix_sha256'],'attempts':[],'eligible':False}
  for nodes in (64,120):
   reduced=reduce_geometry(data,nodes);reduced['observation_matrix'],audit=affine_observation_matrix(reduced['nodes'],reduced['observed_initial'])
   reduced['observation_map_revision']={'rule':'Initial-only affine-complete local observation map','maximum_row_l1':audit['maximum_row_l1'],'maximum_row_l2':audit['maximum_row_l2'],'negative_weights_allowed':True}
   package=OUT/(case+'-nodes'+str(nodes));write_package(reduced,package);persisted=load_package(package)
   assert predictor_digest(persisted)==predictor_digest(reduced)
   attempt={'nodes':nodes,'package':str(package),'manifest_sha256':sha(package/'manifest.json'),'predictor_digest':predictor_digest(persisted),'affine_map_diagnostic':audit}
   try:
    backend=GeometryAwareGaussianBoundaryBackend(persisted,'cpu')
    mass=backend.model.mass.detach().cpu().numpy()
    if not np.isfinite(mass).all()or not(mass>0).all():raise ValueError('Nonfinite/nonpositive node masses')
    attempt.update(state='qualified_initial_geometry',diagnostic=backend.geometry_diagnostic,minimum_normalized_node_mass=float(mass.min()),maximum_normalized_node_mass=float(mass.max()))
    row.update(eligible=True,package=str(package),nodes=nodes,geometry='filtered',selected_manifest_sha256=sha(package/'manifest.json'),selected_predictor_digest=predictor_digest(persisted))
   except (ValueError,RuntimeError)as error:
    message=str(error);attempt.update(state='failed_initial_geometry',error=message)
    may_repair=nodes==64 and ('disconnected or has unused nodes'in message or'nonpositive'in message)
    attempt['eligible_for_predeclared120repair']=may_repair
   row['attempts'].append(attempt)
   if row['eligible']or not attempt.get('eligible_for_predeclared120repair',False):break
  rows.append(row);write(ROOT/'phystwin-v05-geometry-results.json',{'plan_sha256':sha(planpath),'rows':rows,'all4processed':len(rows)==4,'suffix_target_access':False,'device':'cpu'})
  print(case,row['eligible'],row.get('nodes'),[a['state']for a in row['attempts']],flush=True)
 lines=['Initial-only PhysTwin v05 geometry preparation; CPU only, strict prefix archives, no suffix targets.']
 for r in rows:
  lines.append(r['case']+': '+('READY '+r['package']if r['eligible']else'NOT QUALIFIED'))
  for a in r['attempts']:lines.append('  '+str(a['nodes'])+' nodes: '+a['state']+' '+a.get('error',''))
 (ROOT/'phystwin-v05-geometry-results.txt').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
