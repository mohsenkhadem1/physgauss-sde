"""Read-only fixed common-time full-field CPU/GPU comparison, no solves."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from scipy import sparse
from experiments.campaign_newton_stable_material import StableP2Block
from experiments.campaign_p2_torch import TorchP2
from experiments.run_p2_consistent_transient import query_operator
ROOT=Path(__file__).resolve().parents[1]
def record(path):
 p=ROOT/path;return {'path':path,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def run():
 cpu='runs/gaussian-campaign-v01/p2-consistent-temporal-refinement-v02/tets96-smooth_ramp-hz1920.npz';gpu='runs/gaussian-campaign-v01/p2-gpu-refinement-v01/tets96-hz1920.npz';cp='runs/operator-dev/p2-consistent-temporal-refinement-v02/protocol.json';gp='runs/operator-dev/p2-gpu-refinement-v01/protocol.json';cj='runs/gaussian-campaign-v01/p2-consistent-temporal-refinement-v02/tets96-smooth_ramp-hz1920.json';gj='runs/gaussian-campaign-v01/p2-gpu-refinement-v01/tets96-hz1920.json'
 cs=json.loads((ROOT/cj).read_text());gs=json.loads((ROOT/gj).read_text())
 if not cs['completed'] or not gs['completed'] or record(cpu)['sha256']!=cs['capture']['sha256'] or record(gpu)['sha256']!=gs['capture']['sha256']:raise ValueError('Completed hashed reference pair required')
 for p in [cp,gp]:
  for f,h in json.loads((ROOT/p).read_text())['source_sha256'].items():
   if record(f)['sha256']!=h:raise ValueError('Changed source '+f)
 with np.load(ROOT/cpu,allow_pickle=False) as a,np.load(ROOT/gpu,allow_pickle=False) as b:
  for k in ['rest_m','elements','tets','faces','constraint_mask']:np.testing.assert_array_equal(a[k],b[k])
  idx=a['common_state_indices'];times=a['times_s'][idx];clock_error=float(np.max(abs(times-b['times_s'])))
  if clock_error>64*np.finfo(float).eps:raise ValueError('Unmatched common physical clocks')
  P=a['positions_m'][idx];V=a['velocities_m_s'][idx];position_error=np.linalg.norm(P-b['positions_m'],axis=-1);velocity_error=np.linalg.norm(V-b['velocities_m_s'],axis=-1)
  m=StableP2Block((4,2,2),young=5000.,poisson=.35,density=625.,gravity=0.,traction=1.,order=3);t=TorchP2(m,'cpu');M=sparse.csr_matrix((a['mass_csr_data'],a['mass_csr_indices'],a['mass_csr_indptr']),shape=tuple(a['mass_shape']));local=t.mass_local.numpy();T=m.elements
  reconstructed=sparse.coo_matrix((local.ravel(),(np.repeat(T,10,1).ravel(),np.tile(T,(1,10)).ravel())),shape=M.shape).tocsr();mass_delta=M-reconstructed
  probe_operator=query_operator(m,b['material_probes_m']);cpu_probes=np.stack([b['material_probes_m']+probe_operator@u for u in a['displacements_m'][idx]])
  clocks=np.arange(1,193)*(1/1920);scales=np.sin((np.pi/2)*np.minimum(clocks/.05,1))**2;source_commands=scales[:,None,None]*t.force.numpy();command_error=float(np.max(abs(source_commands-a['force_commands_N'])))
  ledger_clocks=np.array([row['time_s'] for row in gs['ledger']]);state_clock_error=float(np.max(abs(ledger_clocks-a['times_s'][1:])))
  out={'schema':'C38-independent-CPU-GPU-P2-reference-comparison-v0.1','completed':True,'comparison_kind':'POSTHOC descriptive same-discretization/same1920Hz/sameforce-law comparison; no new admission threshold','source':record('experiments/compare_p2_cpu_gpu_reference.py'),'evidence':[record(p) for p in [cpu,gpu,cp,gp,cj,gj]],'physics':'E5000/nu.35/rho625,96P2tetsQ3,fullconsistentT10mass,fullleftclamp,1Nsin²ramp.05s,BE-startBDF2,.1s','nodes':len(m.rest),'common_frames':25,'geometry_connectivity_and_clamp_bitwise_identical':True,'common_clock_max_difference_s':clock_error,'all_actual_ledger_clock_max_difference_s':state_clock_error,'allnode_max_Euclidean_position_difference_m':float(position_error.max()),'allnode_position_coordinate_RMS_m':float(np.sqrt(np.mean((P-b['positions_m'])**2))),'allnode_max_Euclidean_velocity_difference_m_s':float(velocity_error.max()),'allnode_velocity_coordinate_RMS_m_s':float(np.sqrt(np.mean((V-b['velocities_m_s'])**2))),'GPU63probe_max_Euclidean_position_difference_m':float(np.linalg.norm(cpu_probes-b['probe_positions_m'],axis=-1).max()),'mass_CSR_max_entry_difference_kg':float(np.max(abs(mass_delta.data))) if mass_delta.nnz else 0.,'CPU_total_mass_kg':float(M.sum()),'GPU_reconstructed_total_mass_kg':float(reconstructed.sum()),'GPU_pinned_recipe_vs_CPU_saved_force_max_component_difference_N':command_error,'force_evidence_limit':'GPU force array is reconstructed from pinned source recipe and actual model force, not a stored per-step GPU command getter','CPU_max_original_physical_R_N':cs['maximum_physical_R_N'],'GPU_max_original_physical_R_N':gs['maximum_physical_R_N'],'spatial_time_or_training_qualification':False,'limits':['CPU1920 does not supply a96tet successive-refinement temporal pass','Current meshes are not spatially qualified dynamic data','No physics calls, fitting, GPU calls or response-based selection occurred']}
 return out
if __name__=='__main__':
 torch.set_num_threads(2);out=ROOT/'runs/operator-dev/p2-cpu-gpu-reference-comparison-v01';out.mkdir(parents=True,exist_ok=False);result=run();p=out/'comparison.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['evidence','limits']},indent=2))
