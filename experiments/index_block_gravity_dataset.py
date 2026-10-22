"""Hash-index local development captures without publishing raw arrays/media."""
import argparse,hashlib,json
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(output):
    native=Path('runs/gaussian-campaign-v01/isaac-block-gravity-v01');dense=Path('runs/gaussian-campaign-v01/isaac-block-dense-drop-v01');video=Path('runs/gaussian-campaign-v01/block-support-gravity-videos-v01')
    execution=json.loads((native/'execution.json').read_text());assert execution['terminal'] and execution['completed'] and len(execution['cells'])==8
    dex=json.loads((dense/'execution.json').read_text());assert dex['terminal'] and dex['completed']
    rows=[]
    for r in execution['cells']:
        p=native/r['id'];summary=json.loads((p/'summary.json').read_text());assert sha(p/'summary.json')==r['summary_sha256'];assert sha(p/'trajectory.npz')==r['trajectory_sha256']
        case=summary['case'];use_dense=case['kind']=='drop';vpath=video/(case['id']+('-dense' if use_dense else ''))/'video.json';v=json.loads(vpath.read_text())
        captured=(dense if use_dense else native)/case['id']/'trajectory.npz';assert sha(captured)==v.get('native_source_sha256',v.get('source_sha256'))
        rows.append({'case':case,'density_kg_m3':625.,'poisson':.35,'gravity_m_s2':[0,0,-9.81],'geometry_m':[.24,.08,.08],'physics_hz':240,'nodes':325,'tetrahedra':1152,'fixed_nodes':summary['native_verification']['flags']['enabled_fixed_count'],'nominal_mass_kg':summary['nominal_solid_mass_kg'],'raw_capture':str(captured),'raw_capture_sha256':sha(captured),'raw_capture_frames':v.get('recorded_frames',summary['frames']),'video_manifest':str(vpath),'video_manifest_sha256':sha(vpath),'peak_proper_rigid_removed_coordinate_RMS_m':v['peak_nonrigid_coordinate_RMS_m'],'role':'freefall_control' if case['kind']=='freefall' else'development_demo','native_temporal_qualified':False,'contact_accuracy_qualified':False,'eligible_as_validated_physical_inference_reference':False})
    result={'schema':'local-block-support-gravity-development-index-v0.1','cases':rows,'count':8,'source_execution':{'path':str(native/'execution.json'),'sha256':sha(native/'execution.json')},'dense_observation_execution':{'path':str(dense/'execution.json'),'sha256':sha(dense/'execution.json')},'raw_arrays_and_media_included':False,'raw_files_available_on_originating_machine_only':True,'has_calibrated_RGBD_observations':False,'video_is_visualization_not_prediction':True,'not_a_sealed_generalization_test':True,'qualification':'Native time convergence fails; contact penetration remains. These scenarios are development captures, not a validated physical inference benchmark.'}
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(result,indent=2)+'\n');return {'cases':len(rows),'index':str(p),'sha256':sha(p)}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(build(a.output)))
