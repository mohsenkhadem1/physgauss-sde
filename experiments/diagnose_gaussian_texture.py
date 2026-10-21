"""Public-RGB texture calibration/observability audit; no trajectory labels."""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import hashlib,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from train_gaussian_inverse import collect_cases,apply_catalog
from gaussian_texture_features import static_kernel_measurements,invert_color_curve,frame_texture_features


def audit(roots,catalog,calibration_path,output):
    output=Path(output)
    if output.exists():raise FileExistsError('Diagnostics are immutable; choose a fresh output')
    output.mkdir(parents=True);(output/'provenance').mkdir()
    for name in ('gaussian_texture_features.py','diagnose_gaussian_texture.py'):
        shutil.copyfile(Path(__file__).with_name(name),output/'provenance'/name)
    calibration=json.loads(Path(calibration_path).read_text());cases=apply_catalog(collect_cases(roots),catalog)
    rows=[]
    for case in cases:
        episode=case['path'];split=case['recipe']['split']
        with np.load(episode/'reference.npz') as raw:reference={k:raw[k] for k in raw.files}
        with np.load(episode/'observations/cameras.npz') as raw:c={k:raw[k] for k in raw.files}
        # TRAIN only frame0: calibration adequacy, not further motion fitting.
        indices=[0] if split=='train' else range(len(c['times_s']))
        features=[]
        for index in indices:
            frame=int(c['frame_ids'][index]);pose=c['camera_to_world_usd_row'][index]
            rgb=np.asarray(Image.open(episode/'observations/rgb'/f'{frame:05d}.png').convert('RGB'))/255.
            mask=np.asarray(Image.open(episode/'observations/masks'/f'{frame:05d}.png').convert('L'))>0
            measured=static_kernel_measurements(reference,pose,c['K'],rgb,mask)
            feature,face_audits=frame_texture_features(reference,pose,c['K'],rgb,mask,calibration)
            features.append(feature);faces=[];jacobians=[]
            for face in range(6):
                selected=(measured['face']==face)&measured['valid']
                if selected.sum()<5:continue
                curve=calibration['face_curves'].get(str(face),calibration['global_curve'])
                w,error=invert_color_curve(curve,measured['observed_rgb'][selected])
                gate=error<max(.06,4*curve['training_rgb_rmse']);ids=np.flatnonzero(selected)[gate]
                if len(ids)<5:continue
                delta=w[gate]-measured['w_reference'][ids];jacobians.append(measured['projective_gradient_world'][ids])
                f=feature[face*13:(face+1)*13]
                faces.append({'face':face,'samples':len(ids),'w_residual_rmse':float(np.sqrt(np.mean(delta**2))),
                    'apparent_shift_norm_mm':float(np.linalg.norm(f[1:3])*.01*1000),
                    'rgb_curve_distance_mean':float(error[gate].mean()),'rank':face_audits[face]['rank']})
            combined=np.concatenate(jacobians) if jacobians else np.zeros((1,3))
            eigen=np.linalg.eigvalsh(combined.T@combined/len(combined))
            rows.append({'key':case['key'],'split':split,'frame':frame,'time_s':float(c['times_s'][index]),
                         'faces':faces,'projective_eigenvalues_m_minus2':eigen.tolist()})
        np.savez_compressed(output/(case['key'].replace('/','__')+'.npz'),features=np.asarray(features),
                            frame_ids=np.asarray(c['frame_ids'])[list(indices)])
    summary={}
    for split in ('train','validation'):
        for phase in ('reference','later'):
            chosen=[r for r in rows if r['split']==split and (r['frame']==0)==(phase=='reference')]
            face_rows=[f for r in chosen for f in r['faces']]
            if not face_rows:continue
            values=np.array([r['apparent_shift_norm_mm'] for r in face_rows])
            summary[split+'_'+phase]={'frames':len(chosen),'face_measurements':len(values),
                'apparent_shift_norm_median_mm':float(np.median(values)),
                'apparent_shift_norm_p90_mm':float(np.quantile(values,.9)),
                'per_face':{str(f):{'measurements':len(v),'median_shift_mm':float(np.median(v))}
                    for f in range(6) if len(v:=[r['apparent_shift_norm_mm'] for r in face_rows if r['face']==f])}}
    result={'summary':summary,'rows':rows,'catalog_sha256':json.loads(Path(catalog).read_text())['catalog_sha256'],
            'calibration_sha256':hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest(),
            'qualification':'Reference-frame values diagnose zero-motion bias; later values are apparent shifts, not trajectory errors. No depth, truth motion, hidden visibility or TEST files read.'}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roots',nargs='+',required=True)
    p.add_argument('--catalog',required=True);p.add_argument('--calibration',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=audit(a.roots,a.catalog,a.calibration,a.output);print(json.dumps(result['summary'],indent=2))
