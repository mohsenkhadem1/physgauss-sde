"""Score sealed common-appearance images; no model fitting or alignment."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from scripts.complex_scenes_v18.shared_appearance import AUTHOR
from scripts.complex_scenes_v18.run_development import ROOT,write
from scripts.compositional_dynamics_v17.render_metrics import compare_renderings


def verified(folder):
    folder=Path(folder);seal=json.loads((folder/'images-seal.json').read_text())
    if not seal['completed']:raise ValueError('Unsealed image prediction')
    keys=set()
    for row in seal['records']:
        key=(row['frame'],row['camera']['name'])
        if key in keys:raise ValueError('Duplicate image slot')
        keys.add(key)
        if hashlib.sha256((folder/row['path']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Changed rendered image')
    return seal


def score(prediction,reference):
    prediction=Path(prediction);reference=Path(reference)
    pred=verified(prediction)  # Complete prediction census before reference access.
    ref=verified(reference)
    if pred['appearance_sha256']!=ref['appearance_sha256']or pred['public_query']!=ref['public_query']:raise ValueError('Shared appearance/query required')
    refs={(r['frame'],r['camera']['name']):r for r in ref['records']}
    if set(refs)!={(r['frame'],r['camera']['name'])for r in pred['records']}:raise ValueError('Exact matching image census required')
    sys.path.insert(0,str(AUTHOR))
    from gaussian_splatting.lpipsPyTorch import LPIPS
    perceptual=LPIPS('alex','0.1').cuda().eval();digest=hashlib.sha256()
    for key,value in sorted(perceptual.state_dict().items()):digest.update(key.encode());digest.update(value.detach().cpu().numpy().tobytes())
    rows=[];self_error=0.
    with torch.no_grad():
        for row in pred['records']:
            other=refs[(row['frame'],row['camera']['name'])]
            if row['camera']!=other['camera'] or abs(row['time_s']-other['time_s'])>1e-9:raise ValueError('Identical cameras and physical times required')
            with np.load(prediction/row['path'])as p,np.load(reference/other['path'])as r:
                a={k:p[k]for k in p.files};b={k:r[k]for k in r.files}
            a['camera']=row['camera'];b['camera']=other['camera'];result=compare_renderings(a,b)
            def tensor(rgb):return torch.tensor(rgb,dtype=torch.float32,device='cuda').permute(2,0,1)[None]*2-1
            x,y=tensor(a['rgb']),tensor(b['rgb']);lp=float(perceptual(x,y).item())
            if not rows:self_error=float(perceptual(x,x).abs().max())
            result.update(frame=row['frame'],camera_name=row['camera']['name'],lpips_alex_v01=lp);rows.append(result)
    if self_error>1e-7:raise ValueError('Perceptual identity check failed')
    names=['psnr_global_db','psnr_foreground_union_db','ssim_rgb_global','ssim_rgb_foreground_union','silhouette_iou','lpips_alex_v01']
    return dict(scope='Object-only shared appearance and transport; not independent native reconstruction quality. Mean over fixed sampled frames and views, not independent statistical samples.',appearance_sha256=pred['appearance_sha256'],lpips_state_sha256=digest.hexdigest(),lpips_identity_error=self_error,means={key:float(np.mean([r[key]for r in rows if r[key]is not None]))for key in names},rows=rows,prediction_images_seal_sha256=hashlib.sha256((prediction/'images-seal.json').read_bytes()).hexdigest(),reference_images_seal_sha256=hashlib.sha256((reference/'images-seal.json').read_bytes()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);args=p.parse_args();torch.set_num_threads(1)
    if Path(args.output).name!=args.output:raise ValueError('Report basename required')
    result=score(args.prediction,args.reference);write(ROOT/'reports/complex-scenes-v18'/args.output,result);print(json.dumps(result['means']))
