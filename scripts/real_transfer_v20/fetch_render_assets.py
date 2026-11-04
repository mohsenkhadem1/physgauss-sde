"""Small fixed-frame rendering audit subset; licensed/raw assets stay local."""
import argparse,hashlib,json,urllib.request,zlib
from pathlib import Path
import numpy as np
from remotezip import RemoteZip
from .development_run import ROOT,BASE,write
from scripts.real_transfer_v19.phystwin_campaign import CASES

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cases',nargs='+',default=CASES);args=parser.parse_args()
    destination=BASE/'render-assets';destination.mkdir(exist_ok=True)
    planpath=destination/'snapshot.json'
    if planpath.exists():revision=json.loads(planpath.read_text())['revision']
    else:
        with urllib.request.urlopen('https://huggingface.co/api/datasets/Jianghanxiao/PhysTwin',timeout=30) as f:revision=json.load(f)['sha']
        write(planpath,dict(repository='Jianghanxiao/PhysTwin',revision=revision,selection='Frame zero and three evenly spaced official test frames, all three cameras; final released Gaussian appearance.'))
    records=[]
    def fetch(z,url,name):
        path=destination/name;info=z.getinfo(name)
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True);payload=z.read(name)
            tmp=path.with_suffix(path.suffix+'.partial');tmp.write_bytes(payload);tmp.replace(path)
        payload=path.read_bytes()
        if len(payload)!=info.file_size or zlib.crc32(payload)!=info.CRC:raise ValueError('Archive member mismatch: '+name)
        records.append(dict(path=str(path.relative_to(ROOT)),member=name,archive=url,size=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
        return payload
    url=f'https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/{revision}/data.zip'
    with RemoteZip(url) as z:
        names=set(z.namelist())
        for case in args.cases:
            prefix=f'data/different_types/{case}/'
            split=json.loads((ROOT/'runs/third_party/PhysTwin'/prefix/'split.json').read_text())
            frames=sorted(set([0]+np.linspace(split['train'][1],split['test'][1]-1,3,dtype=int).tolist()))
            for camera in range(3):
                fetch(z,url,prefix+f'mask/mask_info_{camera}.json')
                for frame in frames:
                    fetch(z,url,prefix+f'color/{camera}/{frame}.png')
                    masks=sorted(n for n in names if n.startswith(prefix+f'mask/{camera}/') and n.endswith(f'/{frame}.png'))
                    for name in masks:fetch(z,url,name)
            print('images',case,frames,flush=True)
            write(destination/f'{case}-frames.json',dict(case=case,frames=frames,cameras=[0,1,2],scope='Sparse fixed-frame audit, not full-video rendering evaluation or novel-view test. Appearance trained on the initial three cameras.'))
    url=f'https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/{revision}/gaussian_output.zip'
    with RemoteZip(url) as z:
        names=z.namelist()
        for case in args.cases:
            selected=[n for n in names if n.startswith(f'gaussian_output/{case}/') and (n.endswith('/iteration_10000/point_cloud.ply') or n.endswith('/cfg_args'))]
            if len(selected)!=2:raise ValueError('Ambiguous Gaussian checkpoint')
            for name in selected:fetch(z,url,name)
            print('gaussians',case,flush=True)
    write(destination/'download-manifest.json',dict(revision=revision,records=records))

if __name__=='__main__':main()
