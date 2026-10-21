import importlib,json,sys,traceback,time
from pathlib import Path
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'upstream'))
import torch  # Normal official modules preload torch before native extension symbols.
rows=[]
for name in ['simple_knn._C','diff_gaussian_rasterization','hashencoder','scene_PIDG','gaussian_renderer','train_pidg','render','metrics']:
    start=time.time()
    try:
        m=importlib.import_module(name);rows.append({'module':name,'ok':True,'file':getattr(m,'__file__',None),'seconds':time.time()-start})
    except Exception as e:
        rows.append({'module':name,'ok':False,'error':type(e).__name__+': '+str(e),'seconds':time.time()-start});traceback.print_exc()
result={'role':'Official import/JIT build smoke only; no training, dataset or GPU workload','modules':rows,'all_passed':all(r['ok'] for r in rows)}
(root/'metadata/import-smoke.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));sys.exit(0 if result['all_passed'] else 1)
