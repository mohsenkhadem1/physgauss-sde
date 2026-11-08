"""Bounded public tar inventory; preserves only small checkpoint-like members."""
import requests,tarfile,json,io,time
from pathlib import Path
class Limited:
 def __init__(self,raw,cap):self.raw=raw;self.cap=cap;self.count=0
 def read(self,n=-1):
  if self.count>=self.cap:raise RuntimeError('Declared256MiB compressed inventory budget reached')
  b=self.raw.read(min(n if n>=0 else self.cap,self.cap-self.count));self.count+=len(b);return b
root=Path('runs/gap-crossmethod-v02/mphygs-native/assets');root.mkdir(parents=True,exist_ok=True)
u='https://drive.usercontent.google.com/download?id=1tTHDllwRoH4NyQICTRCxbY4lyKjwhYco&export=download&confirm=t';r=requests.get(u,stream=True,timeout=60);r.raise_for_status();f=Limited(r.raw,256*1024**2);members=[];error=None
try:
 with tarfile.open(fileobj=f,mode='r|gz') as tar:
  for m in tar:
   members.append({'name':m.name,'size':m.size});print(m.name,m.size,flush=True)
   if m.isfile() and ('material_field.ckpt' in m.name or m.name.endswith('config.json')) and m.size<32*1024**2:
    out=root/Path(m.name).name;out.write_bytes(tar.extractfile(m).read())
except Exception as e:error=repr(e)
r.close();Path('reports/gap-crossmethod-v02/mphygs-archive-probe.json').write_text(json.dumps({'url':u,'total_archive_bytes':int(r.headers['Content-Length']),'compressed_bytes_read':f.count,'members':members,'stopped_reason':error,'complete_inventory':error is None},indent=2))
