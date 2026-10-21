"""Tiny official renderer/kernel API smoke. No optimizer, data or training."""
from pathlib import Path
import sys,os,json,hashlib,traceback,subprocess
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'upstream'))
result={'role':'synthetic official API render/backward/kernel smoke; no training or dataset','CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'tests':[]}
try:
    import torch,numpy as np
    from PIL import Image
    from types import SimpleNamespace
    from gaussian_renderer import render
    from scene_PIDG.gaussian_model import GaussianModel
    from scene_PIDG.cameras import MiniCam
    from utils.graphics_utils import BasicPointCloud,getProjectionMatrix
    from hashencoder import HashEncoder
    torch.manual_seed(1701);np.random.seed(1701);torch.set_num_threads(2)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
    result.update(torch_version=torch.__version__,torch_cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),device_count=torch.cuda.device_count(),physical_gpu_query=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,pci.bus_id,name','--format=csv,noheader'],text=True).strip())
    xyz=np.array([[x,y,2.0] for x in [-.25,0,.25] for y in [-.25,0,.25]],dtype=np.float32)
    colors=np.array([[.2+.6*(i%3)/2,.25+.5*(i//3)/2,.6] for i in range(9)],dtype=np.float32)
    pc=GaussianModel(0);pc.create_from_pcd(BasicPointCloud(points=xyz,colors=colors,normals=np.zeros_like(xyz)),1.0)
    view=torch.eye(4,device='cuda');proj=getProjectionMatrix(znear=.01,zfar=10,fovX=1.0,fovY=1.0).transpose(0,1).cuda()
    cam=MiniCam(32,32,1.0,1.0,.01,10,view,view@proj)
    pipe=SimpleNamespace(debug=True,compute_cov3D_python=False,convert_SHs_python=False)
    bg=torch.zeros(3,device='cuda')
    for flow_grad in [False,True]:
        for p in [pc._xyz,pc._features_dc,pc._opacity,pc._scaling,pc._rotation]:p.grad=None
        out=render(cam,pc,pipe,bg,0,0,0,enable_flow_grad=flow_grad)
        image=out['render'];assert list(image.shape)==[3,32,32] and torch.isfinite(image).all()
        assert image.max().item()>0 and out['visibility_filter'].any()
        image.square().mean().backward();torch.cuda.synchronize()
        gradients={}
        for name in ['_xyz','_features_dc','_opacity','_scaling','_rotation']:
            grad=getattr(pc,name).grad;assert grad is not None and torch.isfinite(grad).all(),name
            gradients[name]={'finite':True,'absolute_sum':grad.abs().sum().item()}
        assert gradients['_xyz']['absolute_sum']>0 and gradients['_opacity']['absolute_sum']>0
        pixels=(image.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        path=root/'metadata'/('render-flow-'+str(flow_grad).lower()+'.png');Image.fromarray(pixels).save(path)
        result['tests'].append({'name':'official-render-forward-backward','enable_flow_grad':flow_grad,'shape':list(image.shape),'visible_gaussians':int(out['visibility_filter'].sum().item()),'range':[image.min().item(),image.max().item()],'gradients':gradients,'image_path':str(path),'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    enc=HashEncoder(input_dim=3,num_levels=2,level_dim=2,base_resolution=4,log2_hashmap_size=8,desired_resolution=8).cuda()
    x=torch.tensor([[-.4,-.2,.1],[.2,.3,-.1],[.4,-.3,.2]],device='cuda',requires_grad=True)
    y=enc(x);assert torch.isfinite(y).all()
    gx=torch.autograd.grad(y.square().sum(),x,create_graph=True)[0];assert torch.isfinite(gx).all()
    gx.square().sum().backward();torch.cuda.synchronize()
    assert enc.embeddings.grad is not None and torch.isfinite(enc.embeddings.grad).all()
    result['tests'].append({'name':'official-hashencoder-forward-first-second-backward','shape':list(y.shape),'finite':True,'first_gradient_abs_sum':gx.abs().sum().item(),'embedding_gradient_abs_sum':enc.embeddings.grad.abs().sum().item()})
    v,stress=pc.get_velocity_and_stress(torch.tensor(.3,device='cuda'));assert list(v.shape)==[9,3] and list(stress.shape)==[9,6] and torch.isfinite(v).all() and torch.isfinite(stress).all()
    result['tests'].append({'name':'restored-official-velocity-stress-network','velocity_shape':list(v.shape),'stress_shape':list(stress.shape),'finite':True})
    result.update(all_passed=True,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
except Exception as e:
    result.update(all_passed=False,error=type(e).__name__+': '+str(e));traceback.print_exc()
(root/'metadata/render-smoke.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));sys.exit(0 if result['all_passed'] else 1)
