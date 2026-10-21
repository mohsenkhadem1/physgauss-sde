"""One bounded finite-difference control of the released official renderer."""
from pathlib import Path
import sys,os,json,traceback
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'upstream'))
result={'role':'synthetic RGB-only derivative diagnostic; no dataset/training/source repair','CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'completed':False,'rows':[]}
try:
    import torch,numpy as np
    from types import SimpleNamespace
    from gaussian_renderer import render
    from scene_PIDG.gaussian_model import GaussianModel
    from scene_PIDG.cameras import MiniCam
    from utils.graphics_utils import BasicPointCloud,getProjectionMatrix
    assert torch.cuda.device_count()==1 and os.environ['CUDA_VISIBLE_DEVICES']=='1'
    torch.manual_seed(1701);np.random.seed(1701);torch.set_num_threads(2)
    xyz=np.array([[x,y,2.] for x in [-.25,0,.25] for y in [-.25,0,.25]],dtype=np.float32)
    colors=np.array([[.2+.6*(i%3)/2,.25+.5*(i//3)/2,.6] for i in range(9)],dtype=np.float32)
    pc=GaussianModel(0);pc.create_from_pcd(BasicPointCloud(points=xyz,colors=colors,normals=np.zeros_like(xyz)),1.)
    view=torch.eye(4,device='cuda');proj=getProjectionMatrix(znear=.01,zfar=10,fovX=1.,fovY=1.).transpose(0,1).cuda();cam=MiniCam(32,32,1.,1.,.01,10,view,view@proj)
    pipe=SimpleNamespace(debug=True,compute_cov3D_python=False,convert_SHs_python=False);bg=torch.zeros(3,device='cuda')
    choices=[('_xyz',(0,0)),('_xyz',(0,2)),('_scaling',(0,0)),('_scaling',(4,0))];epsilons=[.003,.001,.0003]
    originals={name:getattr(pc,name).detach().clone() for name in ['_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation']}
    def reset():
        with torch.no_grad():
            for name,value in originals.items():getattr(pc,name).copy_(value);getattr(pc,name).grad=None
        for p in pc.velocity_net.parameters():p.grad=None
        pc.force_sigma.grad=None
    def forward(flag):
        out=render(cam,pc,pipe,bg,0,0,0,enable_flow_grad=flag);im=out['render'];assert torch.isfinite(im).all() and int(out['visibility_filter'].sum().item())==9
        return im.square().mean(),im
    base_images={}
    for flag in [False,True]:
        reset();loss,im=forward(flag);base_images[str(flag)]=im.detach().clone();loss.backward();torch.cuda.synchronize()
        base_loss=loss.item();grads={(name,index):getattr(pc,name).grad[index].item() for name,index in choices}
        for name,index in choices:
            row={'flow_flag':flag,'parameter':name,'index':list(index),'analytic':grads[(name,index)],'base_loss':base_loss,'central_differences':[]}
            for eps in epsilons:
                losses=[]
                for sign in [1,-1]:
                    reset()
                    with torch.no_grad():
                        getattr(pc,name)[index]=originals[name][index]+sign*eps;value,_=forward(flag);losses.append(value.item())
                fd=(losses[0]-losses[1])/(2*eps);analytic=row['analytic']
                row['central_differences'].append({'epsilon':eps,'plus_loss':losses[0],'minus_loss':losses[1],'derivative':fd,'analytic_over_fd':analytic/fd if abs(fd)>1e-12 else None,'relative_error_to_fd':abs(analytic-fd)/max(abs(fd),1e-12),'within_5_percent':abs(analytic-fd)<=.05*max(abs(fd),1e-12)})
            result['rows'].append(row)
    reset();torch.cuda.synchronize()
    result.update(completed=True,loss='mean(rendered_RGB squared), fixed black target',forward_images_bitwise_equal=torch.equal(base_images['False'],base_images['True']),epsilons=epsilons,all_derivatives_within_5_percent=all(d['within_5_percent'] for row in result['rows'] for d in row['central_differences']),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
except Exception as e:
    result['error']=type(e).__name__+': '+str(e);traceback.print_exc()
(root/'metadata/derivative-control.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));sys.exit(0 if result['completed'] else 1)
