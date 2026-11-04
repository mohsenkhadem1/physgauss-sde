"""Common observable metrics. No simulator energy is treated as ground truth."""
import numpy as np

def manual_response_metrics(predicted,truth,start,end):
    """Inputs already correspond by frame-zero matching; units are metres.

    Pair-distance changes remove rigid motion and initial matching offsets.
    Missing reference annotations are excluded; missing predictions are failures.
    These are standard geometric diagnostics, not claimed as novel metrics.
    """
    p=np.asarray(predicted);y=np.asarray(truth)
    if p.shape!=y.shape or p.ndim!=3 or p.shape[-1]!=3:raise ValueError('Track shape mismatch')
    rows=[]
    initial=np.isfinite(y[0]).all(1)
    for t in range(start,end):
        valid=initial & np.isfinite(y[t]).all(1)
        n=int(valid.sum())
        row=dict(frame=t,annotated_points=n)
        if not n:row['status']='no_annotation';rows.append(row);continue
        if not np.isfinite(p[[0,t]][:,valid]).all():
            row['status']='prediction_failure';rows.append(row);continue
        a=p[t,valid];b=y[t,valid];a0=p[0,valid];b0=y[0,valid]
        i,j=np.triu_indices(n,1)
        row.update(status='valid',tracking_mm=float(np.linalg.norm(a-b,axis=1).mean()*1000),
                   displacement_mm=float(np.linalg.norm((a-a0)-(b-b0),axis=1).mean()*1000))
        if len(i):
            change_p=np.linalg.norm(a[i]-a[j],axis=1)-np.linalg.norm(a0[i]-a0[j],axis=1)
            change_y=np.linalg.norm(b[i]-b[j],axis=1)-np.linalg.norm(b0[i]-b0[j],axis=1)
            row['pair_distance_change_mm']=float(np.abs(change_p-change_y).mean()*1000)
        rows.append(row)
    summary={key:float(np.mean([r[key] for r in rows if key in r])) if any(key in r for r in rows) else None
             for key in ('tracking_mm','displacement_mm','pair_distance_change_mm')}
    summary.update(total_frames=len(rows),valid_frames=sum(r['status']=='valid' for r in rows),
                   failed_frames=sum(r['status']=='prediction_failure' for r in rows),
                   unannotated_frames=sum(r['status']=='no_annotation' for r in rows))
    summary['complete_prediction']=summary['failed_frames']==0
    return dict(summary=summary,frames=rows)

def tetra_sanity(rest,predicted,elements):
    rest=np.asarray(rest);p=np.asarray(predicted);e=np.asarray(elements)
    dm=np.stack([rest[e[:,i]]-rest[e[:,0]] for i in (1,2,3)],axis=-1)
    determinant=np.linalg.det(dm)
    if np.any(np.abs(determinant)<1e-15):raise ValueError('Degenerate reference tetrahedron')
    ds=np.stack([p[:,e[:,i]]-p[:,e[:,0]] for i in (1,2,3)],axis=-1)
    J=np.linalg.det(ds)/determinant[None]
    finite=np.isfinite(J).all(1)
    return dict(finite_frames=int(finite.sum()),total_frames=len(p),unused_nodes=int(len(rest)-len(np.unique(e))),
                minimum_volume_ratio=float(J[finite].min()) if finite.any() else None,
                inverted_element_frame_fraction=float((J[finite]<=0).mean()) if finite.any() else None,
                scope='Numerical validity of this tetrahedral representation; not force accuracy or a cross-representation ranking.')

def foreground_render_metrics(actual,predicted,reference_mask,lpips_model=None):
    """Fixed reference-mask bounding box, black background; optional frozen LPIPS.

    Do not derive the crop from a method's own prediction. Pass linear-consistent
    RGB arrays in [0,1]. Camera/exposure/rendering policy must be shared.
    """
    from skimage.metrics import structural_similarity
    a=np.asarray(actual,float);b=np.asarray(predicted,float);mask=np.asarray(reference_mask,bool)
    if a.shape!=b.shape or a.shape[:2]!=mask.shape or a.shape[-1]!=3:raise ValueError('Image shape mismatch')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite image')
    if min(a.min(),b.min())<0 or max(a.max(),b.max())>1:raise ValueError('Expected RGB in [0,1]')
    yy,xx=np.where(mask)
    if not len(xx):raise ValueError('Empty reference mask')
    y0,y1=max(0,int(yy.min())-8),min(len(mask),int(yy.max())+9)
    x0,x1=max(0,int(xx.min())-8),min(mask.shape[1],int(xx.max())+9)
    # Inputs must already have the common background treatment. Keep predicted
    # spill outside the object in this crop rather than masking it away.
    a=a[y0:y1,x0:x1];b=b[y0:y1,x0:x1];mse=float(np.mean((a-b)**2))
    result=dict(crop_xyxy=[x0,y0,x1,y1],psnr_db=float(-10*np.log10(max(mse,1e-12))),
                ssim=float(structural_similarity(a,b,data_range=1.,channel_axis=-1)),lpips=None)
    if lpips_model is not None:
        import torch
        device=next(lpips_model.parameters()).device
        tensor=lambda v:torch.as_tensor(v.transpose(2,0,1).copy(),dtype=torch.float32,device=device)[None]*2-1
        with torch.no_grad():result['lpips']=float(lpips_model(tensor(a),tensor(b)).item())
    return result
