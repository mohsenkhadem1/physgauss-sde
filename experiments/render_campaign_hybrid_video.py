"""LOCAL actual-versus-predicted paired-orbit video for the hybrid TRAIN pilot.

Only completed supplied-physics predictions are rendered. No simulator, GPU,
checkpoint inference, image conditioning or posterior construction is invoked.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
import imageio_ffmpeg


FPS=20
MARGIN_M=.08


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path,allow_pickle=False) as f:return {key:f[key].copy() for key in f.files}


def selected_pairs(ids,train_mask,requested):
    held=np.flatnonzero(~train_mask)
    if not len(held):raise ValueError('No held cross-action pairs in dataset')
    if requested is None:return [int(held[0])]
    if requested=='all':return held.tolist()
    result=[]
    for token in requested.split(','):
        token=token.strip();matches=np.flatnonzero(ids.astype(str)==token)
        index=int(matches[0]) if len(matches)==1 else int(token)
        if index not in held:raise ValueError('Render only declared held cross-action pairs: '+token)
        if index not in result:result.append(index)
    return result


def gaussian_transport(positions,reference,tetrahedra):
    """Uniform-tet physical covariance; no learned uncertainty or jitter."""
    rest=reference[tetrahedra];center0=rest.mean(1);offset=rest-center0[:,None]
    covariance0=np.einsum('eni,enj->eij',offset,offset)/20
    inverse=np.linalg.inv(np.swapaxes(rest[:,1:]-rest[:,:1],1,2))
    corners=positions[:,tetrahedra]
    F=np.swapaxes(corners[:,:,1:]-corners[:,:,:1],-1,-2)@inverse[None]
    return corners.mean(2),F@covariance0[None]@np.swapaxes(F,-1,-2),np.linalg.det(F)


def ellipsoid_lines(centers,covariances):
    values,vectors=np.linalg.eigh(covariances)
    valid=np.isfinite(values).all(1)&(values>0).all(1)
    angles=np.linspace(0,2*np.pi,21);circles=np.zeros((3,len(angles),3))
    for j,(a,b) in enumerate([(0,1),(0,2),(1,2)]):circles[j,:,a]=np.cos(angles);circles[j,:,b]=np.sin(angles)
    # Invalid shape covariances are visibly counted rather than repaired.
    root=vectors[valid]*np.sqrt(values[valid])[:,None,:]
    lines=centers[valid,None,None]+np.einsum('eij,csj->ecsi',root,circles)
    return lines.reshape(-1,len(angles),3),valid


class Panel:
    def __init__(self,ax,reference,faces,kernel_ids,all_tets,limits,title):
        self.ax=ax;self.faces=faces;self.kernel_ids=kernel_ids
        ref_centers=reference[all_tets].mean(1);span=max(float(np.ptp(reference[:,0])),1e-12)
        colors=cm.viridis(np.clip((ref_centers[:,0]-reference[:,0].min())/span,0,1))
        self.kernel_colors=colors[kernel_ids]
        face_color=cm.viridis(np.clip((reference[faces].mean(1)[:,0]-reference[:,0].min())/span,0,1))
        self.mesh=Poly3DCollection(reference[faces],facecolors=face_color,edgecolors='#31465b',linewidths=.22,alpha=.15)
        ax.add_collection3d(self.mesh)
        self.lines=Line3DCollection([],linewidths=.6,alpha=.73);ax.add_collection3d(self.lines,autolim=False)
        self.points=ax.scatter([],[],[],s=3,depthshade=False)
        self.status=ax.text2D(.02,.04,'',transform=ax.transAxes,fontsize=8,color='#9f2020')
        ax.set_title(title,fontsize=13,pad=3,color='#172c3e')
        for name,bounds in zip(['x','y','z'],limits):getattr(ax,'set_'+name+'lim')(bounds)
        ax.set_box_aspect(limits[:,1]-limits[:,0]);ax.set_proj_type('ortho')
        ax.set_xlabel('x (m)',fontsize=8,labelpad=-2);ax.set_ylabel('y (m)',fontsize=8,labelpad=-2);ax.set_zlabel('z (m)',fontsize=8,labelpad=-2)
        ax.tick_params(labelsize=7,pad=-1)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.fill=False
        ax.grid(False)

    def update(self,q,centers,covariance,time_s,inverted_count):
        self.mesh.set_verts(q[self.faces]);segments,valid=ellipsoid_lines(centers,covariance)
        self.lines.set_segments(segments);self.lines.set_color(np.repeat(self.kernel_colors[valid],3,axis=0))
        self.points._offsets3d=(centers[:,0],centers[:,1],centers[:,2]);self.points.set_color(self.kernel_colors)
        notes=[]
        if inverted_count:notes.append(str(inverted_count)+' inverted tetrahedra')
        if not valid.all():notes.append(str(int((~valid).sum()))+' invalid shape ellipsoids omitted')
        self.status.set_text('; '.join(notes))
        # Identical mild orbit, independent of either trajectory's amplitude.
        self.ax.view_init(elev=22+4*np.sin(np.pi*time_s/3),azim=-64+26*time_s/3)


def run(dataset,predictions,output,*,pairs=None,kernels=100):
    dataset,predictions,output=map(Path,[dataset,predictions,output])
    if output.exists():raise FileExistsError('Fresh LOCAL video output required')
    data=load(dataset);pred=load(predictions)
    ids=data['ids'];train=np.asarray(data['train_mask'],dtype=bool)
    if not np.array_equal(ids,pred['ids']) or not np.array_equal(train,pred['train_mask']):raise ValueError('Prediction/dataset pair identities differ')
    times=np.asarray(data['times'],dtype=float)
    if times.shape!=(61,) or not np.allclose(times,np.arange(61)/20,rtol=0,atol=2e-7) or not np.array_equal(data['times'],pred['times_s']):raise ValueError('Common61-frame0..3s clocks required')
    prediction=np.asarray(pred['predicted_positions_m'],dtype=float);truth=np.asarray(data['q'],dtype=float)
    if prediction.shape!=truth.shape or prediction.shape[:2]!=(len(ids),61) or not np.isfinite(prediction).all() or not np.isfinite(truth).all():raise ValueError('Complete finite actual/predicted trajectories required')
    chosen=selected_pairs(ids,train,pairs);faces=np.asarray(data['faces'],dtype=int);tets=np.asarray(data['tets'],dtype=int)
    if not 1<=kernels<=len(tets):raise ValueError('Kernel sample count out of bounds')
    kernel_ids=np.unique(np.linspace(0,len(tets)-1,kernels,dtype=int))
    output.parent.mkdir(parents=True,exist_ok=True);first_png=output.with_suffix('.first-frame.png')
    fig=plt.figure(figsize=(12.8,6.4),dpi=100,facecolor='#f7f9fc')
    caption=fig.text(.5,.945,'',ha='center',fontsize=11,color='#243c51')
    fig.text(.5,.905,'Held cross-action on known layouts | from rest | supplied physics | zero image observations',ha='center',fontsize=9,color='#46596a')
    fig.text(.5,.035,'Material ellipsoids show physical 1-sigma shape, NOT uncertainty. True scale; same fixed reference bounds + 0.08 m margin.',ha='center',fontsize=8,color='#46596a')
    fig.text(.5,.01,'Independent FEM reference; prediction is a forward diagnostic, not a posterior or native Isaac result.',ha='center',fontsize=8,color='#46596a')
    writer=imageio_ffmpeg.write_frames(str(output),(1280,640),fps=FPS,codec='libx264',quality=7,
        macro_block_size=2,ffmpeg_log_level='error',output_params=['-movflags','+faststart'])
    writer.send(None);metadata=[];frame_count=0;axes=[]
    try:
        for index in chosen:
            for ax in axes:ax.remove()
            reference=np.asarray(data['rest'][index],dtype=float)
            limits=np.stack([reference.min(0)-MARGIN_M,reference.max(0)+MARGIN_M],axis=1)
            axes=[fig.add_subplot(1,2,j+1,projection='3d',facecolor='#f7f9fc') for j in range(2)]
            fig.subplots_adjust(left=.015,right=.985,bottom=.085,top=.84,wspace=.01)
            panels=[Panel(axes[0],reference,faces,kernel_ids,tets,limits,'Actual · Independent FEM'),
                Panel(axes[1],reference,faces,kernel_ids,tets,limits,'Predicted · supplied physics')]
            actual_geometry=gaussian_transport(truth[index],reference,tets)
            predicted_geometry=gaussian_transport(prediction[index],reference,tets)
            errors=np.sqrt(np.mean((prediction[index]-truth[index])**2,axis=(1,2)))
            clipped=np.any((prediction[index]<limits[:,0])|(prediction[index]>limits[:,1]),axis=-1)
            for frame,time_s in enumerate(times):
                caption.set_text(str(ids[index])+'   |   t = '+format(time_s,'.2f')+' s   |   coordinate RMSE = '+format(errors[frame]*1000,'.3f')+' mm')
                for panel,q,geometry in zip(panels,[truth[index],prediction[index]],[actual_geometry,predicted_geometry]):
                    centers,covariances,J=geometry
                    panel.update(q[frame],centers[frame,kernel_ids],covariances[frame,kernel_ids],float(time_s),int((J[frame]<=0).sum()))
                if clipped[frame].any():panels[1].status.set_text(panels[1].status.get_text()+' | '+str(int(clipped[frame].sum()))+' vertices outside fixed bounds')
                fig.canvas.draw();rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
                if frame_count==0:plt.imsave(first_png,rgb)
                writer.send(rgb);frame_count+=1
            metadata.append({'index':index,'id':str(ids[index]),'held_cross_action':True,'reference_limits_m':limits.tolist(),
                'coordinate_rmse_m':float(np.sqrt(np.mean((prediction[index]-truth[index])**2))),
                'maximum_predicted_vertices_outside_fixed_bounds':int(clipped.sum(1).max()),
                'minimum_actual_J':float(actual_geometry[2].min()),'minimum_predicted_J':float(predicted_geometry[2].min())})
    finally:writer.close();plt.close(fig)
    # Decode the actual MP4 to verify frame count and first-frame dimensions.
    reader=imageio_ffmpeg.read_frames(str(output),pix_fmt='rgb24');video_info=next(reader);decoded=sum(1 for _ in reader)
    if decoded!=frame_count or tuple(video_info['size'])!=(1280,640) or abs(video_info['duration']-frame_count/FPS)>1/FPS:
        raise RuntimeError('Encoded video frame count/size/duration verification failed')
    result={'local_only':True,'dataset_sha256':sha(dataset),'predictions_sha256':sha(predictions),'renderer_sha256':sha(__file__),
        'prediction_model_label':predictions.parent.name,'predictions':str(predictions),
        'output':str(output),'video_sha256':sha(output),'bytes':output.stat().st_size,'first_frame_png':str(first_png),
        'fps':FPS,'frames':frame_count,'duration_s':video_info['duration'],'expected_duration_s':frame_count/FPS,
        'duration_note':'61 inclusive physical timestamps0..3s occupy3.05s at20fps per pair.',
        'resolution':[1280,640],'ellipsoids_sampled':len(kernel_ids),'ellipsoid_shape':'exactuniformtetsecondmoment transported bycompatibleF;1sigma,notuncertainty',
        'same_camera_scale_colors':True,'motion_magnification':1.,'pairs':metadata,'posterior_or_image_inference_claim':False}
    output.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',required=True);parser.add_argument('--predictions',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--pairs',help='Comma-separated held pair IDs/globalindices, or all; defaultfirstheld');parser.add_argument('--kernels',type=int,default=100)
    args=parser.parse_args();result=run(args.dataset,args.predictions,args.output,pairs=args.pairs,kernels=args.kernels)
    print(json.dumps({key:result[key] for key in ['frames','duration_s','bytes','output']}))
