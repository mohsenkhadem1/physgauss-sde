"""Measured scale-symmetry teaser; constructed explanations, not fitted posterior."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors,font_manager
from mpl_toolkits.mplot3d import proj3d
from .mechanics import LinearGaussianBody
from . import surface as bank
from .visualize import draw_body

ROOT=bank.ROOT;OUT=ROOT/'output/joint-field-surface-v04'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def marginal_scale_check(theta,unit,force,body):
    """Same actual surface observation model, co-scaled latent force prior."""
    N=body.interpolation(bank.points()).cpu().numpy()
    pointunit=np.einsum('pn,hncl->hpcl',N,unit)
    H=bank.map_observations(dict(unit=pointunit,probe=np.zeros((2,20,3,6))))['H']
    scales=np.geomspace(.5,2.,5)
    assert all(np.isclose(scales,tau).any() for tau in [1.,2.])
    noise=bank.NOISE**2*np.eye(H.shape[1]);L=bank.load_cov()
    CA=H[0]@L@H[0].T+noise;CB=H[1]@(4*L)@H[1].T+noise
    y=H[0]@force.reshape(-1)
    def loglik(C):
        chol=np.linalg.cholesky(C);white=np.linalg.solve(chol,y)
        return float(-.5*(white@white+2*np.log(np.diag(chol)).sum()+len(y)*np.log(2*np.pi)))
    la,lb=loglik(CA),loglik(CB);fixed=loglik(H[1]@L@H[1].T+noise)
    error=float(abs(CA-CB).max());relative=error/float(abs(CA).max())
    assert relative<1e-10 and abs(la-lb)<1e-9
    return dict(surface_initial_observation_count=len(y),force_prior_scales=[1.,2.],scales_are_hierarchy_prior_atoms=True,noise_std_m=bank.NOISE,maximum_covariance_difference_m2=error,relative_maximum_covariance_difference=relative,log_likelihood_A=la,log_likelihood_B_coscaled=lb,absolute_log_likelihood_difference=abs(la-lb),log_likelihood_B_fixed_nominal_prior=fixed,constructed_observations_m=y.tolist(),scope='Conditional marginal likelihood identity for two spatial/scale hypothesis atoms. Equal component likelihood is not a claim that entire posterior marginals are identical. Fixed force prior does not share this symmetry.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');args=ap.parse_args();torch.set_num_threads(1)
    source=ROOT/'reports/joint-field-surface-v04/protocol.json'
    theta=np.array([[np.log(16000),.4,0.,np.log(8000),.02],[np.log(32000),.4,0.,np.log(16000),.02]]);scale=2.
    grid=bank.grid();matches=[np.flatnonzero(np.isclose(grid,t,rtol=0,atol=1e-12).all(1)) for t in theta]
    assert all(len(m)==1 for m in matches)
    assert ((theta>=bank.BOUNDS[:,0]-1e-12)&(theta<=bank.BOUNDS[:,1]+1e-12)).all()
    body=LinearGaussianBody('block',(8,4,4),device=args.device)
    with torch.no_grad():
        solved=body.solve(theta);unit=solved['unit_response'].cpu().numpy();probe=solved['probe_response'].cpu().numpy();material=body.modulus_at_points(theta,body.points).cpu().numpy()/1000
    force=np.column_stack([.025*np.sin(np.linspace(.3,2.4,8)),.02*np.cos(np.linspace(.2,2.,8))])
    motion=np.stack([np.einsum('ncl,tl->tnc',unit[0],force),np.einsum('ncl,tl->tnc',unit[1],force*scale)])
    cameras=np.array([[[1,0,0],[0,0,1]],[[0,1,0],[0,0,1]],[[2**-.5,2**-.5,0],[0,0,1]]])
    projected=np.einsum('aki,htni->hatnk',cameras,motion)
    max_difference=float(np.abs(projected[0]-projected[1]).max())
    calibrated=bank.PROBE_FORCE*probe[:,:,:,0]
    rms=np.sqrt(np.mean(np.sum(calibrated**2,axis=-1),axis=1))*1000
    ratio=float(rms[1]/rms[0]);assert abs(ratio-.5)<1e-10
    allnode_difference=float(abs(motion[0]-motion[1]).max());assert allnode_difference<1e-12
    assert max_difference<1e-12
    p=np.array([[-.015,-.03,.025]]);N=body.interpolation(p).cpu().numpy()
    track=np.einsum('pn,htnc->htpc',N,motion)[:,:,0,2]*1000
    from .surface_visibility import Visibility
    visible=Visibility('block',(8,4,4));point_index=int(np.flatnonzero(np.isclose(visible.points,p[0]).all(1))[0])
    track_checks=[visible.check(d,[dict(name='teaser_front_track',point=point_index,camera_direction=[0.,-1.,0.])])[0] for d in motion[0]]
    assert all(r['visible'] for r in track_checks)
    nodal=dict(points=body.points.cpu().numpy(),triangles=body.surface_triangles.cpu().numpy())
    # Every displacement is displayed at actual geometric scale; only the
    # separately labelled plot shows displacement coordinates in millimetres.
    fig=plt.figure(figsize=(12.5,4.0));gs=fig.add_gridspec(1,4,width_ratios=[1.05,1.05,1.,1.],left=.005,right=.985,bottom=.29,top=.75,wspace=.30)
    fig.text(.015,.965,'The same motion can have different physical explanations',fontsize=18,weight='bold',va='top')
    fig.text(.015,.865,'Co-scaled stiffness and force preserve motion; a calibrated force separates the explanations.',fontsize=13.5)
    cmap=plt.colormaps['viridis'];norm=colors.Normalize(float(material.min()),float(material.max()));titles=['Explanation A','Explanation B']
    for i in range(2):
        ax=fig.add_subplot(gs[0,i],projection='3d');draw_body(ax,nodal,motion[i,4],material[i],cmap=cmap,norm=norm,azim=-62,elev=24)
        ax.set_title(titles[i],fontsize=12.5,weight='bold',pad=4)
        # External load arrow lengths reflect the same 2 force ratio, with
        # an explicit glyph legend; their length is not object displacement.
        length=8*(1 if i==0 else scale);x=theta[i,4]*1000
        ax.text2D(.5,-.08,f"{np.exp(theta[i,0])/1000:.0f} kPa; {np.exp(theta[i,3])/1000:.0f} kN/m\nLoad {'F(t)' if i==0 else '2 F(t)'}",transform=ax.transAxes,ha='center',fontsize=10)
        # More headroom for the schematic force arrow, unchanged body scale.
        ax.set_zlim(-5,85);ax.set_box_aspect((130,70,90))
        for dx in [-9,0,9]:
            start=proj3d.proj_transform(x+dx,-15,60+length,ax.get_proj())
            end=proj3d.proj_transform(x+dx,-15,60,ax.get_proj())
            ax.annotate('',xy=end[:2],xytext=start[:2],xycoords='data',textcoords='data',
                        arrowprops=dict(arrowstyle='-|>',color='#de7c27',lw=1.5,mutation_scale=10),zorder=100)

    ax=fig.add_subplot(gs[0,2]);ax.plot(bank.TIMES,track[0],color='#168f89',lw=3,label='A');ax.plot(bank.TIMES,track[1],color='#7151a6',lw=1.7,ls='--',label='B')
    ax.set_title('Identical motion',fontsize=12,weight='bold',pad=13);ax.set_xlabel('Load-sequence time');ax.set_ylabel('Vertical displacement (mm)');ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.18);ax.legend(loc='best',frameon=False,ncol=2,fontsize=9)
    ax.text(.5,-.60,'Three-camera difference\n'+f'{max_difference*1000:.1e} mm',ha='center',transform=ax.transAxes,fontsize=9.5)
    ax=fig.add_subplot(gs[0,3]);bars=ax.bar([0,1],rms,color=['#168f89','#7151a6'],width=.55)
    ax.set_xticks([0,1],['A','B']);ax.set_ylabel('RMS motion (mm)');ax.set_title('Known 0.08 N probe',fontsize=11.5,weight='bold',pad=13);ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,float(rms.max())*1.35);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    for bar,v in zip(bars,rms):ax.text(bar.get_x()+bar.get_width()/2,v+float(rms.max())*.04,f'{v:.3f}',ha='center',fontsize=10)
    ax.text(.5,-.60,f'Response B / A = {ratio:.2f}\nKnown-force probe at x = −40 mm',ha='center',transform=ax.transAxes,fontsize=9.5)
    cax=fig.add_axes([.055,.16,.32,.025]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal');cb.set_label('Spatial Young modulus (kPa)',fontsize=9);cb.ax.tick_params(labelsize=8)
    # Construction/geometry disclaimers are carried by the paper caption.
    from matplotlib.text import Text
    for ax in fig.axes:ax.tick_params(labelsize=13.5)
    for text in fig.findobj(Text):
        if text.get_text():text.set_fontsize(max(text.get_fontsize(),13.5))
    OUT.mkdir(exist_ok=True,parents=True)
    for ext in ['png','pdf']:fig.savefig(OUT/f'ambiguity.{ext}',dpi=200,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
    record=dict(scope='Constructed exact scale symmetry, not inferred posterior and not a benchmark result. Linear quasistatic mechanics with no gravity. Both constructed parameter choices are atoms within the primary surface-v04 prior; not the old2.5qualification test.',shape='block',cells=[8,4,4],theta=theta.tolist(),dictionary_indices=[int(m[0]) for m in matches],within_primary_bounds=True,scale_factor=scale,all_node_max_difference_m=allnode_difference,force_history_A_N=force.tolist(),force_history_B_N=(force*scale).tolist(),illustrated_time_index=4,track_reference_point_m=p.tolist(),track_visibility_all8states=True,max_three_camera_projection_difference_m=max_difference,probe_force_N=bank.PROBE_FORCE,probe_spec=body.probe_specs[0],probe_node_vector_rms_mm=rms.tolist(),probe_response_ratio=ratio,source_hashes={str(source.relative_to(ROOT)):digest(source),'scripts/joint_field_identification/surface.py':digest(Path(__file__).with_name('surface.py')),'scripts/joint_field_identification/mechanics.py':digest(Path(__file__).with_name('mechanics.py')),'scripts/joint_field_identification/surface_ambiguity.py':digest(__file__)},display='Actual geometry scale; force-arrow lengths are glyphs with2ratio, not displacement. Line panel is explicitly labelled displacement coordinates.')
    record['marginal_force_scale_symmetry']=marginal_scale_check(theta,unit,force,body)
    (OUT/'ambiguity-manifest.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))

if __name__=='__main__':main()
