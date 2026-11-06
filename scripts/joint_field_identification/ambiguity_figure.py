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
from . import bank
from .visualize import draw_body

ROOT=bank.ROOT;OUT=ROOT/'output/joint-field-identification-v01'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');args=ap.parse_args();torch.set_num_threads(1)
    source=ROOT/'reports/joint-field-identification-v01/qualification.json';qualified=json.loads(source.read_text())['scaling']
    theta=np.array([qualified['original_theta'],qualified['scaled_theta']]);scale=qualified['scale_factor']
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
    ratio=float(rms[1]/rms[0]);assert abs(ratio-qualified['expected_probe_ratio'])<1e-10
    assert max_difference<1e-12
    p=np.array([[-.03,-.03,.06]]);N=body.interpolation(p).cpu().numpy()
    track=np.einsum('pn,htnc->htpc',N,motion)[:,:,0,2]*1000
    nodal=dict(points=body.points.cpu().numpy(),triangles=body.surface_triangles.cpu().numpy())
    # Every displacement is displayed at actual geometric scale; only the
    # separately labelled plot shows displacement coordinates in millimetres.
    fig=plt.figure(figsize=(12.5,4.0));gs=fig.add_gridspec(1,4,width_ratios=[1.05,1.05,1.,1.],left=.005,right=.985,bottom=.29,top=.75,wspace=.30)
    fig.text(.015,.965,'The same motion can have different physical explanations',fontsize=18,weight='bold',va='top')
    fig.text(.015,.865,'Scale material, support and unknown force together: images stay identical. A known force distinguishes the explanations.',fontsize=10.5)
    cmap=plt.colormaps['viridis'];norm=colors.Normalize(float(material.min()),float(material.max()));titles=['Explanation A','Explanation B']
    for i in range(2):
        ax=fig.add_subplot(gs[0,i],projection='3d');draw_body(ax,nodal,motion[i,4],material[i],cmap=cmap,norm=norm,azim=-62,elev=24)
        ax.set_title(titles[i],fontsize=12.5,weight='bold',pad=4)
        # External load arrow lengths reflect the same 2.5 force ratio, with
        # an explicit glyph legend; their length is not object displacement.
        length=8*(1 if i==0 else scale);x=theta[i,4]*1000
        ax.text2D(.5,-.08,f"Mean scale {np.exp(theta[i,0])/1000:.0f} kPa\nSupport {np.exp(theta[i,3])/1000:.0f} kN/m; load {'F(t)' if i==0 else '2.5 F(t)'}",transform=ax.transAxes,ha='center',fontsize=10)
        # More headroom for the schematic force arrow, unchanged body scale.
        ax.set_zlim(-5,85);ax.set_box_aspect((130,70,90))
        for dx in [-9,0,9]:
            start=proj3d.proj_transform(x+dx,-15,60+length,ax.get_proj())
            end=proj3d.proj_transform(x+dx,-15,60,ax.get_proj())
            ax.annotate('',xy=end[:2],xytext=start[:2],xycoords='data',textcoords='data',
                        arrowprops=dict(arrowstyle='-|>',color='#de7c27',lw=1.5,mutation_scale=10),zorder=100)

    ax=fig.add_subplot(gs[0,2]);ax.plot(bank.TIMES,track[0],color='#168f89',lw=3,label='A');ax.plot(bank.TIMES,track[1],color='#7151a6',lw=1.7,ls='--',label='B')
    ax.set_title('Identical observed motion',fontsize=12,weight='bold',pad=13);ax.set_xlabel('Load-sequence time');ax.set_ylabel('Vertical displacement (mm)');ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.18);ax.legend(loc='best',frameon=False,ncol=2,fontsize=9)
    ax.text(.5,-.38,'Maximum difference across 3 cameras\n'+f'{max_difference*1000:.1e} mm',ha='center',transform=ax.transAxes,fontsize=9.5)
    ax=fig.add_subplot(gs[0,3]);bars=ax.bar([0,1],rms,color=['#168f89','#7151a6'],width=.55)
    ax.set_xticks([0,1],['A','B']);ax.set_ylabel('Displacement RMS (mm)');ax.set_title('Same calibrated 0.08 N probe',fontsize=11.5,weight='bold',pad=13);ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,float(rms.max())*1.35);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    for bar,v in zip(bars,rms):ax.text(bar.get_x()+bar.get_width()/2,v+float(rms.max())*.04,f'{v:.3f}',ha='center',fontsize=10)
    ax.text(.5,-.38,f'Response B / A = {ratio:.2f}\nKnown-force probe at x = −40 mm',ha='center',transform=ax.transAxes,fontsize=9.5)
    cax=fig.add_axes([.055,.16,.32,.025]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal');cb.set_label('Spatial Young modulus (kPa)',fontsize=9);cb.ax.tick_params(labelsize=8)
    fig.text(.015,.015,'Constructed linear-mechanics symmetry, not an inferred posterior. Colors show hidden stiffness, not RGB. Geometry uses true scale; arrows illustrate top loading.',fontsize=9)
    OUT.mkdir(exist_ok=True,parents=True)
    for ext in ['png','pdf']:fig.savefig(OUT/f'ambiguity.{ext}',dpi=200,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
    record=dict(scope='Constructed exact scale symmetry, not inferred posterior and not a benchmark result. Linear quasistatic mechanics with no gravity. Parameters and force history match qualification.py.',shape='block',cells=[8,4,4],theta=theta.tolist(),scale_factor=scale,force_history_A_N=force.tolist(),force_history_B_N=(force*scale).tolist(),illustrated_time_index=4,track_reference_point_m=p.tolist(),max_three_camera_projection_difference_m=max_difference,probe_force_N=bank.PROBE_FORCE,probe_spec=body.probe_specs[0],probe_node_vector_rms_mm=rms.tolist(),probe_response_ratio=ratio,source_hashes={str(source.relative_to(ROOT)):digest(source),'scripts/joint_field_identification/qualification.py':digest(Path(__file__).with_name('qualification.py')),'scripts/joint_field_identification/mechanics.py':digest(Path(__file__).with_name('mechanics.py')),'scripts/joint_field_identification/ambiguity_figure.py':digest(__file__)},display='Actual geometry scale; force-arrow lengths are glyphs with2.5ratio, not displacement. Line panel is explicitly labelled displacement coordinates.')
    (OUT/'ambiguity-manifest.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))

if __name__=='__main__':main()
