"""Measured FEM/posterior visualizations; fixed case zero, no generated evidence.

Raw arrays stay under runs/. PNG/video output is permitted by default. PDF output
requires --pdf after the parent has invoked the artifact-operation marker.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from . import bank,campaign
from .mechanics import LinearGaussianBody
from .inference import marginal_quantiles

ROOT=bank.ROOT;BASE=ROOT/'runs/joint-field-identification-v01';REPORT=ROOT/'reports/joint-field-identification-v01';OUT=ROOT/'output/joint-field-identification-v01'
for path in (Path.home()/'.local/share/fonts/arial').glob('*.TTF'):
    font_manager.fontManager.addfont(str(path))
plt.rcParams.update({'font.family':'Arial','font.size':11,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','axes.titlecolor':'black','pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
ERROR=colors.LinearSegmentedColormap.from_list('position_error',['#12a8a1','#b6dbca','#ffe599','#ec9946','#b62f27'])
MATERIAL=plt.colormaps['viridis']
LABELS={'initial':'Initial posterior','fixed':'Fixed measurements','active':'Adaptive measurements'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct(shape,development=False):
    data=campaign.load_bank(shape);maps=bank.maps(data)
    truth_path=BASE/shape/('development.npz' if development else 'matched-truth.npz')
    truth=dict(np.load(truth_path));case_path=REPORT/shape/'matched/case-000.json'
    p0=campaign.initial(maps,truth['y'][0]);post={'initial':p0};chosen={}
    if development:
        # Development smoke visualization uses identical fixed actions in both
        # columns; the label makes this explicit and is never a result figure.
        fixed=json.loads((REPORT/shape/'fixed-schedules.json').read_text())
        schedules={'fixed':fixed['fixed']['actions'],'active':fixed['fixed']['actions']}
    else:
        case=json.loads(case_path.read_text())
        schedules={name:[r['action'] for r in case['rows'] if r['policy']==name and r['step']>0] for name in ('fixed','active')}
        if any(len(a)!=4 for a in schedules.values()):raise ValueError('Expected completed four-measurement cases')
    for name,actions in schedules.items():
        p=p0
        for a in actions:p=campaign.action_update(p,maps,a,truth['measurements'][0,a])
        post[name]=p;chosen[name]=[maps['actions'][a]['name'] for a in actions]
    return data,truth,post,chosen,[truth_path]+([] if development else [case_path])


def nodal_dictionary(shape,data,device):
    cache=BASE/shape/'visual-nodal.npz';cachemeta=cache.with_suffix('.json')
    source=dict(bank=sha(BASE/f'{shape}.npz'),mechanics=sha(Path(__file__).with_name('mechanics.py')))
    if cache.exists() and cachemeta.exists() and json.loads(cachemeta.read_text())['source']==source:
        return dict(np.load(cache))
    body=LinearGaussianBody(shape,tuple(data['cells']),device=device);unit=[];probe=[];modulus=[]
    for i in range(0,len(data['theta']),16):
        with torch.no_grad():
            theta=torch.as_tensor(data['theta'][i:i+16],dtype=torch.float64,device=device)
            out=body.solve(theta,probe_specs=bank.probe_specs())
            unit.append(out['unit_response'].cpu().numpy());probe.append(out['probe_response'].cpu().numpy())
            modulus.append(body.modulus_at_points(theta,body.points).cpu().numpy())
    arrays=dict(unit=np.concatenate(unit),probe=np.concatenate(probe),material=np.concatenate(modulus),points=body.points.cpu().numpy(),triangles=body.surface_triangles.cpu().numpy(),mass=body.mass.cpu().numpy())
    np.savez(cache,**arrays);cachemeta.write_text(json.dumps(dict(source=source,cache_sha256=sha(cache),scope='Full-node unit-response dictionary for display only; no inference refit.'),indent=2))
    return arrays


def weighted_quantiles(values,weights,quantiles=(.05,.95)):
    order=np.argsort(values,axis=0);v=np.take_along_axis(values,order,axis=0);w=np.take_along_axis(np.broadcast_to(weights[:,None],values.shape),order,axis=0);cdf=np.cumsum(w,axis=0)
    return np.stack([v[np.argmax(cdf>=q,axis=0),np.arange(values.shape[1])] for q in quantiles])


def predictions(nodal,truth,post):
    idx=int(truth['indices'][0]);force=truth['force'][0].reshape(8,2)
    result={'reference':dict(replay=np.einsum('ncl,tl->tnc',nodal['unit'][idx],force),changed=bank.PROBE_FORCE*nodal['probe'][idx,...,8],material=nodal['material'][idx])}
    for name,p in post.items():
        replay=np.einsum('h,hncl,htl->tnc',p.weights,nodal['unit'],p.means.reshape(-1,8,2),optimize=True)
        changed=np.einsum('h,hnc->nc',p.weights,nodal['probe'][...,8])*bank.PROBE_FORCE
        lo,hi=weighted_quantiles(nodal['material'],p.weights)
        result[name]=dict(replay=replay,changed=changed,material=p.weights@nodal['material'],material_lo=lo,material_hi=hi)
    return result


def draw_body(ax,nodal,displacement,values=None,cmap=ERROR,norm=None,azim=-60,elev=23,reference=False):
    xyz=(nodal['points']+displacement)*1000;tri=nodal['triangles'];surface=xyz[tri]
    if values is None:face=np.tile(colors.to_rgba('#8b7fbb' if reference else '#75b5b6'),(len(tri),1))
    else:face=cmap(norm(np.asarray(values)[tri].mean(1)))
    # Geometric directional lighting changes brightness only, identically for
    # every method; it does not alter geometry or numeric colorbar values.
    poly=Poly3DCollection(surface,facecolors=face,edgecolors=(.08,.08,.08,.18),linewidths=.12,antialiased=True)
    ax.add_collection3d(poly)
    ax.set_xlim(-65,65);ax.set_ylim(-35,35);ax.set_zlim(-5,65);ax.set_box_aspect((130,70,70));ax.view_init(elev=elev,azim=azim);ax.set_axis_off()
    ax.set_proj_type('ortho')


def save(fig,path,pdf=False):
    fig.savefig(path.with_suffix('.png'),dpi=190,bbox_inches='tight',pad_inches=.04)
    if pdf:fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.04)
    plt.close(fig)


def response_figure(cohort,prefix,pdf=False):
    methods=['reference','initial','fixed','active'];vmax=max(float(np.linalg.norm(c['pred'][m]['changed']-c['pred']['reference']['changed'],axis=-1).max()*1000) for c in cohort.values() for m in methods[1:]);norm=colors.Normalize(0,max(vmax,.001))
    fig=plt.figure(figsize=(13.2,6.0));gs=fig.add_gridspec(2,4,left=.015,right=.985,bottom=.20,top=.86,wspace=.0,hspace=.16)
    fig.text(.02,.965,'Does the inferred explanation predict a different load?',fontsize=19,weight='bold',va='top')
    fig.text(.02,.91,'Case 000 · held load x = +20 mm · 0.08 N downward · true scale · fixed/adaptive: 4 measurements',fontsize=11)
    metrics={}
    for row,(shape,c) in enumerate(cohort.items()):
        metrics[shape]={}
        for col,m in enumerate(methods):
            ax=fig.add_subplot(gs[row,col],projection='3d');u=c['pred'][m]['changed'];err=np.linalg.norm(u-c['pred']['reference']['changed'],axis=-1)*1000
            draw_body(ax,c['nodal'],u,None if m=='reference' else err,norm=norm,reference=m=='reference')
            ax.set_title(('Reference' if m=='reference' else LABELS[m])+('' if row else ''),fontsize=12,weight='bold',pad=-1)
            rms=float(np.sqrt(np.sum(c['nodal']['mass']*err**2)/np.sum(c['nodal']['mass'])));metrics[shape][m]=rms
            ax.text2D(.5,-.02,shape.title() if m=='reference' else f'Volume-weighted error: {rms:.3f} mm',transform=ax.transAxes,ha='center',fontsize=10)
    cax=fig.add_axes([.32,.12,.4,.022]);fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=ERROR),cax=cax,orientation='horizontal').set_label('Point-position error (mm); one shared scale',fontsize=10)
    fig.text(.02,.012,'Reference is a known simulated response. Predictions are posterior means. These panels are geometry, not photorealistic reconstruction.',fontsize=9)
    save(fig,OUT/f'{prefix}changed-load-comparison',pdf);return metrics


def material_figure(cohort,prefix,pdf=False):
    lo=min(float(c['nodal']['material'].min()) for c in cohort.values())/1000;hi=max(float(c['nodal']['material'].max()) for c in cohort.values())/1000
    norm=colors.Normalize(lo,hi);widthmax=max(float((c['pred']['active']['material_hi']-c['pred']['active']['material_lo']).max())/1000 for c in cohort.values());widthnorm=colors.Normalize(0,max(widthmax,.01))
    fig=plt.figure(figsize=(13.2,5.8));gs=fig.add_gridspec(2,4,left=.015,right=.985,bottom=.20,top=.85,wspace=.0,hspace=.12)
    fig.text(.02,.965,'Material is a spatial field, not one stiffness number',fontsize=19,weight='bold',va='top')
    fig.text(.02,.91,'Case 000 · material coordinates · same modulus scale across shapes and estimates',fontsize=11)
    for row,(shape,c) in enumerate(cohort.items()):
        for col,m in enumerate(['reference','initial','active','uncertainty']):
            ax=fig.add_subplot(gs[row,col],projection='3d');unc=m=='uncertainty'
            value=(c['pred']['active']['material_hi']-c['pred']['active']['material_lo'])/1000 if unc else c['pred'][m]['material']/1000
            draw_body(ax,c['nodal'],np.zeros_like(c['nodal']['points']),value,cmap=plt.colormaps['magma'] if unc else MATERIAL,norm=widthnorm if unc else norm)
            title='True material' if m=='reference' else ('90% interval width' if unc else LABELS[m]);ax.set_title(title,fontsize=12,weight='bold',pad=-1)
            if col==0:ax.text2D(.5,-.02,shape.title(),transform=ax.transAxes,ha='center',fontsize=10)
    cax=fig.add_axes([.19,.12,.35,.022]);fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=MATERIAL),cax=cax,orientation='horizontal').set_label('Young modulus (kPa)',fontsize=10)
    cax=fig.add_axes([.73,.12,.2,.022]);fig.colorbar(plt.cm.ScalarMappable(norm=widthnorm,cmap='magma'),cax=cax,orientation='horizontal').set_label('Interval width (kPa)',fontsize=10)
    fig.text(.02,.012,'Intervals are marginal posterior intervals under the finite hypothesis model; their empirical coverage is evaluated separately.',fontsize=9)
    save(fig,OUT/f'{prefix}material-fields',pdf)


def force_figure(cohort,prefix,pdf=False):
    fig,axes=plt.subplots(2,2,figsize=(11.3,5.8),sharex=True,layout='constrained');x=np.linspace(-.06,.06,81);records={}
    for row,(shape,c) in enumerate(cohort.items()):
        profile=np.exp(-.5*((x[None]-c['data']['theta'][:,4,None])/.018)**2);profile/=np.trapezoid(profile,x,axis=1)[:,None]
        trueprofile=np.exp(-.5*((x-c['truth']['theta'][0,4])/.018)**2);trueprofile/=np.trapezoid(trueprofile,x)
        records[shape]={}
        for col,m in enumerate(['initial','active']):
            ax=axes[row,col];T=np.zeros((len(profile),len(x),16));T[:,:,8]=profile
            p=c['post'][m];mom=p.target(T);lo,hi=marginal_quantiles(p,T);true=c['truth']['force'][0,8]*trueprofile
            ax.fill_between(x*1000,lo,hi,color='#badedc',alpha=.8,label='90% posterior interval')
            ax.plot(x*1000,mom.mean,color='#178b88',lw=2.2,label='Posterior mean');ax.plot(x*1000,true,color='black',lw=1.6,ls='--',label='Reference')
            ax.axhline(0,color='.7',lw=.6);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
            ax.set_title(f'{shape.title()} | {LABELS[m]}',weight='bold',fontsize=12);ax.set_ylabel('Normalized longitudinal\nload-basis profile (N/m)');ax.set_xlabel('Material x coordinate (mm)')
            records[shape][m]=dict(x_m=x.tolist(),true_N_per_m=true.tolist(),mean_N_per_m=mom.mean.tolist(),lower90_N_per_m=lo.tolist(),upper90_N_per_m=hi.tolist())
        limits=[a.get_ylim() for a in axes[row]];low=min(v[0] for v in limits);high=max(v[1] for v in limits)
        for a in axes[row]:a.set_ylim(low,high)
    axes[0,0].legend(fontsize=9,loc='best');fig.suptitle('Inferring the unknown longitudinal load basis',fontsize=18,weight='bold')
    fig.supxlabel('Case 000 · time index 4 · top-load mode · normalized 1D Gaussian proxy, not exact integrated surface traction',fontsize=10)
    save(fig,OUT/f'{prefix}force-fields',pdf)
    (BASE/f'{prefix}visual-force-profiles.json').write_text(json.dumps(records))


def video(shape,c,prefix,frames_per_state=6):
    ffmpeg=ROOT/'runs/media-tooling/ffmpeg-7.0.2-amd64-static/ffmpeg';path=OUT/f'{prefix}{shape}-quasistatic-replay.mp4'
    methods=['reference','initial','fixed','active'];vmax=max(float(np.linalg.norm(c['pred'][m]['replay']-c['pred']['reference']['replay'],axis=-1).max()*1000) for m in methods[1:]);norm=colors.Normalize(0,max(vmax,.001))
    fig=plt.figure(figsize=(16,4.6),dpi=100);axes=[fig.add_subplot(1,4,i+1,projection='3d') for i in range(4)];fig.subplots_adjust(left=.005,right=.995,bottom=.20,top=.81,wspace=.0)
    fig.text(.02,.96,f'{shape.title()} | actual and inferred deformation',fontsize=21,weight='bold',va='top');subtitle=fig.text(.02,.86,'',fontsize=12)
    fig.text(.02,.025,'Known simulated geometry · true deformation scale · orbiting camera · quasistatic replay, not a dynamic forecast',fontsize=11)
    cax=fig.add_axes([.35,.15,.3,.022]);fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=ERROR),cax=cax,orientation='horizontal').set_label('Point-position error (mm)',fontsize=10)
    width,height=fig.canvas.get_width_height()
    proc=subprocess.Popen([str(ffmpeg),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(frames_per_state),'-i','-','-an','-vcodec','libx264','-pix_fmt','yuv420p','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-crf','19',str(path)],stdin=subprocess.PIPE)
    for frame in range(8*frames_per_state):
        t=frame//frames_per_state;azim=-70+70*frame/(8*frames_per_state-1);subtitle.set_text(f'Case 000 · load time {bank.TIMES[t]:.2f} · state {t+1}/8 · original unknown load sequence')
        for ax,m in zip(axes,methods):
            ax.clear();u=c['pred'][m]['replay'][t];err=np.linalg.norm(u-c['pred']['reference']['replay'][t],axis=-1)*1000
            draw_body(ax,c['nodal'],u,None if m=='reference' else err,norm=norm,azim=azim,reference=m=='reference')
            ax.set_title('Reference' if m=='reference' else LABELS[m],fontsize=14,weight='bold',pad=-1)
            if m!='reference':
                rms=np.sqrt(np.sum(c['nodal']['mass']*err**2)/np.sum(c['nodal']['mass']));ax.text2D(.5,-.01,f'Error: {rms:.3f} mm',transform=ax.transAxes,ha='center',fontsize=12)
        fig.canvas.draw();image=np.asarray(fig.canvas.buffer_rgba())[...,:3];proc.stdin.write(np.ascontiguousarray(image).tobytes())
    proc.stdin.close();code=proc.wait();plt.close(fig)
    if code:raise RuntimeError('ffmpeg failed')
    return path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:2');ap.add_argument('--development',action='store_true');ap.add_argument('--pdf',action='store_true');ap.add_argument('--skip-video',action='store_true');args=ap.parse_args();torch.set_num_threads(1);OUT.mkdir(exist_ok=True,parents=True)
    prefix='DEVELOPMENT-' if args.development else '';cohort={};sources=[]
    if args.development:LABELS['active']='Development fixed duplicate'
    for shape in ['block','step']:
        data,truth,post,chosen,paths=reconstruct(shape,args.development);nodal=nodal_dictionary(shape,data,args.device)
        cohort[shape]=dict(data=data,truth=truth,post=post,chosen=chosen,nodal=nodal,pred=predictions(nodal,truth,post));sources.extend(paths+[BASE/f'{shape}.npz',BASE/shape/'visual-nodal.npz'])
    metrics=response_figure(cohort,prefix,args.pdf);material_figure(cohort,prefix,args.pdf);force_figure(cohort,prefix,args.pdf)
    if not args.skip_video:
        for shape,c in cohort.items():video(shape,c,prefix)
    manifest=dict(selection='Predeclared case000 in both shapes; held calibrated-load query8 (top x+.02m), force time4; no outcome-based selection.',development=args.development,actual_geometry_scale=True,changed_load_force_N=bank.PROBE_FORCE,changed_load_surface_error_rms_mm=metrics,action_choices={s:c['chosen'] for s,c in cohort.items()},sources={str(p.relative_to(ROOT)):sha(p) for p in sources},generator_sha256=sha(__file__),scope='Actual FEM/posterior predictions. Geometry and conditional material/load fields, not photorealistic RGB. Video is 8-state quasistatic replay with camera orbit; no dynamic interpolation or motion exaggeration.')
    (OUT/f'{prefix}figure-manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
