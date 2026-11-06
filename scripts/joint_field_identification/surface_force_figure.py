"""Exact FEM surface-quadrature force patches, fixed v04 case000/time4."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from . import surface,surface_campaign,visualize
from .mechanics import LinearGaussianBody
from .surface_visualize import BASE,REPORT,OUT,ROOT,sha

def patches(body,theta):
    q=body.surface_points.numpy();area=body.surface_quadrature_weights.numpy();xx=theta[:,4]
    raw=np.exp(-.5*((q[None,:,0]-xx[:,None])/body.footprint_width)**2)
    matrices=[];centers=[];modes=[];labels=[]
    for mode,face in enumerate(['top','side']):
        mask=body._surface_masks[face].numpy();w=raw*area[None]*mask[None];w/=w.sum(1)[:,None]
        # Exact quadrature grouped into 8 longitudinal by2 transverse patches.
        xb=np.minimum(((q[:,0]+.06)/.12*8).astype(int),7)
        transverse=(q[:,1] if face=='top' else q[:,2]);mid=0 if face=='top' else .03
        tb=(transverse>=mid).astype(int)
        for x in range(8):
            for t in range(2):
                ids=mask&(xb==x)&(tb==t)
                if not ids.any():continue
                centers.append(np.average(q[ids],axis=0,weights=area[ids]));matrices.append(w[:,ids].sum(1));modes.append(mode);labels.append([face,x,t])
    return np.stack(matrices,1),np.array(centers),np.array(modes),labels

def main():
    torch.set_num_threads(1);records={};source=[];allforces=[]
    for shape in ['block','step']:
        bp=BASE/f'{shape}.npz';tp=BASE/shape/'matched-truth.npz';cp=REPORT/shape/'matched/case-000.json'
        data=dict(np.load(bp));truth=dict(np.load(tp));case=json.loads(cp.read_text());maps=surface.maps(data)
        p0=surface_campaign.initial(maps,truth['y'][0]);p=p0
        for row in case['rows']:
            if row['policy']=='active' and row['step']>0:p=surface_campaign.action_update(p,maps,row['action'],truth['measurements'][0,row['action']])
        body=LinearGaussianBody(shape,tuple(data['cells']));weights,centers,modes,labels=patches(body,data['theta']);tw,_,_,_=patches(body,truth['theta'][0:1])
        cache=ROOT/f'runs/joint-field-identification-v01/{shape}/visual-nodal.npz';nodal=dict(np.load(cache))
        force={'true':tw[0]*truth['force'][0,8+modes]};material={'true':nodal['material'][int(truth['indices'][0])]}
        for key,post in [('initial',p0),('active',p)]:
            force[key]=np.einsum('h,hp,hp->p',post.weights,weights,post.means[:,8+modes]);material[key]=post.weights@nodal['material']
        # Check conservation separately for both original unit-load modes.
        assert np.max(abs(weights[:,modes==0].sum(1)-1))<1e-12
        assert np.max(abs(weights[:,modes==1].sum(1)-1))<1e-12
        records[shape]=dict(force=force,material=material,centers=centers,modes=modes,labels=labels,nodal=nodal)
        allforces.extend(force.values());source.extend([bp,tp,cp,cache])
    maximum=max(float(abs(f).max()) for f in allforces);glyph_m_per_N=.032/maximum
    plt=visualize.plt;norm=visualize.colors.Normalize(min(float(r['material'][k].min()) for r in records.values() for k in r['material']),max(float(r['material'][k].max()) for r in records.values() for k in r['material']))
    fig=plt.figure(figsize=(11.3,7.0));gs=fig.add_gridspec(2,3,left=.015,right=.99,bottom=.22,top=.87,wspace=.01,hspace=.22)
    for row,(shape,r) in enumerate(records.items()):
        for col,key in enumerate(['true','initial','active']):
            ax=fig.add_subplot(gs[row,col],projection='3d',computed_zorder=False);visualize.draw_body(ax,r['nodal'],np.zeros_like(r['nodal']['points']),r['material'][key],visualize.MATERIAL,norm,azim=58)
            vec=np.zeros_like(r['centers']);vec[r['modes']==0,2]=-r['force'][key][r['modes']==0];vec[r['modes']==1,1]=-r['force'][key][r['modes']==1]
            ax.quiver(*(r['centers']*1000).T,*(vec*glyph_m_per_N*1000).T,color='#d04a24',arrow_length_ratio=.25,linewidth=1.1,normalize=False,zorder=100)
            ax.set_title({'true':'Reference','initial':'Initial posterior','active':'Adaptive measurements'}[key],fontsize=13,weight='bold',pad=-1)
            ax.text2D(.5,.02,f'{shape.title()} · total Fz = {-r["force"][key][r["modes"]==0].sum()*1000:.1f}; Fy = {-r["force"][key][r["modes"]==1].sum()*1000:.1f} mN',transform=ax.transAxes,ha='center',fontsize=10)
    fig.suptitle('Joint material and spatial load fields',x=.02,ha='left',fontsize=19,weight='bold');fig.text(.02,.91,'Case 000 · time4 · exact surface quadrature summed into patches · posterior mean forces',fontsize=10)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=visualize.MATERIAL),cax=fig.add_axes([.31,.145,.39,.019]),orientation='horizontal');cb.set_label('Young modulus (Pa)',fontsize=10)
    fig.text(.02,.035,f'Orange arrows: integrated force per patch (N); shared glyph scale {glyph_m_per_N:.2f} m/N (32 mm = {maximum*1000:.2f} mN).\nRest geometry at true scale; signed applied loads, not unilateral contact. Arrow lengths are force symbols, not displacements.',fontsize=9)
    visualize.save(fig,OUT/'spatial-force-fields',True)
    manifest=dict(selection='Fixed case000/time4, both shapes, no outcome selection',geometry='Rest geometry at true scale; material hidden field not RGB',force_definition='Exact surface quadrature integrated into 8 longitudinal x2 transverse patches on each top/side face, N per patch. Directions top -z, side -y; signed forces.',shared_glyph_m_per_N=glyph_m_per_N,maximum_patch_force_N=maximum,sources={str(p.relative_to(ROOT)):sha(p) for p in source},generator_sha256=sha(__file__),cases={s:dict(patch_centers_m=r['centers'].tolist(),patch_modes=r['labels'],force_N={k:v.tolist() for k,v in r['force'].items()}) for s,r in records.items()})
    (OUT/'spatial-force-manifest.json').write_text(json.dumps(manifest,indent=2));print('Saved exact spatial force figure',maximum,glyph_m_per_N)

if __name__=='__main__':main()
