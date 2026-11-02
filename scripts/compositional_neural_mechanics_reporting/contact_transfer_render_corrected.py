"""Reporting-only color-map/layout correction; frozen physics and inputs unchanged."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scripts.compositional_neural_mechanics.contact_transfer import ROOT,R,OUT,PLAN,FAMILIES,verify,sha,write


def qualification_label(report,family,label):
    if not report.get('complete',False):return 'partial/unqualified'
    if report['stage']=='preflight':return 'development coarse preflight'
    methods=('analytic_teacher','shared_laws')if label=='field_difference'else(label,)
    ok=all(report.get('qualification',{}).get(family+'_'+method,{}).get('passed',False)for method in methods)
    return 'qualified'if ok else'unqualified'


def main(stage):
    p=json.loads(PLAN.read_text());verify(p);reportpath=R/f'contact-transfer-{stage}.json';report=json.loads(reportpath.read_text())
    if not report.get('terminal',False)or report['plan_sha256']!=sha(PLAN):raise ValueError('Terminal provenance-bound stage required')
    n,q=(3,16)if stage=='preflight'else(8,32);data={};hashes={str(reportpath.relative_to(ROOT)):sha(reportpath)}
    for family in FAMILIES:
        for method in ('analytic_teacher','shared_laws'):
            key=f'{family}-{method}-n{n}-q{q}';row=next(x for x in report['rows']if x['key']==key)
            if row['completed']:
                file=OUT/'states'/(key+'.npz')
                if sha(file)!=row['response_sha256']:raise ValueError('Changed rendered response')
                data[(family,method)]=np.load(file)['surface_frames'][-1];hashes[str(file.relative_to(ROOT))]=sha(file)
    errors={f:np.linalg.norm(data[(f,'shared_laws')]-data[(f,'analytic_teacher')],axis=-1)*1e6 for f in FAMILIES if all((f,m)in data for m in ('analytic_teacher','shared_laws'))};vmax=max([float(x.max())for x in errors.values()]+[1.]);norm=Normalize(0,vmax)
    fig=plt.figure(figsize=(13,7));theta=np.linspace(0,2*np.pi,32);phi=np.linspace(0,np.pi,20);sphere=np.stack([.12+.045*np.outer(np.cos(theta),np.sin(phi)),.045*np.outer(np.sin(theta),np.sin(phi)),.080+.045*np.outer(np.ones_like(theta),np.cos(phi))],-1)
    for i,family in enumerate(FAMILIES):
        for j,label in enumerate(('analytic_teacher','shared_laws','field_difference')):
            ax=fig.add_subplot(2,3,3*i+j+1,projection='3d');key=(family,'shared_laws'if j==2 else label)
            if key not in data or(j==2 and family not in errors):ax.text2D(.2,.5,'Missing simulation',transform=ax.transAxes);continue
            for face,xyz in enumerate(data[key]):
                colors=plt.get_cmap('magma')(norm(errors[family][face]))if j==2 else None
                style=dict(facecolors=colors,shade=False)if j==2 else dict(color='#4096bd'if j==0 else'#65ad78',shade=True)
                ax.plot_surface(*np.moveaxis(xyz,-1,0),linewidth=0,antialiased=False,**style)
            ax.plot_surface(*np.moveaxis(sphere,-1,0),color='#d5944b',alpha=.25,linewidth=0)
            ax.plot_surface(*np.meshgrid([-.02,.26],[-.06,.06]),np.full((2,2),-.04001),color='gray',alpha=.2)
            status=qualification_label(report,family,label)
            ax.set_title(f"{family}\n{label.replace('_',' ')}\n{status}",fontsize=8);ax.set(xlim=(-.02,.26),ylim=(-.06,.06),zlim=(-.055,.13));ax.set_box_aspect((.28,.12,.185));ax.view_init(20,-65);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_zlabel('z (m)');ax.set_xticks([0,.12,.24]);ax.set_yticks([-.04,0,.04]);ax.set_zticks([-.04,.04,.12]);ax.tick_params(labelsize=7)
    fig.suptitle('Frozen learned bulk + bilateral laws with known normal contact\nAnalytic teacher simulation versus learned-law simulation; matched known descriptors; physical scale',fontsize=12)
    fig.subplots_adjust(top=.84,hspace=.4,wspace=.12,right=.88)
    cax=fig.add_axes([.92,.2,.015,.55]);fig.colorbar(ScalarMappable(norm=norm,cmap='magma'),cax=cax,label='Surface position difference (µm)')
    files=[]
    for extension in ('png','svg'):
        file=R/f'contact-transfer-{stage}-corrected.{extension}';fig.savefig(file,dpi=160);files.append(file)
    plt.close(fig);write(R/f'contact-transfer-{stage}-corrected-render.json',dict(plan_sha256=sha(PLAN),input_sha256=hashes,source_sha256=sha(Path(__file__)),output_sha256={str(f.relative_to(ROOT)):sha(f)for f in files},surface_error_max_um=vmax,scope='Both declared families; no magnification or per-case selection; not realvideo, not dynamic simulation'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['preflight','fine'],required=True);main(p.parse_args().stage)
