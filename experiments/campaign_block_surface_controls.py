"""Pure public known traction programs, consistent P1/P2 quadrature, S2 BC masks.

Forcing has NO material/support argument. Dead world force directions are held
on reference material patches; deformed moment must be recomputed, not assumed.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
SIZE=(.24,.08,.08)
EDGES=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
PATTERNS={'axial_pull':{'force_N':[1.,0.,0.],'torsion_Nm':0.,'bending_Nm':0.},'transverse_end':{'force_N':[0.,0.,-.12],'torsion_Nm':0.,'bending_Nm':0.},'bending_couple':{'force_N':[0.,0.,0.],'torsion_Nm':0.,'bending_Nm':.02},'torsion_couple':{'force_N':[0.,0.,0.],'torsion_Nm':.02,'bending_Nm':0.},'top_middle_down':{'force_N':[0.,0.,-.12],'torsion_Nm':0.,'bending_Nm':0.,'patch':'top_middle_band'}}
SUPPORTS=('left_face','right_face','both_ends','bottom_face','separated_bottom_patches')

def wrench(positions,forces,origin=(.12,0,0)):
 q=np.asarray(positions);f=np.asarray(forces);o=np.asarray(origin)
 if q.shape!=f.shape or q.ndim!=2 or q.shape[1]!=3 or not all(np.isfinite(x).all() for x in [q,f,o]):raise ValueError('Finite matchingphysicalnodes/forces required')
 return f.sum(0),np.cross(q-o,f).sum(0)

def support_groups(rest,kind):
 X=np.asarray(rest);left=abs(X[:,0])<1e-13;right=abs(X[:,0]-.24)<1e-13;bottom=abs(X[:,2]+.04)<1e-13
 if kind not in SUPPORTS:raise ValueError('DeclaredhardBCkind required')
 rows={'left_face':[left],'right_face':[right],'both_ends':[left,right],'bottom_face':[bottom],'separated_bottom_patches':[bottom&(X[:,0]<=.06+1e-13),bottom&(X[:,0]>=.18-1e-13)]}[kind]
 if any(not r.any() for r in rows):raise ValueError('Meshdoesnotrepresentdeclaredsupport')
 out=np.zeros((2,len(X)),dtype=bool);out[:len(rows)]=rows
 if np.any(out.sum(0)>1):raise ValueError('Supportgroupsmustbedisjoint')
 return out,np.array([True]*len(rows)+[False]*(2-len(rows)))

def surface_control(rest,tets,faces,pattern,elements=None):
 if pattern not in PATTERNS:raise ValueError('Knownpublicpattern required')
 X=np.asarray(rest);T=np.asarray(tets);faces=np.asarray(faces);e=T if elements is None else np.asarray(elements)
 if e.shape not in [(len(T),4),(len(T),10)]:raise ValueError('ExplicitP1orP2connectivity required')
 quadratic=e.shape[1]==10;edge={tuple(sorted(map(int,t[[i,j]]))):int(row[4+k]) for t,row in zip(T,e) for k,(i,j) in enumerate(EDGES)} if quadratic else{}
 L,W,H=SIZE;control=PATTERNS[pattern];band=control.get('patch')=='top_middle_band'
 selected=faces[np.all((abs(X[faces,2]-.04)<1e-13)&(X[faces,0]>=.06-1e-13)&(X[faces,0]<=.18+1e-13),1)] if band else faces[np.all(abs(X[faces,0]-.24)<1e-13,1)]
 area=.12*W if band else W*H;Iz=W*H**3/12;Jp=W*H*(W**2+H**2)/12;F=np.array(control['force_N']);tx=control['torsion_Nm'];my=control['bending_Nm'];f=np.zeros_like(X);actualarea=0.
 z,w=leggauss(3);z=(z+1)/2;w=w/2
 for tri in selected:
  coords=X[tri];A=.5*np.linalg.norm(np.cross(coords[1]-coords[0],coords[2]-coords[0]));actualarea+=A
  ids=np.r_[tri,[edge[tuple(sorted(map(int,tri[[i,j]])))] for i,j in [(0,1),(1,2),(2,0)]]] if quadratic else tri
  for i,a in enumerate(z):
   for j,b in enumerate(z):
    lam=np.array([(1-a)*(1-b),a,(1-a)*b]);x=lam@coords;N=np.r_[lam*(2*lam-1),4*lam[0]*lam[1],4*lam[1]*lam[2],4*lam[2]*lam[0]] if quadratic else lam
    traction=F/area+np.array([my*x[2]/Iz,-tx*x[2]/Jp,tx*x[1]/Jp]);np.add.at(f,ids,A*2*w[i]*w[j]*(1-a)*N[:,None]*traction)
 if not np.isclose(actualarea,area,rtol=0,atol=1e-14):raise ValueError('Completepublicrightfacearea mismatch')
 actualF,actualM=wrench(X,f);expectedM=np.cross([0,0,H/2] if band else[L/2,0,0],F)+[tx,my,0]
 if np.max(abs(actualF-F))>1e-12 or np.max(abs(actualM-expectedM))>1e-12:raise ValueError('Knownresultant/restmomentnormalization failed')
 return f,{'pattern':pattern,'basis':'P2' if quadratic else'P1','units':{'nodal_force':'N','moment':'Nm','traction':'N/m²'},'reference_patch_area_m2':area,'patch':'top_middle_band' if band else'right_full_face','rest_origin_m':[.12,0,0],'rest_resultant_N':actualF.tolist(),'rest_moment_Nm':actualM.tolist(),'role':'explicit_known_actuator_query','law':'Materialattachedreferencepatch,worlddeadtractionvectors;actualdeformedmomentmustberecomputed','no_material_or_BC_input':True}

def time_multiplier(name,t):
 t=np.asarray(t,dtype=float)
 if not np.isfinite(t).all() or np.any(t<0) or np.any(t>3):raise ValueError('Declared0..3scontrol required')
 if name=='ramp_hold':return np.where(t<.5,np.sin(np.pi*t)**2,1.)
 if name=='bipolar_pulse':return np.where(t<1.5,np.sin(np.pi*t/1.5)**2,-np.sin(np.pi*(t-1.5)/1.5)**2)
 raise ValueError('Prospectiveknownhistory required')

def actuator_clock(name):
 times=np.arange(361,dtype=float)/120;held=time_multiplier(name,(times[:-1]+times[1:])/2)
 return times,held
