"""Render exact saved Q2 mechanical state via fixed material-boundary sampling."""
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.physical_refinement_v15.fixture_geometry import layout_fixtures

def boundary_grid(cells=(24,8,8)):
    axes=[np.linspace(0,.24,cells[0]+1),np.linspace(-.04,.04,cells[1]+1),np.linspace(-.04,.04,cells[2]+1)];vertices=[];faces=[]
    for axis in range(3):
        other=[i for i in range(3)if i!=axis]
        for side in [0,-1]:
            begin=len(vertices);shape=[len(axes[j])for j in other]
            for a in axes[other[0]]:
                for b in axes[other[1]]:
                    p=np.zeros(3);p[axis]=axes[axis][side];p[other]=[a,b];vertices.append(p)
            for i in range(shape[0]-1):
                for j in range(shape[1]-1):
                    q=[begin+i*shape[1]+j,begin+(i+1)*shape[1]+j,begin+(i+1)*shape[1]+j+1,begin+i*shape[1]+j+1];faces.extend([[q[0],q[1],q[2]],[q[0],q[2],q[3]]])
    return np.asarray(vertices),np.asarray(faces)

def scene_from_state(model,u,layout,poses):
    rest,faces=boundary_grid();positions=rest+model.query_field(u,rest)
    # Fixed material texture independent of modulus/support stiffness.
    color=.25+.5*(rest-np.array([0,-.04,-.04]))/np.array([.24,.08,.08])
    meshes=[dict(vertices=positions,triangles=faces,color=color)]
    fixtures,_=layout_fixtures(layout)
    for i,fixture in enumerate(fixtures):
        p=fixture['vertices'];x=0. if i==0 else .24;inner=np.isclose(p[:,0],x);q=p[inner].copy();q[:,0]=x
        positions=p.copy();positions[inner]=q+model.query_field(u,q)
        center=np.array([x,0,0]);R=Rotation.from_rotvec(poses[i,3:]).as_matrix();positions[~inner]=(p[~inner]-center)@R.T+center+poses[i,:3]
        meshes.append(dict(fixture,vertices=positions))
    return meshes
