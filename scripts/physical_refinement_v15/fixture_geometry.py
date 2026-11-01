"""Visible graded-area ligament representation of the SAME homogenized Robin footprint.
This is effective isotropic support geometry, not resolved ligament elasticity.
"""
import numpy as np
from scripts.position_ood.smooth_fixture import footprint

def graded_fixture(kind,left,grid=10,total_area_m2=.0008,length_m=.012):
    # Midpoint cells represent a graded attached-area fraction. Equal material and
    # ligament length imply cross-section area follows the continuum footprint.
    centers=(np.arange(grid)+.5)*.08/grid-.04
    yz=np.stack(np.meshgrid(centers,centers,indexing='ij'),-1).reshape(-1,2)
    weight=footprint(yz[:,0],yz[:,1],kind);areas=total_area_m2*weight/weight.sum()
    radii=np.sqrt(areas/(2*np.sqrt(2.)))
    if np.any(radii>.5*.08/grid):raise ValueError('Ligaments overlap; decrease declared total area')
    vertices=[];faces=[];angles=np.arange(8)*2*np.pi/8
    x=0. if left else .24;outer=x+(-length_m if left else length_m)
    for (y,z),radius in zip(yz,radii):
        begin=len(vertices)
        for xx in [x,outer]:vertices.extend(np.c_[np.full(8,xx),y+radius*np.cos(angles),z+radius*np.sin(angles)].tolist())
        for j in range(8):
            k=(j+1)%8;faces.extend([[begin+j,begin+k,begin+8+k],[begin+j,begin+8+k,begin+8+j]])
        for j in range(1,7):faces.extend([[begin,begin+j,begin+j+1],[begin+8,begin+8+j+1,begin+8+j]])
    return dict(vertices=np.asarray(vertices),triangles=np.asarray(faces),color=np.array([.45,.45,.47])),dict(yz_m=yz,normalized_area=areas/areas.sum(),cross_section_area_m2=areas,footprint_values=weight,interpretation='Fixed-length visible ligament cross sections sample exact normalized Gaussian*taper attached-area density. Continuum Robin law is homogenized; finite rendered ligaments do not resolve isotropic fixture elasticity. Unknown totalK has no appearance effect.')

def layout_fixtures(layout):
    if layout not in ('parallel_lower','crossed'):raise ValueError('Undeclared support layout')
    a,ma=graded_fixture('lower',True);b,mb=graded_fixture('lower'if layout=='parallel_lower'else'upper',False)
    return [a,b],[ma,mb]
