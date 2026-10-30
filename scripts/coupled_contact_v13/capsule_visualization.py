"""Development capsule geometry drawing; no production scene bindings."""
import numpy as np


def capsule_bounds(collider):
    half=.5*collider.segment_length
    extent=half*np.abs(collider.rotation[:,1])+collider.radius
    return collider.center-extent,collider.center+extent


def capsule_surfaces(collider,resolution=28):
    phi=np.linspace(0,2*np.pi,resolution+1)
    parts=[];half=collider.segment_length/2;r=collider.radius
    angle,y=np.meshgrid(phi,[-half,half])
    parts.append(np.stack([r*np.cos(angle),y,r*np.sin(angle)],axis=-1))
    angle,theta=np.meshgrid(phi,np.linspace(0,np.pi/2,resolution//2+1))
    for sign in (-1,1):
        parts.append(np.stack([r*np.sin(theta)*np.cos(angle),sign*(half+r*np.cos(theta)),r*np.sin(theta)*np.sin(angle)],axis=-1))
    return [np.moveaxis(part@collider.rotation.T+collider.center,-1,0) for part in parts]


def draw_capsule(ax,collider,**kwargs):
    for x,y,z in capsule_surfaces(collider):ax.plot_surface(x,y,z,**kwargs)
