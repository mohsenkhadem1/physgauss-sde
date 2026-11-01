"""Fixed image features and explicit correlated synthetic sensor covariance.
Illustrative sensitivity model, not calibrated hardware or posterior coverage.
"""
import numpy as np
from scipy.ndimage import shift
from scipy.special import ndtr

GRID=4
FAR_CLIP=1.
NOISE_LEVELS=((.0001,.002),(.0003,.005),(.001,.01))

def features(rgb,depth):
    H,W=depth.shape
    if H%GRID or W%GRID:raise ValueError('Image dimensions must divide fixed grid')
    image=np.concatenate([rgb,np.where(np.isfinite(depth),depth,FAR_CLIP)[...,None]],axis=2)
    return image.reshape(GRID,H//GRID,GRID,W//GRID,4).mean((1,3)).reshape(-1)

def covariance(rgb,depth,depth_sd,rgb_sd):
    H,W=depth.shape;N=(H//GRID)*(W//GRID)
    yy,xx=np.meshgrid((np.arange(GRID)+.5)*H/GRID,(np.arange(GRID)+.5)*W/GRID,indexing='ij');points=np.c_[xx.ravel(),yy.ravel()]
    distance=np.sum((points[:,None]-points[None])**2,axis=2)
    # Whole-image offset + smooth tile bias + independent pixel component.
    spatial=np.ones((GRID*GRID,GRID*GRID))+np.exp(-distance/(2*64.**2))+np.eye(GRID*GRID)/N
    scales=np.array([rgb_sd]*3+[depth_sd]);C=np.kron(spatial,np.diag(scales**2))
    image=np.concatenate([rgb,np.where(np.isfinite(depth),depth,FAR_CLIP)[...,None]],axis=2)
    derivatives=[]
    for axis in [0,1]:
        delta=[0.,0.,0.];delta[axis]=1.
        plus=shift(image,delta,order=1,mode='nearest');minus=shift(image,[-x for x in delta],order=1,mode='nearest')
        derivatives.append((features(plus[:,:,:3],plus[:,:,3])-features(minus[:,:,:3],minus[:,:,3]))/2)
    J=np.asarray(derivatives).T;C+=.5**2*(J@J.T)
    return C

def compare(a,b):
    delta=features(a['rgb'],a['depth'])-features(b['rgb'],b['depth']);rows=[]
    for sd,rgb in NOISE_LEVELS:
        C=(covariance(a['rgb'],a['depth'],sd,rgb)+covariance(b['rgb'],b['depth'],sd,rgb))/2
        # Shared nuisance covariance yields the stated equal-covariance Gaussian KL.
        d2=float(delta@np.linalg.solve(C,delta));idx=np.arange(3,len(delta),4);dd=delta[idx];Cd=C[np.ix_(idx,idx)];depth_d2=float(dd@np.linalg.solve(Cd,dd))
        rows.append(dict(depth_noise_sd_m=sd,rgb_noise_sd=rgb,mahalanobis_squared=d2,gaussian_KL_nats=.5*d2,depth_only_mahalanobis_squared=depth_d2,equal_prior_gaussian_classification_error=float(ndtr(-np.sqrt(max(0,d2))/2))))
    return dict(feature_dimensions=len(delta),noise_sweep=rows,assumptions='Pooled equal covariance; global+64pixel smooth bias, independentpixel component,0.5pixel registrationjitter. Synthetic prescribed noise, nothardwarecalibration. Tilemeanfeatures may discard information; large/small divergence only concerns this sensor-feature model.')
