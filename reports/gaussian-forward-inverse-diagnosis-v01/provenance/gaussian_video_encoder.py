"""Small RGB/mask and calibrated-camera encoder, without pretrained downloads."""
import torch
from torch import nn


class GaussianVideoEncoder(nn.Module):
    """Map public video inputs to 128 context features for inverse inference.

    Image input is [B,T,4,H,W]: RGB in [0,1] and supplied foreground mask.
    Poses are USD camera-to-world row matrices [B,T,4,4] or [T,4,4].
    Static reference nodes [N,3] are known metric rest geometry; neither dynamic
    positions nor material/support labels are arguments. Missing frames must be
    absent or marked false in observation_mask [B,T].
    """
    def __init__(self, context_dim=128):
        super().__init__()
        self.cnn=nn.Sequential(
            nn.Conv2d(4,24,5,stride=2,padding=2),nn.SiLU(),
            nn.Conv2d(24,48,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(48,64,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(64,96,3,stride=2,padding=1),nn.SiLU(),
            nn.AdaptiveAvgPool2d((2,2)),nn.Flatten(),nn.Linear(384,128),nn.SiLU())
        self.camera=nn.Sequential(nn.Linear(18,64),nn.SiLU(),nn.Linear(64,128))
        self.geometry=nn.Sequential(nn.Linear(9,64),nn.SiLU(),nn.Linear(64,128))
        layer=nn.TransformerEncoderLayer(128,4,256,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.output=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,context_dim))

    def forward(self,images,poses,k,times,rest_points,observation_mask=None,calibration_size=(640,480)):
        if images.ndim!=5 or images.shape[2]!=4:
            raise ValueError('Expected[B,T,4,H,W] RGB plus mask')
        b,t=images.shape[:2];device=images.device;dtype=images.dtype
        poses=torch.as_tensor(poses,device=device,dtype=dtype)
        poses=torch.broadcast_to(poses,(b,t,4,4))
        k=torch.as_tensor(k,device=device,dtype=dtype)
        if k.ndim==3 and k.shape[0]==b:k=k[:,None]
        k=torch.broadcast_to(k,(b,t,3,3))
        times=torch.as_tensor(times,device=device,dtype=dtype)
        times=torch.broadcast_to(times,(b,t))
        rest=torch.as_tensor(rest_points,device=device,dtype=dtype)
        rest=torch.broadcast_to(rest,(b,*rest.shape[-2:]))
        lower=rest.amin(-2);upper=rest.amax(-2);center=(lower+upper)/2
        extent=(upper-lower).clamp_min(1e-5);scale=extent.norm(dim=-1,keepdim=True)
        relative_eye=(poses[:,:,3,:3]-center[:,None])/scale[:,None]
        rotation=poses[:,:,:3,:3].flatten(-2)
        calibration=torch.stack((k[:,:,0,0]/calibration_size[0],k[:,:,1,1]/calibration_size[1],k[:,:,0,2]/calibration_size[0],k[:,:,1,2]/calibration_size[1]),-1)
        temporal=torch.stack((times/3.,torch.sin(times*torch.pi/3.)),-1)
        camera=torch.cat((relative_eye,rotation,calibration,temporal),-1)
        geometry=torch.cat((center,extent,rest.std(-2,unbiased=False)),-1)
        encoded=self.cnn(images.reshape(b*t,*images.shape[2:])).reshape(b,t,128)
        encoded=encoded+self.camera(camera)+self.geometry(geometry)[:,None]
        if observation_mask is None:observation_mask=torch.ones((b,t),device=device,dtype=torch.bool)
        observation_mask=torch.as_tensor(observation_mask,device=device,dtype=torch.bool)
        if not observation_mask.any(-1).all():raise ValueError('Each clip needs an observation')
        contextual=self.temporal(encoded,src_key_padding_mask=~observation_mask)
        pooled=(contextual*observation_mask[:,:,None]).sum(1)/observation_mask.sum(1,keepdim=True)
        return self.output(pooled)
