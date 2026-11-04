"""Small shared visual scene encoder and independent point-query decoder.

Known calibration; predicts geometry from RGB features and source UV, no mesh.
Reconstruction uses target-time images, not a causal physical forecast.
"""
import torch
from torch import nn
from torch.nn import functional as F

class SceneQueryModel(nn.Module):
    def __init__(self,image_blind=False,width=128):
        super().__init__();self.image_blind=image_blind
        self.visual=nn.Linear(256,width);self.token_geometry=nn.Sequential(nn.Linear(19,width),nn.GELU(),nn.Linear(width,width))
        layer=nn.TransformerEncoderLayer(width,4,256,dropout=0.,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.query=nn.Sequential(nn.Linear(256+16+3+12+1,width),nn.GELU(),nn.Linear(width,width))
        self.attention=nn.ModuleList([nn.MultiheadAttention(width,4,dropout=0.,batch_first=True)for _ in range(2)])
        self.norm=nn.ModuleList([nn.LayerNorm(width)for _ in range(2)])
        self.ffn=nn.ModuleList([nn.Sequential(nn.Linear(width,256),nn.GELU(),nn.Linear(256,width))for _ in range(2)])
        self.depth=nn.Linear(width,1);self.delta=nn.Linear(width,3)
        nn.init.zeros_(self.depth.weight);nn.init.constant_(self.depth.bias,-.4)
        nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
    def forward(self,features,token_geometry,uv,rays,camera0,time_s):
        # features B,4,256,12,12: source camera0/1 then target camera0/1.
        if self.image_blind:features=torch.zeros_like(features)
        B=features.shape[0];tokens=features.permute(0,1,3,4,2).reshape(B,-1,256)
        memory=self.encoder(self.visual(tokens)+self.token_geometry(token_geometry))
        local=F.grid_sample(features[:,0],uv[:,None]*2-1,align_corners=False).squeeze(2).transpose(1,2)
        angles=uv[...,None]*uv.new_tensor([1.,2.,4.,8.])*torch.pi
        fourier=torch.cat([angles.sin(),angles.cos()],-1).flatten(-2)
        pose=camera0[:,:3,:].reshape(B,1,12).expand(-1,uv.shape[1],-1)
        q=self.query(torch.cat([local,fourier,rays,pose,time_s[:,None,None].expand(-1,uv.shape[1],1)],-1))
        for attn,norm,ffn in zip(self.attention,self.norm,self.ffn):
            q=q+attn(norm(q),memory,memory,need_weights=False)[0];q=q+ffn(norm(q))
        depth=.1+F.softplus(self.depth(q))
        source=torch.einsum('bqj,bij->bqi',rays*depth,camera0[:,:3,:3])+camera0[:,None,:3,3]
        delta=.1*self.delta(q)*torch.clamp(time_s[:,None,None]/.1,max=1.)
        return source+delta,source
