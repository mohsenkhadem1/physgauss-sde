"""Intrinsic surface-chart posterior for the declared known cuboid asset.

The invertible density is over16 chart coefficients plus discrete categories,
including declared inactive nuisance coordinates. Decoding to25 physical fields
is noninvertible: inactive controls are removed and load tangents are clipped to
the finite face. No25-dimensional physical Lebesgue density is claimed.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
try:
    from gaussian_operator import ConditionalRealNVP,physical_vector,category_labels
except ImportError:
    from experiments.gaussian_operator import ConditionalRealNVP,physical_vector,category_labels

CHART_NAMES=('log_E_Pa','log_rho_kg_m3','nu','log_drag_s_inverse',
             'support0_u','support0_v','support1_u','support1_v',
             'load0_u','load0_v','load1_u','load1_v','force_scale_N',
             'couple_angle_rad','moving_angle_rad','moving_amplitude_m')
LOWER=np.array([math.log(1300),math.log(350),.25,math.log(.3),*([-.25]*4),*([-.55]*4),.114,-math.pi,-math.pi,.006],np.float32)
UPPER=np.array([math.log(22000),math.log(900),.45,math.log(.9),*([.25]*4),*([.55]*4),.252,math.pi,math.pi,.010],np.float32)
TOLERANCE=np.array([.015,.01,.0015,.02,*([.001]*4),*([.0015]*4),.0005,.015,.015,.00005],np.float32)


class ChartTransform(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('lower',torch.tensor(LOWER));self.register_buffer('upper',torch.tensor(UPPER))
    def inverse(self,z):return self.lower+(self.upper-self.lower)*torch.sigmoid(z)
    def forward(self,x):
        y=((x-self.lower)/(self.upper-self.lower)).clamp(1e-6,1-1e-6)
        return torch.logit(y)
    def log_jacobian(self,z):
        return (torch.log(self.upper-self.lower)+F.logsigmoid(z)+F.logsigmoid(-z)).sum(-1)


def _face_axes(face):
    table=torch.tensor([[1,2],[0,2],[0,1]],device=face.device)
    return table[face%3]


def _face_normal(face,dtype):
    return F.one_hot(face%3,3).to(dtype)*torch.where(face<3,-1.,1.)[...,None]


def _perpendicular_basis(separation):
    norm=separation.norm(dim=-1,keepdim=True)
    fallback=torch.zeros_like(separation);fallback[...,0]=1
    unit=torch.where(norm>1e-8,separation/norm.clamp_min(1e-8),fallback)
    axis=F.one_hot(unit.abs().argmin(-1),3).to(unit)
    first=F.normalize(torch.linalg.cross(unit,axis),dim=-1)
    second=torch.linalg.cross(unit,first)
    return first,second


def decode_chart(chart,boundary_type,force_type,support_face,load_faces,center,size):
    """Bounded chart→metric controls, with exact categorical force conventions."""
    center=torch.as_tensor(center,device=chart.device,dtype=chart.dtype);size=torch.as_tensor(size,device=chart.device,dtype=chart.dtype)
    def point(tangent,face,clip=False):
        coordinates=torch.zeros((*face.shape,3),device=chart.device,dtype=chart.dtype)
        coordinates.scatter_(-1,_face_axes(face),tangent.clamp(-.5,.5) if clip else tangent)
        coordinates.scatter_(-1,(face%3)[...,None],torch.where(face<3,-.5,.5)[...,None].to(chart))
        return center+coordinates*size
    support0=point(chart[...,4:6],support_face)
    support1=point(chart[...,6:8],(support_face+3)%6)
    support1=torch.where((boundary_type==2)[...,None],support1,support0)
    load0=point(chart[...,8:10],load_faces[...,0],True);load1=point(chart[...,10:12],load_faces[...,1],True)
    normal=_face_normal(load_faces[...,0],chart.dtype)
    tangent=torch.linalg.cross(normal,F.one_hot(normal.abs().argmin(-1),3).to(chart))
    first,second=_perpendicular_basis(load1-load0)
    couple=torch.cos(chart[...,13:14])*first+torch.sin(chart[...,13:14])*second
    direction=torch.where((force_type==0)[...,None],-normal,normal)
    direction=torch.where((force_type==2)[...,None],tangent,direction)
    direction=torch.where((force_type==3)[...,None],couple,direction)
    amplitude=chart[...,12]*torch.where(boundary_type==0,.25,1.)
    force0=amplitude[...,None]*direction
    force1=torch.where((force_type==3)[...,None],-force0,torch.zeros_like(force0))
    motion=torch.zeros_like(force0)
    motion.scatter_(-1,_face_axes(support_face),torch.stack((torch.cos(chart[...,14]),torch.sin(chart[...,14])),-1)*chart[...,15:16])
    motion=torch.where((boundary_type==3)[...,None],motion,torch.zeros_like(motion))
    return torch.cat((chart[...,:4],support0,support1,load0,load1,force0,force1,motion),-1)


def encode_labels(case,spec,points):
    """Supervised TRAIN/VAL labels only; never invoked by runtime inference."""
    points=np.asarray(points);center=(points.min(0)+points.max(0))/2;size=points.max(0)-points.min(0)
    raw=physical_vector(case,spec,points);bc,force=category_labels(case)
    axis,sign=spec['support_faces'][0];support_face=int(axis+(3 if sign>0 else 0))
    normals=np.asarray(spec['reference_load_normals']);axes=abs(normals).argmax(-1)
    load_faces=axes+3*(normals[np.arange(2),axes]>0)
    chart=np.zeros(16,np.float32);chart[:4]=raw[:4]
    support_axes=[i for i in range(3) if i!=axis]
    chart[4:6]=((raw[4:7]-center)/size)[support_axes]
    chart[6:8]=((raw[7:10]-center)/size)[support_axes]
    for load in range(2):
        tangent_axes=[i for i in range(3) if i!=axes[load]]
        chart[8+2*load:10+2*load]=((raw[10+3*load:13+3*load]-center)/size)[tangent_axes]
    chart[12]=spec['peak_force_N']/(.25 if bc==0 else 1.)
    first,second=_perpendicular_basis(torch.tensor(raw[13:16]-raw[10:13]))
    direction=np.asarray(spec['couple_direction_world'])
    chart[13]=np.arctan2(direction@second.numpy(),direction@first.numpy())
    motion=np.asarray(spec['motion_direction_world'])[support_axes]
    chart[14]=np.arctan2(motion[1],motion[0]);chart[15]=spec['motion_amplitude_m'] if bc==3 else .008
    if not ((chart>=LOWER-1e-6)&(chart<=UPPER+1e-6)).all():raise ValueError('Physical label outside declared chart prior')
    return {'chart':chart,'boundary_type':bc,'force_type':force,'support_face':support_face,
            'load_faces':load_faces.astype(np.int64),'physical':raw}


def dequantize_chart(chart,boundary_type,force_type,generator=None):
    """TRAIN tolerance; inactive dimensions are independent bounded nuisances."""
    noise=torch.randn(chart.shape,device=chart.device,dtype=chart.dtype,generator=generator)
    value=chart+noise*chart.new_tensor(TOLERANCE)
    uniform=torch.rand(chart.shape,device=chart.device,dtype=chart.dtype,generator=generator)
    lower=chart.new_tensor(LOWER);upper=chart.new_tensor(UPPER)
    nuisance=lower+(upper-lower)*uniform
    inactive=torch.zeros_like(chart,dtype=torch.bool)
    inactive[...,6:8]=(boundary_type!=2)[...,None]
    # Both original handle locations remain mechanically active even at zero
    # second force: the Isaac handles add mass/rigidity, as in the Green teacher.
    inactive[...,13]=(force_type!=3)
    inactive[...,14:16]=(boundary_type!=3)[...,None]
    value=torch.where(inactive,nuisance,value)
    value[...,13:15]=torch.remainder(value[...,13:15]+math.pi,2*math.pi)-math.pi
    return torch.maximum(torch.minimum(value,upper-(upper-lower)*1e-5),lower+(upper-lower)*1e-5)


class SurfaceChartPosterior(nn.Module):
    def __init__(self,context_dim=128,hidden=64,layers=4,center=(0.,0.,.25),size=(.24,.12,.08)):
        super().__init__();self.context_dim=context_dim;self.transform=ChartTransform()
        self.register_buffer('center',torch.tensor(center));self.register_buffer('size',torch.tensor(size))
        def head(extra,outputs):return nn.Sequential(nn.Linear(context_dim+extra,hidden),nn.SiLU(),nn.Linear(hidden,outputs))
        self.joint_head=head(0,16);self.support_head=head(16,6);self.load0_head=head(22,6);self.load1_head=head(28,6)
        self.flow=ConditionalRealNVP(context_dim+34,dimension=16,layers=layers,hidden=hidden)
    def categorical_logits(self,context):
        logits=self.joint_head(context).reshape(*context.shape[:-1],4,4)
        return torch.logsumexp(logits,-1),torch.logsumexp(logits,-2)
    def _conditions(self,context,bc,force,support,loads):
        joint=bc*4+force
        c0=torch.cat((context,F.one_hot(joint,16).to(context)),-1)
        c1=torch.cat((c0,F.one_hot(support,6).to(context)),-1)
        c2=torch.cat((c1,F.one_hot(loads[...,0],6).to(context)),-1)
        c3=torch.cat((c2,F.one_hot(loads[...,1],6).to(context)),-1)
        return joint,c0,c1,c2,c3
    def discrete_log_prob(self,context,bc,force,support,loads):
        joint,c0,c1,c2,_=self._conditions(context,bc,force,support,loads)
        def selected(logits,label):return logits.log_softmax(-1).gather(-1,label[...,None]).squeeze(-1)
        return (selected(self.joint_head(context),joint)+selected(self.support_head(c0),support)
                +selected(self.load0_head(c1),loads[...,0])+selected(self.load1_head(c2),loads[...,1]))
    def log_prob_chart(self,chart,context,bc,force,support,loads,include_categories=True):
        _,_,_,_,condition=self._conditions(context,bc,force,support,loads)
        z=self.transform(chart);result=self.flow.log_prob(z,condition)-self.transform.log_jacobian(z)
        if include_categories:result=result+self.discrete_log_prob(context,bc,force,support,loads)
        valid=((chart>self.transform.lower)&(chart<self.transform.upper)).all(-1)
        return torch.where(valid,result,torch.full_like(result,-torch.inf))
    def sample(self,context,samples=32,boundary_type=None,force_type=None,support_face=None,load_faces=None,generator=None):
        b=len(context);expanded=context[:,None].expand(-1,samples,-1)
        def sample_head(logits):
            p=logits.softmax(-1);return torch.multinomial(p.reshape(-1,p.shape[-1]),1,generator=generator).reshape(b,samples)
        joint_logits=self.joint_head(expanded).reshape(b,samples,4,4)
        if boundary_type is None and force_type is None:
            joint=sample_head(joint_logits.flatten(-2));bc=joint//4;force=joint%4
        elif boundary_type is not None and force_type is not None:
            bc=boundary_type[:,None].expand(-1,samples);force=force_type[:,None].expand(-1,samples);joint=bc*4+force
        else:raise ValueError('Provide both known BC/load categories or neither')
        c0=torch.cat((expanded,F.one_hot(joint,16).to(context)),-1)
        support=sample_head(self.support_head(c0)) if support_face is None else support_face[:,None].expand(-1,samples)
        c1=torch.cat((c0,F.one_hot(support,6).to(context)),-1)
        load0=sample_head(self.load0_head(c1)) if load_faces is None else load_faces[:,0,None].expand(-1,samples)
        c2=torch.cat((c1,F.one_hot(load0,6).to(context)),-1)
        load1=sample_head(self.load1_head(c2)) if load_faces is None else load_faces[:,1,None].expand(-1,samples)
        loads=torch.stack((load0,load1),-1);condition=torch.cat((c2,F.one_hot(load1,6).to(context)),-1)
        noise=torch.randn(b,samples,16,device=context.device,dtype=context.dtype,generator=generator)
        z,_=self.flow.inverse(noise,condition);chart=self.transform.inverse(z)
        physical=decode_chart(chart,bc,force,support,loads,self.center,self.size)
        return {'physical':physical,'effective_physical':physical,'chart':chart,'boundary_type':bc,'force_type':force,
                'support_face':support,'load_faces':loads,'log_prob_intrinsic':self.log_prob_chart(chart,expanded,bc,force,support,loads)}
