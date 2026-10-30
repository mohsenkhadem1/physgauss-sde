"""Compiled exact P1 normal contact for planes/boxes, refined friction quadrature.

The force integral uses REST boundary-triangle area. Sparse material quadrature
is retained for spheres/bowls. No silent numerical fallback is permitted.
"""
from pathlib import Path
import ctypes,fcntl,hashlib,json,os,subprocess,shutil
import numpy as np
from .sparse_surface_contact import SparseSurfaceContactMixin,SparseSurfaceFEMSystem,with_sparse_surface_contact
ROOT=Path(__file__).resolve().parents[2]
_LIBRARY=None


def load_kernel():
    global _LIBRARY
    if _LIBRARY is not None:return _LIBRARY
    source=Path(__file__).with_suffix('.cpp');source_bytes=source.read_bytes();source_hash=hashlib.sha256(source_bytes).hexdigest()
    compiler=os.environ.get('CXX') or shutil.which('c++') or shutil.which('g++')
    bundled=Path.home()/'.local/share/physgauss-external/pidg-official-v01/env/bin/x86_64-conda-linux-gnu-c++'
    if compiler is None and bundled.exists():compiler=str(bundled)
    if compiler is None:raise RuntimeError('A C++17 compiler is required; set CXX explicitly')
    compiler=str(Path(compiler).resolve());flags=['-std=c++17','-O3','-fPIC','-shared']
    version=subprocess.run([compiler,'--version'],capture_output=True,text=True,check=True).stdout
    key=hashlib.sha256(source_bytes+compiler.encode()+version.encode()+json.dumps(flags).encode()).hexdigest()
    folder=ROOT/'runs/coupled-contact-v13/tools/clipped-contact'/key;folder.mkdir(parents=True,exist_ok=True);library=folder/'contact.so'
    with (folder/'build.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        snapshot=folder/'source.cpp'
        if snapshot.exists() and hashlib.sha256(snapshot.read_bytes()).hexdigest()!=source_hash:raise RuntimeError('Compiler source snapshot changed')
        if not snapshot.exists():snapshot.write_bytes(source_bytes)
        if not library.exists():
            temp=folder/f'contact-{os.getpid()}.so';command=[compiler,*flags,str(snapshot),'-o',str(temp)]
            done=subprocess.run(command,capture_output=True,text=True)
            if done.returncode:raise RuntimeError('Clipped contact compilation failed: '+done.stderr)
            temp.replace(library)
            (folder/'build.json').write_text(json.dumps(dict(source=str(source),source_sha256=source_hash,compiler_path=compiler,compiler=version,flags=flags,command=command,library_sha256=hashlib.sha256(library.read_bytes()).hexdigest()),indent=2)+'\n')
    metadata=json.loads((folder/'build.json').read_text())
    if metadata.get('source_sha256')!=source_hash or metadata.get('compiler_path')!=compiler or metadata.get('compiler')!=version or metadata.get('flags')!=flags:raise RuntimeError('Compiled contact metadata does not match requested build')
    if hashlib.sha256(library.read_bytes()).hexdigest()!=metadata['library_sha256']:raise RuntimeError('Compiled contact cache hash mismatch')
    lib=ctypes.CDLL(str(library));fn=lib.integrate_contact
    D=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');I=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    fn.argtypes=[ctypes.c_int,ctypes.c_int,I,D,D,D,ctypes.c_int,D,D,D,D,D,D,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,D,D,D,D];fn.restype=ctypes.c_int
    _LIBRARY=(fn,metadata);return _LIBRARY


def integrate(system,collider,friction_level=0):
    if collider.kind not in ('plane','box') or np.isfinite(collider.mass):raise ValueError('Analytic contact requires prescribed/static plane or box')
    fn,_=load_kernel();tri=np.ascontiguousarray(system.surface_triangles,dtype=np.int64);rest=system.rest[tri]
    area=np.ascontiguousarray(.5*np.linalg.norm(np.cross(rest[:,1]-rest[:,0],rest[:,2]-rest[:,0]),axis=1))
    kd=collider.stiffness/.0004;mass_density=system.mass.sum()/area.sum();cd=2*collider.damping_ratio*np.sqrt(kd*mass_density)
    force=np.zeros_like(system.x);reaction=np.zeros(3);torque=np.zeros(3);stats=np.zeros(9)
    arrays=[np.ascontiguousarray(a,dtype=np.float64) for a in [system.x,system.v,collider.center,collider.rotation,collider.half_extents,collider.normal,collider.velocity,collider.omega]]
    x,v,c,R,h,n,vel,omega=arrays
    status=fn(len(system.x),len(tri),tri,area,x,v,0 if collider.kind=='plane' else 1,c,R,h,n,vel,omega,kd,cd,collider.friction,collider.friction_speed,friction_level,force,reaction,torque,stats)
    if status or not all(np.isfinite(a).all() for a in (force,reaction,torque,stats)):raise FloatingPointError(f'Clipped contact kernel failed with status {status}')
    return force,reaction,torque,stats


class ClippedSurfaceContactMixin:
    friction_quadrature_level=0

    def contact_forces(self):
        all_colliders=self.colliders;other=[(i,c) for i,c in enumerate(all_colliders) if c.kind not in ('plane','box')]
        self.colliders=[c for _,c in other]
        try:force,rr,tt,depth,power,count=super().contact_forces()
        finally:self.colliders=all_colliders
        reactions=[np.zeros(3) for _ in all_colliders];torques=[np.zeros(3) for _ in all_colliders]
        for (i,_),r,t in zip(other,rr,tt):reactions[i]=r;torques[i]=t
        for i,c in enumerate(all_colliders):
            if c.kind not in ('plane','box'):continue
            f,r,t,stats=integrate(self,c,self.friction_quadrature_level)
            force+=f;reactions[i]=r;torques[i]=t;depth=max(depth,float(stats[0]));power+=float(stats[1]);count+=int(stats[2])
            self._last_friction_active+=int(stats[3]);self._last_friction_saturated+=int(stats[4])
        return force,reactions,torques,depth,power,count


class ClippedSurfaceFEMSystem(ClippedSurfaceContactMixin,SparseSurfaceFEMSystem):
    pass


def with_clipped_surface_contact(base):
    if issubclass(base,ClippedSurfaceContactMixin):return base
    return type('ClippedSurface'+base.__name__,(ClippedSurfaceContactMixin,with_sparse_surface_contact(base)),{})
