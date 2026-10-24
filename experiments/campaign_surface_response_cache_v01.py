"""Optional exact-node LOCAL response tables, never an interpolated surrogate.

A hash-bound explicit admission receipt is required before any table is built.
This module contains no actual model, cohort, observations, or execution.
"""
import hashlib
import json
import mmap
import shutil
from pathlib import Path
import numpy as np
from experiments.campaign_surface_p2_inference_v01 import LabelledModal, IntegrationPlan

RSS_LIMIT_BYTES=4*1024**3
DISK_RESERVE_BYTES=30*1024**3


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def model_fingerprint(model):
    h=hashlib.sha256(json.dumps(dict(labels=model.labels,channels=model.channels),sort_keys=True).encode())
    for array in (model.modal.eigenvalues,model.modal.coefficients):
        value=np.ascontiguousarray(array,dtype='<f8')
        h.update(str(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()


def _rss():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    raise RuntimeError('Current RSS unavailable; cannot enforce cache budget')


def _sources():
    import experiments.campaign_surface_p2_inference_v01 as engine
    return {str(Path(__file__).resolve()):_sha(__file__),str(Path(engine.__file__).resolve()):_sha(engine.__file__)}


def _specifications(plan):
    finest=plan.simpson[-1]
    if any((finest-1)%(n-1) for n in plan.simpson):raise ValueError('Nested Simpson levels required')
    yield 'Simpson',finest,np.linspace(0,1,finest)
    for n in plan.gauss:
        axis,_=np.polynomial.legendre.leggauss(n)
        yield 'Gauss',n,(axis+1)/2


def build_cache(model,directory,*,admission_receipt,admission_SHA256,plan=IntegrationPlan()):
    """Build shared memmaps only after explicit receipt and resource checks.

    Receipt JSON must contain cache_admission=true, model_fingerprint, a
    protocol_SHA256, and scope qualified_physics_catalog or synthetic_controls.
    Synthetic scope is capped at65nodes/axis. Actual physics qualification is
    an external scientific decision; exact model bytes are bound here.
    """
    admission_receipt=Path(admission_receipt)
    if _sha(admission_receipt)!=admission_SHA256:raise ValueError('Admission receipt SHA mismatch')
    admission=json.loads(admission_receipt.read_text())
    fingerprint=model_fingerprint(model)
    if admission.get('cache_admission') is not True or admission.get('model_fingerprint')!=fingerprint:
        raise ValueError('Explicit admission and exact model fingerprint required')
    if not isinstance(admission.get('protocol_SHA256'),str) or len(admission['protocol_SHA256'])!=64 or any(c not in '0123456789abcdef' for c in admission['protocol_SHA256']):
        raise ValueError('Protocol SHA256 required')
    if admission.get('scope') not in ('qualified_physics_catalog','synthetic_controls'):
        raise ValueError('Physics-catalog or bounded synthetic admission required')
    if admission['scope']=='synthetic_controls' and max(*plan.simpson,*plan.gauss)>65:
        raise ValueError('Synthetic cache cannot allocate actual study tables')
    directory=Path(directory)
    if directory.exists() and any(directory.iterdir()):raise FileExistsError('New empty cache directory required; no overwrite/resume')
    directory.mkdir(parents=True,exist_ok=True)
    specifications=list(_specifications(plan));R=len(model.labels);Q=model.channels
    projected=sum(n*n*R*Q*8 for _,n,_ in specifications)
    workspace=plan.chunk*(R*Q+len(model.modal.eigenvalues)+6)*8*4+64*1024**2
    if _rss()+workspace>=RSS_LIMIT_BYTES:raise MemoryError('Current RSS plus conservative chunk workspace exceeds4GiB')
    if shutil.disk_usage(directory).free<projected+DISK_RESERVE_BYTES:
        raise OSError('Projected cache plus30GiB disk reserve unavailable')
    manifest=dict(schema='surface-response-cache-v01',status='INCOMPLETE',model_fingerprint=fingerprint,
        labels=list(model.labels),channels=Q,plan=dict(simpson=list(plan.simpson),gauss=list(plan.gauss),chunk=plan.chunk),
        source_SHA256=_sources(),admission_receipt_SHA256=admission_SHA256,admission=admission,
        projected_table_bytes=projected,RSS_limit_bytes=RSS_LIMIT_BYTES,disk_reserve_bytes=DISK_RESERVE_BYTES,
        interpolation=False,axis_lookup='Exact floating-point equality only; otherwise exact modal fallback',tables=[])
    manifestpath=directory/'manifest.json';manifestpath.write_text(json.dumps(manifest,indent=2)+'\n')
    predict=model.select(model.labels)
    for family,n,axis in specifications:
        name=f'{family}-{n}.npy';path=directory/name
        table=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=(n*n,R,Q))
        for start in range(0,n*n,plan.chunk):
            if _rss()+workspace>=RSS_LIMIT_BYTES:raise MemoryError('Cache write RSS budget exhausted')
            if shutil.disk_usage(directory).free<DISK_RESERVE_BYTES:raise OSError('Cache write disk reserve exhausted')
            index=np.arange(start,min(start+plan.chunk,n*n))
            unit=np.column_stack((axis[index//n],axis[index%n]))
            values=np.asarray(predict(unit),float)
            if values.shape!=(len(index),R,Q) or not np.isfinite(values).all():raise FloatingPointError('Invalid exact cache responses')
            table[start:start+len(index)]=values
        table.flush();del table
        manifest['tables'].append(dict(family=family,n=n,path=name,shape=[n*n,R,Q],SHA256=_sha(path)))
        manifestpath.write_text(json.dumps(manifest,indent=2)+'\n')
    manifest['status']='COMPLETE'
    manifestpath.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


class CachedLabelledModal(LabelledModal):
    """Share parent independent-data likelihood with exact-node cached select."""
    def __init__(self,model,directory,*,manifest_SHA256):
        super().__init__(model.modal,model.labels)
        directory=Path(directory);manifestpath=directory/'manifest.json'
        if _sha(manifestpath)!=manifest_SHA256:raise ValueError('Cache manifest SHA mismatch')
        manifest=json.loads(manifestpath.read_text())
        if manifest.get('status')!='COMPLETE' or manifest.get('model_fingerprint')!=model_fingerprint(model):
            raise ValueError('Complete exact-model cache required')
        if manifest.get('source_SHA256')!=_sources():raise ValueError('Cache/engine source changed; no stale reuse')
        if manifest.get('labels')!=list(model.labels) or manifest.get('channels')!=model.channels:
            raise ValueError('Cache output layout mismatch')
        self._fallback_model=model
        self.tables=[]
        for entry in manifest['tables']:
            path=directory/entry['path']
            if path.parent!=directory or _sha(path)!=entry['SHA256']:raise ValueError('Cache file path/hash mismatch')
            table=np.load(path,mmap_mode='r',allow_pickle=False);n=entry['n']
            if table.dtype!=np.float64 or table.shape!=(n*n,len(self.labels),self.channels):raise ValueError('Cache table layout mismatch')
            if entry['family']=='Simpson':axis=np.linspace(0,1,n)
            elif entry['family']=='Gauss':axis=(np.polynomial.legendre.leggauss(n)[0]+1)/2
            else:raise ValueError('Unknown cache quadrature rule')
            self.tables.append((axis,table))
        self.manifest=manifest

    def select(self,labels):
        labels=tuple(labels);fallback=self._fallback_model.select(labels)
        indices=[self.labels.index(label) for label in labels]
        def predict(unit):
            u=np.asarray(unit,float)
            if u.ndim!=2 or u.shape[1]!=2 or not np.isfinite(u).all() or np.any((u<0)|(u>1)):
                raise ValueError('Finite unit-square coordinates')
            if _rss()+len(u)*(len(labels)*self.channels+len(self.modal.eigenvalues)+6)*8*4+64*1024**2>=RSS_LIMIT_BYTES:
                raise MemoryError('Cached response read RSS budget exhausted')
            output=np.empty((len(u),len(labels),self.channels));missing=np.ones(len(u),bool)
            for axis,table in self.tables:
                rows=np.flatnonzero(missing)
                if not len(rows):break
                x=u[rows];position=np.searchsorted(axis,x);safe=np.minimum(position,len(axis)-1)
                match=np.all((position<len(axis))&(axis[safe]==x),axis=1)
                hit=rows[match]
                if len(hit):
                    flat=safe[match,0]*len(axis)+safe[match,1]
                    output[hit]=table[flat][:,indices,:];missing[hit]=False
                    # Output has copied these read-only pages. Release mapped
                    # residency so scanning2.7GiB across objects stays bounded.
                    table._mmap.madvise(mmap.MADV_DONTNEED)
            if np.any(missing):output[missing]=fallback(u[missing])
            return output
        return predict
