"""Check correction changes only the mask, and versioned exports preserve old inputs."""
import json,hashlib
import numpy as np
from .campaign import ROOT,CASES,write

def main():
    rows=[]
    for case in CASES:
        old=ROOT/'runs/real-transfer-v19/inputs'/f'{case}-240/prefix.npz'
        new=ROOT/'runs/real-transfer-v19/inputs/incoming-v22'/f'{case}-240/prefix.npz'
        with np.load(old) as a,np.load(new) as b:
            assert set(a.files)==set(b.files)
            changed=[k for k in a.files if not np.array_equal(a[k],b[k],equal_nan=True)]
            assert changed==['observed_mask'],changed
            rows.append(dict(case=case,changed_fields=changed,changed_mask_entries=int((a['observed_mask']!=b['observed_mask']).sum()),
                mask_entries=int(a['observed_mask'].size),old_valid=int(a['observed_mask'].sum()),new_valid=int(b['observed_mask'].sum()),
                old_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),new_sha256=hashlib.sha256(new.read_bytes()).hexdigest()))
    write(ROOT/'reports/real-transfer-v22/input-audit.json',dict(passed=True,cases=rows,
        scope='All exported numeric arrays identical except observed_mask. This checks adapter indexing, not causality of upstream offline tracking.'))
    print('All eight exports differ only in observed_mask')

if __name__=='__main__':main()
