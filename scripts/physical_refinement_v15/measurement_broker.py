"""Selected-only access to precommitted synthetic measurement packets.

The policy receives a request callback, never a dictionary of actual candidate
outcomes. This is a reproducible data-flow contract, not an OS security sandbox.
Packet generation and target response scoring run separately from selection.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MeasurementBroker:
    def __init__(self,manifest,expected_sha256,*,packet_root,budget=2):
        manifest=Path(manifest).resolve();self.root=Path(packet_root).resolve()
        if digest(manifest)!=expected_sha256:raise ValueError('Measurement commitment changed')
        data=json.loads(manifest.read_text())
        if data['role']!='synthetic_observed_packets' or 'target_responses' in data:
            raise ValueError('Only measurement packets admitted')
        self.entries=data['packets'];self.history=[];self.budget=budget
        if 'initial' not in self.entries:raise ValueError('Initial packet required')
        self.log=[]

    def request(self,name):
        if name not in self.entries or name in self.history:raise ValueError('Unavailable or repeated measurement')
        if not self.history and name!='initial':raise ValueError('Initial observation must be consumed first')
        if self.history and len(self.history)-1>=self.budget:raise ValueError('Acquisition budget exhausted')
        entry=self.entries[name];path=(self.root/entry['path']).resolve()
        if not path.is_relative_to(self.root):raise ValueError('Packet path outside measurement root')
        if digest(path)!=entry['sha256']:raise ValueError('Measurement packet changed')
        with np.load(path,allow_pickle=False) as packet:
            if set(packet.files)!={'observation'}:raise ValueError('Packet must contain only noisy observation')
            value=packet['observation'].copy()
        if value.shape!=(entry['dimension'],) or not np.isfinite(value).all():
            raise ValueError('Invalid observation vector')
        self.history.append(name)
        self.log.append(dict(name=name,sha256=entry['sha256'],dimension=len(value)))
        return value

    def receipt(self):
        return dict(budget=self.budget,consumed=list(self.history),packets=list(self.log),
                    scope='Selected-only data flow, not operating-system isolation')
