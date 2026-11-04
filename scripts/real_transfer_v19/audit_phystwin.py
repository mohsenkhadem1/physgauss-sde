"""Independent manual-track arithmetic and input-isolation checks."""
import json,pickle
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial import KDTree
from scripts.external_benchmark.data import phystwin,predictor_payload
from scripts.external_benchmark.evaluate import expand_prediction
from scripts.real_transfer_v19.phystwin_run import ROOT,BASE,write,sha

def main():
    report=ROOT/'reports/real-transfer-v19';scores=json.loads((report/'exploratory-01-scores.json').read_text())
    checks=0;recomputed={m:[] for m in ['neural','analytic','official_phystwin']}
    for row in scores['rows']:
        case=row['case'];data=phystwin(case,120);truth=data['evaluation']['manual'];start=data['train_end']
        for method in recomputed:
            if method=='official_phystwin':
                path=ROOT/'runs/third_party/PhysTwin/reference/experiments'/case/'inference.pkl'
                pred=np.asarray(pickle.load(path.open('rb')))@data['evaluation']['transform']
            else:
                folder=BASE/'fits/exploratory-01'/case/method
                done=json.loads((folder/'complete.json').read_text());path=folder/'prediction.npz'
                assert sha(path)==done['prediction_sha256'];checks+=1
                pred=expand_prediction(data,np.load(path)['predicted'])
            valid0=np.isfinite(truth[0]).all(1);ids=KDTree(pred[0]).query(truth[0,valid0])[1];errors=[]
            for t in range(start,len(truth)):
                target=truth[t,valid0];good=np.isfinite(target).all(1)
                assert np.isfinite(pred[t]).all();checks+=1
                if good.any():errors.append(np.sqrt(np.square(pred[t,ids[good]]-target[good]).sum(-1)).mean()*1000)
            value=float(np.mean(errors));expected=row['results'][method]['full_suffix']['manual_track_mean_mm']
            assert np.isclose(value,expected,atol=1e-8,rtol=1e-10);checks+=1;recomputed[method].append(value)
    for method,values in recomputed.items():
        assert np.isclose(np.mean(values),scores['summary']['full_suffix'][method]['manual_track_mean_mm']);checks+=1
    # Perturb every hidden object position and manual track, leaving prescribed
    # command trajectories intact. Exported predictor inputs must be identical.
    case='single_lift_sloth';original=predictor_payload(phystwin(case,120));read=pickle.load
    def corrupted(file,*args,**kwargs):
        obj=read(file,*args,**kwargs)
        if str(file.name).endswith('final_data.pkl'):
            obj['object_points'][original['train_end']:]=999.
            obj['object_visibilities'][original['train_end']:]=False
        elif str(file.name).endswith('gt_track_3d.pkl'):obj[:]=1234.
        return obj
    with patch('pickle.load',corrupted):changed=predictor_payload(phystwin(case,120))
    for key,value in original.items():
        if isinstance(value,np.ndarray):assert np.array_equal(value,changed[key],equal_nan=True)
        else:assert value==changed[key]
        checks+=1
    write(report/'phystwin-audit.json',dict(passed=True,checks=checks,
        independent_manual_means_mm={m:float(np.mean(v)) for m,v in recomputed.items()},
        hidden_geometry_perturbation_leaves_predictor_inputs_identical=True,
        scope='Independent metric arithmetic and input contract, not an independent optimizer replication.'))
    print('Audit passed:',checks,'checks')

if __name__=='__main__':main()
