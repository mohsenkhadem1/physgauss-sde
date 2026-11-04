from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/video-scene-v24';REPORT=ROOT/'reports/video-scene-v24'
DATA=ROOT/'runs/editable-scene-v23/phystwin-full-rgb/data/different_types'

def partition(case):
    if 'rope'in case or case=='weird_package':return 'test'
    if 'cloth_3'in case or case=='single_lift_dinosor':return 'validation'
    return 'train'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    cases=sorted(p.name for p in DATA.iterdir()if (p/'final_data.pkl').exists())
    plan=dict(cases={g:[c for c in cases if partition(c)==g]for g in ('train','validation','test')},
        seed=2401,steps=2500,batch=3,queries=128,learning_rate=.0003,checkpoint='Fixed final step; no test-based selection',
        source_camera=0,input_cameras=[0,1],held_rgb_camera=2,
        input='RGB at source frame0 and target frame t from cameras0/1; known calibration; source pixel queries. No initial mesh/depth/target tracks in model input.',
        task='Observed-video 3D reconstruction; target-time images ARE inputs. This is not causal forecasting or novel-action prediction.',
        supervision='Author reconstructed 3D tracks, source/target visibility; pseudo labels may use all three capture cameras. Camera2 RGB held out does not make 3D supervision independent of camera2.',
        train_times='Official training prefix only; known-case suffix is temporal reconstruction transfer.',
        object_split='All ropes and package held test; dinosaur and cloth_3 variants validation. Grouping by author names; physical identity not independently verified. Existing sequences historically exposed.',
        query_selection='Frame0 camera0 projection, frontmost point per8pixel bin; label-based query selection disclosed, not automatic segmentation.',
        methods=['visual','image_blind'],baseline='Same network/optimizer/labels, image features zeroed in both training and evaluation; privileged true-initial persistence also reported.',
        encoder='Frozen torchvision ResNet18 ImageNet1K V1 layer3 features, resized384, pooled12x12; shared trainable query transformer.',
        raw_data='Local only; test data never used for optimizer or checkpoint choice.')
    if (REPORT/'protocol.json').exists():assert json.loads((REPORT/'protocol.json').read_text())==plan
    else:write(REPORT/'protocol.json',plan)
    print(plan['cases'])

if __name__=='__main__':main()
