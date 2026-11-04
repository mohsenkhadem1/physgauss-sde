"""Ungated Grounding-DINO + SAM2 preprocessing, explicitly not official SAM3.

Uses a text prompt and frame zero only for initialization, then forward video
propagation. All comparison methods must share these derived observations.
"""
import argparse,contextlib,hashlib,json,subprocess,tempfile
from pathlib import Path
import cv2,numpy as np,torch,imageio_ffmpeg
from PIL import Image
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2_video_predictor
from deform360.processing.masks import process_masks_episode

ROOT=Path(__file__).resolve().parents[2]
CHECKPOINT=ROOT/'runs/real-transfer-v19/checkpoints/sam2.1_hiera_small.pt'

class UngatedPredictor:
    def __init__(self,device):
        self.device=device;self.id='IDEA-Research/grounding-dino-tiny'
        self.processor=AutoProcessor.from_pretrained(self.id)
        self.detector=AutoModelForZeroShotObjectDetection.from_pretrained(self.id).to(device).eval()
        self.sam=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml',str(CHECKPOINT),device=device)
        self.model_id='GroundingDINO-tiny@'+str(self.detector.config._commit_hash)+'+SAM2.1-small@'+hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()+';box=.25;text=.25;first-frame-only'

    def segment(self,video_path,text_prompt,box_xywh=None):
        cap=cv2.VideoCapture(str(video_path));ok,frame=cap.read();cap.release()
        if not ok:raise ValueError('Cannot read initial image')
        image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));prompt=text_prompt.lower().strip().rstrip('.')+'.'
        inputs=self.processor(images=image,text=prompt,return_tensors='pt').to(self.device)
        with torch.inference_mode():outputs=self.detector(**inputs)
        result=self.processor.post_process_grounded_object_detection(outputs,inputs.input_ids,
                        box_threshold=.25,text_threshold=.25,target_sizes=[image.size[::-1]])[0]
        if not len(result['scores']):return None
        index=int(result['scores'].argmax());box=result['boxes'][index].detach().cpu().numpy()
        # Log initialization independently; it must never depend on prediction errors.
        Path(video_path).with_name('sam2-initialization.json').write_text(json.dumps(dict(prompt=prompt,
            box_xyxy=box.tolist(),score=float(result['scores'][index]),model=self.model_id),indent=2)+'\n')
        return self.propagate(Path(video_path),box)

    def propagate(self,video_path,box):
        with tempfile.TemporaryDirectory(prefix='sam2-',dir=ROOT/'runs/real-transfer-v19') as tmp:
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-threads','2','-i',str(video_path),'-q:v','2','-start_number','0',str(Path(tmp)/'%06d.jpg')],check=True)
            # RTX 2080 Ti does not support bfloat16. FP32 is used deliberately.
            with torch.inference_mode():
                state=self.sam.init_state(tmp,offload_video_to_cpu=True,offload_state_to_cpu=True)
                self.sam.add_new_points_or_box(state,frame_idx=0,obj_id=1,box=box)
                for frame,ids,masks in self.sam.propagate_in_video(state):
                    yield frame,(masks[0,0]>0).detach().cpu().numpy()
                self.sam.reset_state(state)
            del state
            torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--object',default='046-sponge');p.add_argument('--episode',type=int,default=0)
    p.add_argument('--device',default='cuda:0');p.add_argument('--cameras',nargs='+');a=p.parse_args()
    torch.set_num_threads(2)
    raw=ROOT/'runs/real-transfer-v19/Deform360/raw'/a.object
    prompt=json.loads((raw/'metadata.json').read_text())['sam_prompt']
    predictor=UngatedPredictor(a.device)
    result=process_masks_episode(ROOT/'runs/real-transfer-v19/Deform360/processed'/a.object,a.episode,
        prompt=prompt,predictor=predictor,cameras=a.cameras,overwrite=False,preview=True)
    print({k:str(v) for k,v in result.items()})
