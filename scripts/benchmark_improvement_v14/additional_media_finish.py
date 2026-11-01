"""Bounded reporting waiter for predeclared B/PhysTwin video sets."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'reports/benchmark-improvement-v14'
BIN=ROOT/'runs/media-tooling/ffmpeg-7.0.2-amd64-static'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def remaining():return (datetime.datetime.fromisoformat('2026-10-31T18:36:34+00:00')-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
def command(args):
    if remaining()<10:raise TimeoutError('Campaign deadline reached')
    return subprocess.run(args,check=True,capture_output=True,text=True,timeout=min(remaining()-5,600)).stdout

def qa(folder):
    path=folder/'provenance.json';provenance=json.loads(path.read_text());records=[]
    for row in provenance['cases']:
        p=ROOT/row['path'];assert sha(p)==row['sha256']
        stream=json.loads(command([str(BIN/'ffprobe'),'-v','error','-count_frames','-show_streams','-of','json',str(p)]))['streams'][0]
        count=row['frames'];fps=row['fps'];assert int(stream['nb_read_frames'])==count;assert stream['avg_frame_rate']==f'{fps}/1';assert abs(float(stream['duration'])-count/fps)<.002
        expected_width=500*(len(row['prediction_sha256'])+1)
        # Multiple panels can share one prediction archive; titles give panel count.
        expected_width=500*len(row['unavailable_frames_by_panel'])
        assert (stream['width'],stream['height'])==(expected_width,500)
        command([str(BIN/'ffmpeg'),'-v','error','-xerror','-i',str(p),'-f','null','-'])
        sheet=Image.new('RGB',(expected_width,1575),'white');draw=ImageDraw.Draw(sheet)
        for j,n in enumerate([0,count//2,count-1]):
            fp=Path('/tmp')/(folder.name+'-'+p.stem+f'-{n}.png');command([str(BIN/'ffmpeg'),'-v','error','-i',str(p),'-vf',f'select=eq(n\\,{n})','-frames:v','1','-y',str(fp)])
            with Image.open(fp)as image:sheet.paste(image,(0,j*525+25))
            draw.text((10,j*525+5),f'{p.stem} | suffix frame{n}/{count} | originalframe{row["frame_interval"][0]+n}',fill='black')
        target=folder/(p.stem+'-contactsheet.png');sheet.save(target)
        records.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p),frames=count,fps=fps,duration=float(stream['duration']),dimensions=[expected_width,500],full_decode_pass=True,unavailable_frames_by_panel=row['unavailable_frames_by_panel'],review_sheet=str(target.relative_to(ROOT)),sheet_sha256=sha(target)))
    result=dict(passed=True,visual_review_passed=False,visual_review_required=True,provenance_sha256=sha(path),helper_sha256=sha(Path(__file__)),clips=records)
    (folder/'machine-qa.json').write_text(json.dumps(result,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=['phystwin','robust']);args=parser.parse_args();kind=args.kind
    manifest_path=R/('phystwin-video-manifest.json'if kind=='phystwin'else'softvt-robust-video-manifest.json');manifest=json.loads(manifest_path.read_text());pin=sha(manifest_path)
    extra=['phystwin-scores.json','phystwin-gaussian-scores.json']if kind=='phystwin'else['softvt-scores-b.json','softvt-summary-b.json']
    required=[ROOT/p for p in manifest['required_before_target_access']]+[R/p for p in extra]
    if kind=='phystwin':
        for p in list(required):
            if p.name=='complete.json':required.append(p.with_name('scores.json'))
    while not all(p.is_file()for p in required):
        if remaining()<120:raise TimeoutError('Media gates incomplete before deadline reserve')
        time.sleep(min(20,remaining()-100))
    if sha(manifest_path)!=pin:raise RuntimeError('Declared manifest changed while waiting')
    gate_hash={str(p.relative_to(ROOT)):sha(p)for p in required}
    (R/(kind+'-media-gate.json')).write_text(json.dumps(dict(authorized_by_root=True,deadline='2026-10-31T18:36:34Z',evidence_sha256=gate_hash,manifest_sha256=pin),indent=2)+'\n')
    os.environ['PATH']=str(BIN)+os.pathsep+os.environ['PATH']
    module='scripts.benchmark_improvement_v14.'+('phystwin_video'if kind=='phystwin'else'softvt_robust_video')
    command([sys.executable,'-m',module,'--authorized-outcome-access'])
    qa(ROOT/manifest['output_directory'])
    if any(sha(ROOT/p)!=h for p,h in gate_hash.items()):raise RuntimeError('Gate evidence changed during rendering')
    print(kind,'render and machine QA complete; root visual review required',flush=True)
if __name__=='__main__':main()
