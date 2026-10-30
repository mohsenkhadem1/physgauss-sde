"""Additive media checks only. Machine pass is never visual approval."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
STUDIES = ('coupled-surface-v13', 'coupled-friction-v13')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=180).stdout


def qa(study):
    report = ROOT/'reports'/study
    pp, vp = report/'protocol.json', report/'video-provenance.json'
    protocol, rows = (json.loads(p.read_text()) for p in (pp, vp))
    expected = {f'{obj}-{scene}.mp4' for obj, scene in protocol['video_cases']}
    if len(expected) != 3 or len(rows) != 3 or {Path(r['path']).name for r in rows} != expected:
        raise RuntimeError('Exactly the three declared video cases required')
    evidence = {str(p.relative_to(ROOT)):sha(p) for p in (pp, vp, Path(__file__))}
    clips = []
    # Kept in /tmp for root visual review; raw trajectories remain untouched.
    review = Path(tempfile.mkdtemp(prefix=study+'-video-qa-'))
    for row in rows:
        rel = Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts or rel.parent != Path('reports')/study:
            raise RuntimeError('Unexpected video location')
        path = ROOT/rel
        if sha(path) != row['sha256']:
            raise RuntimeError('Video provenance digest mismatch')
        if (row['frames'], row['playback_fps'], row['simulation_fps']) != (61, 25, 100):
            raise RuntimeError('Provenance timing mismatch')
        probe = json.loads(run(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(path)]))
        streams = [s for s in probe['streams'] if s['codec_type']=='video']
        if len(streams) != 1:
            raise RuntimeError('Expected one video stream')
        stream = streams[0]
        if int(stream['nb_read_frames']) != 61 or Fraction(stream['avg_frame_rate']) != 25 or Fraction(stream['r_frame_rate']) != 25:
            raise RuntimeError('Decoded count or frame rate mismatch')
        duration = float(stream.get('duration', probe['format']['duration']))
        if abs(duration - 61/25) > .001 or abs(float(probe['format']['duration'])-61/25) > .001:
            raise RuntimeError('Duration mismatch')
        if (stream['width'], stream['height']) != (1600,540):
            raise RuntimeError('Renderer dimensions differ from declared 16x5.4 inches at100 dpi')
        run(['ffmpeg','-v','error','-xerror','-i',str(path),'-map','0:v:0','-f','null','-'])
        folder = review/path.stem
        folder.mkdir()
        run(['ffmpeg','-v','error','-xerror','-i',str(path),'-vf',r'select=eq(n\,0)+eq(n\,30)+eq(n\,60)','-vsync','0',str(folder/'frame-%02d.png')])
        frames = sorted(folder.glob('frame-*.png'))
        if len(frames) != 3:
            raise RuntimeError('Frame extraction incomplete')
        sheet = Image.new('RGB',(1600,3*(540+28)),'white')
        draw = ImageDraw.Draw(sheet)
        for i,(frame,number) in enumerate(zip(frames,(0,30,60))):
            with Image.open(frame) as im:
                if im.size != (1600,540):raise RuntimeError('Extracted dimensions differ')
                sheet.paste(im.convert('RGB'),(0,i*568+28))
            draw.text((8,i*568+6),f'{study} | {path.stem} | frame {number} | physical t={number/100:.2f}s',fill='black')
        target=review/(path.stem+'-sheet.png');sheet.save(target)
        evidence[str(rel)] = sha(path)
        clips.append(dict(path=str(rel),decoded_frames=61,fps=25,duration_seconds=duration,width=1600,height=540,full_decode_passed=True,review_sheet=str(target),review_sheet_sha256=sha(target)))
    if any(sha(ROOT/p)!=digest for p,digest in evidence.items()):
        raise RuntimeError('Media evidence changed during QA')
    result=dict(machine_checks_passed=True,visual_review_passed=False,visual_review_status='Required: inspect sheets and playback; machine checks do not establish fair axes, labels or content.',study=study,evidence_sha256=evidence,clips=clips,ffmpeg_version=run(['ffmpeg','-version']).splitlines()[0],ffprobe_version=run(['ffprobe','-version']).splitlines()[0])
    (report/'video-machine-qa.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',choices=STUDIES,required=True)
    args=parser.parse_args()
    print(json.dumps(qa(args.study),indent=2))
