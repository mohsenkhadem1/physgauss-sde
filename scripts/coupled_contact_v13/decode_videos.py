"""Independent full video decode and fixed first/middle/last contact sheets."""
from pathlib import Path
import argparse,hashlib,json
import imageio.v2 as imageio
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',default='reports/coupled-contact-v13');args=parser.parse_args()
    report=ROOT/args.report;rows=[]
    for entry in json.loads((report/'video-provenance.json').read_text()):
        path=ROOT/entry['path'];reader=imageio.get_reader(path,format='ffmpeg');meta=reader.get_meta_data();frames=[]
        for frame in reader:frames.append(frame)
        reader.close()
        if len(frames)!=entry['frames'] or meta['fps']!=entry['playback_fps']:raise RuntimeError('Video count/fps differs: '+str(path))
        selected=[0,len(frames)//2,len(frames)-1];width=1600;height=int(frames[0].shape[0]*width/frames[0].shape[1])
        canvas=Image.new('RGB',(width,3*(height+26)),'white');draw=ImageDraw.Draw(canvas)
        for i,k in enumerate(selected):
            draw.text((10,i*(height+26)+5),f"Frame {k}, physical time {k/entry['simulation_fps']:.2f} s",fill='black')
            canvas.paste(Image.fromarray(frames[k]).resize((width,height)),(0,i*(height+26)+26))
        sheet=report/(path.stem+'-independent-contactsheet.png');canvas.save(sheet)
        rows.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),decoded_frames=len(frames),fps=meta['fps'],size=meta['size'],contactsheet=str(sheet.relative_to(ROOT)),selected_frames=selected))
    (report/'independent-video-decode.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
