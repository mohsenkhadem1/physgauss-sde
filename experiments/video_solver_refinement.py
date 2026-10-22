"""Compare two recorded Isaac solver settings at true geometric scale."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from experiments.video_block_gravity_suite import load, volume_weights, draw_panel


def render(left, right, output, repeat=12):
    a, ma, pa = load(left)
    b, mb, pb = load(right)
    for key in ['rest_m', 'tets', 'faces', 'fixed_node_ids']:
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_allclose(a['times_s'], b['times_s'], atol=1e-12, rtol=0)
    if len(a['positions_m']) != len(b['positions_m']):
        raise ValueError('Same recorded sample times required')
    for key in ['young_Pa', 'kind', 'boundary', 'duration_s']:
        if ma['case'].get(key) != mb['case'].get(key):
            raise ValueError('Different physical case: '+key)
    for key in ['material', 'gravity_readback_m_s2', 'density_authored_kg_m3']:
        if ma['native_verification'][key] != mb['native_verification'][key]:
            raise ValueError('Different native physical setting: '+key)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    size = (1280, 620)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    w = volume_weights(a['rest_m'], a['tets'])
    target = w @ a['rest_m']
    writer = imageio_ffmpeg.write_frames(str(out/'solver-comparison.mp4'), size,
        fps=30, codec='libx264', output_params=['-crf', '19', '-threads', '2', '-movflags', '+faststart'], macro_block_size=2)
    writer.send(None)
    count = 0
    max_delta = 0.
    try:
        for i, t in enumerate(a['times_s']):
            delta = float(np.linalg.norm(a['positions_m'][i].astype(float)-b['positions_m'][i].astype(float), axis=1).max())
            max_delta = max(max_delta, delta)
            for j in range(repeat):
                phase = (i*repeat+j)/max(len(a['times_s'])*repeat-1, 1)
                canvas = Image.new('RGB', size, '#f6f8fa')
                draw = ImageDraw.Draw(canvas)
                draw.text((18, 10), 'Same physical block, supports and gravity; different numerical settings', font=font, fill='#142b3b')
                for k, (d, m, box) in enumerate([(a, ma, (10, 70, 630, 535)), (b, mb, (650, 70, 1270, 535))]):
                    c=m['case']; hz=round(1/m['clock']['float32_dt_s'])
                    draw.text((18+640*k, 40), f"{hz} Hz | {c['iterations']} iterations per step", font=font, fill='#142b3b')
                    draw_panel(canvas, d['positions_m'][i], d['rest_m'], d['faces'], d['fixed_node_ids'].astype(int), target, phase, box)
                draw.text((18, 552), f't = {float(t):.3f} s | maximum corresponding-node difference = {delta*1e3:.3f} mm', font=font, fill='#142b3b')
                draw.text((18, 582), f'True scale | {repeat}x slow playback | same orbit camera | recorded states held; no interpolation or prediction', font=small, fill='#344c5a')
                writer.send(np.asarray(canvas).tobytes())
                if i == len(a['times_s'])-1 and j == 0:
                    canvas.save(out/'preview.png')
                count += 1
    finally:
        writer.close()
    reader = imageio_ffmpeg.read_frames(str(out/'solver-comparison.mp4'))
    next(reader)
    decoded = sum(1 for _ in reader)
    if decoded != count:
        raise RuntimeError('Video frame count mismatch')
    report = dict(schema='recorded-Isaac-solver-comparison-video-v0.1',
        sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [pa,pb]],
        frames=count, decoded_frames=decoded, fps=30, repeat=repeat, geometric_scale=1,
        camera_orbit_degrees=45, identical_camera=True, is_prediction=False,
        interpolated_simulation_states=False, maximum_corresponding_node_difference_m=max_delta,
        video_sha256=hashlib.sha256((out/'solver-comparison.mp4').read_bytes()).hexdigest())
    (out/'video.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left', required=True)
    p.add_argument('--right', required=True)
    p.add_argument('--output', required=True)
    a=p.parse_args()
    print(json.dumps(render(a.left,a.right,a.output)))
