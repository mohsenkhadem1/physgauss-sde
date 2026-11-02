"""All eight fixed-camera cohort videos, only from scored and sealed images."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.compositional_dynamics_v17_reporting import cohort_appearance_inference as inference
from scripts.compositional_dynamics_v17_reporting.matched_video import projection

ROOT = inference.ROOT
OUTPUT = ROOT / 'reports/compositional-dynamics-v17/fresh-cohort-videos'
CAMERA = 'held-0'
FPS = 10


def panel(path, width, height, error=None):
    if path is None:
        result = Image.new('RGB', (width, height), (240, 240, 240))
        draw = ImageDraw.Draw(result)
        draw.text((24, height // 2 - 24), 'Prediction unavailable' if error else 'Reference unavailable',
                  fill=(150, 35, 35), font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 28))
        draw.text((24, height // 2 + 18), 'Failed attempt retained in evaluation', fill='black')
        return result
    with np.load(path) as f:
        rgb = f['rgb']
    if rgb.shape != (height, width, 3) or not np.isfinite(rgb).all():
        raise ValueError('Image viewport or finite-value mismatch')
    return Image.fromarray(np.uint8(np.round(np.clip(rgb, 0, 1) * 255)))


def build():
    renderer = inference.bound()
    plan, commit = renderer.prediction_gate()
    results_path = ROOT / 'reports/compositional-dynamics-v17/fresh-cohort-appearance-results.json'
    results = renderer.read(results_path)
    if results['plan_sha256'] != renderer.sha(renderer.PLAN) or results['prediction_commit_sha256'] != renderer.sha(renderer.BASE / 'prediction-commit.json'):
        raise ValueError('Completed image score commitment required')
    if len(results['rows']) != 2480 or results['expected_rows'] != 2480:
        raise ValueError('All image slots must be retained')
    for path, digest in results['evidence_sha256'].items():
        if renderer.sha(ROOT / path) != digest:
            raise ValueError('Scored evidence changed: ' + path)
    rows = {(r['world_id'], r['query'], r['method'], r['frame'], r['camera']): r for r in results['rows']}
    sealed = {(r['world_id'], r['query'], r['method'], r['frame'], r['camera']): r for r in commit['rows']}
    if len(rows) != 2480 or set(rows) != set(sealed):
        raise ValueError('Image score and prediction slot census mismatch')
    for key, row in rows.items():
        if any(row.get(field) != sealed[key].get(field) for field in ('completed', 'time_s', 'role', 'prediction')):
            raise ValueError('Scored row differs from sealed prediction')
        world, role, method, frame, camera = key
        other = rows[world, role, 'native' if method == 'shared_q2' else 'shared_q2', frame, camera]
        if row['reference_complete'] != other['reference_complete'] or row.get('reference_sha256') != other.get('reference_sha256'):
            raise ValueError('Methods must share the exact reference image')
        if row['reference_complete']:
            path = renderer.BASE / 'references' / world / role / f'{frame:03d}-{camera}.npz'
            digest = results['evidence_sha256'].get(str(path.relative_to(ROOT)))
            if digest is None or row['reference_sha256'] != digest:
                raise ValueError('Reference image is not bound by scored evidence')
    if OUTPUT.exists():
        raise ValueError('Preserve previous video artifacts')
    OUTPUT.mkdir()
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    import imageio_ffmpeg
    records = []
    for query in plan['queries']:
        world, role = query['world_id'], query['role']
        camera = next(c for c in query['cameras'] if c['name'] == CAMERA)
        width, height = camera['width'], camera['height']
        header, footer = 54, 60
        times = query['times_s']
        if len(times) != 31 or not np.allclose(times, np.arange(31) * .01, atol=1e-12, rtol=0):
            raise ValueError('Fixed 31-frame physical clock required')
        floor = -.03 - {'forecast-40mm': .04, 'forecast-80mm': .08}[role]
        corners = np.array([[-.03, -.07, floor], [.15, -.07, floor], [.15, .07, floor], [-.03, .07, floor], [-.03, -.07, floor]])
        line = projection(corners, camera)
        video = OUTPUT / f'{world}-{role}-held0.mp4'
        preview_frames, evidence, failures = [], {}, {}
        with tempfile.TemporaryDirectory(prefix='v17-cohort-video-') as tmp:
            for i, t in enumerate(times):
                ours = rows[world, role, 'shared_q2', i, CAMERA]
                native = rows[world, role, 'native', i, CAMERA]
                actual = renderer.BASE / 'references' / world / role / f'{i:03d}-{CAMERA}.npz' if ours['reference_complete'] else None
                paths = [actual] + [ROOT / r['prediction']['path'] if r['completed'] else None for r in (ours, native)]
                canvas = Image.new('RGB', (3 * width, header + height + footer), 'white')
                draw = ImageDraw.Draw(canvas)
                for j, (title, path) in enumerate(zip(['Actual simulation', 'Our prediction', 'Native PhysTwin prediction'], paths)):
                    canvas.paste(panel(path, width, height, error=j > 0), (j * width, header))
                    draw.text((j * width + 12, 8), title, font=font, fill='black')
                    if path is not None:
                        evidence[str(path.relative_to(ROOT))] = renderer.sha(path)
                        draw.line([(float(x) + j * width, float(y) + header) for x, y in line], fill=(125, 125, 125), width=2)
                    else:
                        failures[title] = (ours if j == 1 else native).get('error') if j else ours.get('reference_error')
                draw.text((12, header + height + 5), f'{world} | {role} | physical time {t:.2f} s | 10x slow motion | fixed held camera', font=font, fill='black')
                draw.text((12, header + height + 30), 'Same initial Gaussian appearance. No motion magnification. Gray floor outline is a display guide.', font=font, fill='black')
                canvas.save(Path(tmp) / f'{i:03d}.png')
                if i in (0, 10, 20, 30):
                    preview_frames.append(canvas.resize((1272, 297)))
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-framerate', str(FPS), '-i', str(Path(tmp) / '%03d.png'), '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', str(video)], check=True)
        frames = imageio_ffmpeg.read_frames(str(video))
        metadata = next(frames)
        decoded = sum(1 for _ in frames)
        if decoded != 31 or metadata['fps'] != FPS or tuple(metadata['size']) != (3 * width, header + height + footer):
            raise ValueError('Video decode/clock/viewport failed')
        sheet = Image.new('RGB', (1272, 4 * 297), 'white')
        for i, frame in enumerate(preview_frames):
            sheet.paste(frame, (0, i * 297))
        preview = video.with_suffix('.png')
        sheet.save(preview)
        records.append(dict(world_id=world, query=role, video=str(video.relative_to(ROOT)), video_sha256=renderer.sha(video), preview=str(preview.relative_to(ROOT)), preview_sha256=renderer.sha(preview), decoded_frames=decoded, fps=FPS, physical_duration_s=.30, failures=failures, evidence_sha256=evidence))
    renderer.write(OUTPUT / 'manifest.json', dict(selection='All four worlds and both declared loads; fixed held-0 camera, no metric-based selection.', score_sha256=renderer.sha(results_path), source_sha256=renderer.sha(__file__), records=records, visual_review='pending', motion_magnification=1, image_interpolation=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    if not parser.parse_args().execute:
        raise SystemExit('Prepared only; execute after committed image scoring.')
    build()
