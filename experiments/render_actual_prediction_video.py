"""Visualize frozen TEST predictions against captured Isaac RGB; never fit."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
import torch

from gaussian_observation import IsaacGaussianRenderer, load_rgb_inputs, project_reference_crops
from render_isaac_dataset_videos import episode_inputs, font, record
from evaluate_gaussian_baseline_predictions import read_prediction


def render(dataset, predictions, appearance, output, device):
    torch.set_num_threads(2)
    size = (192, 160)
    calibration = json.loads(appearance.read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(output), (1152, 688), fps=10,
        codec='libx264', pix_fmt_in='rgb24', macro_block_size=16,
        output_params=['-crf', '18', '-preset', 'fast', '-movflags', '+faststart'])
    writer.send(None)
    cases = []
    # Both are from the already published four-case montage, selected before
    # this comparison. Include an improvement AND a poor prediction.
    ids = ['z0005', 'z0015']
    for case_index, eid in enumerate(ids):
        episode = dataset / 'episodes' / eid
        _, times, images, _, metadata = episode_inputs(dataset, eid)
        inputs = load_rgb_inputs(episode, size, roi=True)
        ref = inputs['reference']
        with np.load(episode / 'truth/cameras.npz') as raw:
            cameras = {k: raw[k] for k in raw.files}
        prediction_path = predictions / f'{eid}-prediction.npz'
        # Frozen predictions are loaded before hidden truth values.
        q = read_prediction(prediction_path, len(ref['rest_points_m']), times)
        prediction_hash = record(prediction_path)
        with np.load(episode / 'truth/trajectory.npz') as raw:
            actual_q = raw['positions_m']
        observed = inputs['cameras']['frame_ids'].tolist()
        crop = project_reference_crops(ref, cameras['camera_to_world_usd_row'], cameras['K'], size)
        background = np.median(inputs['rgb'][0][inputs['mask'][0] < .01], axis=0)
        renderer = IsaacGaussianRenderer(ref, size=size, device=device, calibration_size=size,
            **{k: calibration[k] for k in ['color_matrix', 'color_bias']})
        hidden = (times > 1) & (times < 2)
        rmse = float(np.sqrt(np.mean((q[hidden] - actual_q[hidden])**2)))
        static_rmse = float(np.sqrt(np.mean((ref['rest_points_m'] - actual_q[hidden])**2)))
        print(f'{eid}: frozen prediction loaded; rendering 61 frames', flush=True)
        for frame, t in enumerate(times):
            with torch.no_grad():
                result = renderer(q[[frame]], cameras['camera_to_world_usd_row'][[frame]],
                                  crop['K'][[frame]], background)
            predicted = Image.fromarray(np.rint(np.clip(result['rgb'][0].cpu().numpy(), 0, 1)*255).astype(np.uint8))
            box = tuple(map(int, crop['crop_boxes_xyxy'][frame]))
            actual = Image.open(images[frame]).convert('RGB').crop(box).resize(size, Image.Resampling.BILINEAR)
            canvas = Image.new('RGB', (1152, 688), '#111722')
            draw = ImageDraw.Draw(canvas)
            draw.text((18, 8), f'Actual vs predicted bending | example {case_index+1}/2: {eid}', font=font(26), fill='white')
            draw.text((18, 44), f't = {t:.2f} s | half-speed playback | same orbiting camera and crop | no motion amplification', font=font(18), fill='#c8d1df')
            draw.text((18, 79), 'ACTUAL: captured Isaac simulation', font=font(22), fill='#83dfb4')
            draw.text((594, 79), 'PREDICTED: frozen RGB-D model', font=font(22), fill='#80cbff')
            canvas.paste(actual.resize((576, 480), Image.Resampling.LANCZOS), (0, 112))
            canvas.paste(predicted.resize((576, 480), Image.Resampling.LANCZOS), (576, 112))
            draw.line((576, 112, 576, 592), fill='#111722', width=3)
            label = 'MISSING: these frames were not given to the model' if hidden[frame] else ('INPUT SNAPSHOT' if frame in observed else 'Between input snapshots')
            draw.text((18, 596), label, font=font(18), fill='#ffce75' if hidden[frame] else '#c8d1df')
            draw.text((18, 622), '10 input snapshots before/after 1-2 s; this fills a gap, not a future forecast.', font=font(17), fill='#c8d1df')
            draw.text((18, 647), 'Prediction uses Gaussian rendering: its texture/lighting differs from Isaac.', font=font(16), fill='#c8d1df')
            x0, x1, y = 840, 1118, 613
            draw.line((x0, y, x1, y), fill='#68758a', width=6)
            draw.line((x0+(x1-x0)/3, y, x0+2*(x1-x0)/3, y), fill='#ffce75', width=6)
            for obs in observed:
                x = x0+(x1-x0)*float(times[obs])/3
                draw.line((x, y-7, x, y+7), fill='#80cbff', width=2)
            x = x0+(x1-x0)*float(t)/3
            draw.ellipse((x-4, y-4, x+4, y+4), fill='white')
            for tick in range(4):
                draw.text((x0+(x1-x0)*tick/3-4, 626), f'{tick}s', font=font(13), fill='white')
            writer.send(np.asarray(canvas))
            if frame == 30:
                canvas.save(output.with_name(output.stem+f'.{eid}.poster.png'))
            if frame % 20 == 0:
                print(f'{eid}: frame {frame}/60', flush=True)
        if record(prediction_path) != prediction_hash:
            raise RuntimeError('Frozen prediction changed')
        cases.append({**metadata, 'prediction': prediction_hash,
            'observed_frames': observed, 'crop_boxes_xyxy': crop['crop_boxes_xyxy'].tolist(),
            'hidden_coordinate_rmse_m': rmse, 'static_hidden_coordinate_rmse_m': static_rmse})
        del renderer
    writer.close()
    manifest = {'kind': 'Actual captured Isaac RGB versus frozen TEST prediction',
        'selection': 'Two visibly moving cases from the previously published four-case montage; includes failure',
        'model': '25D RGB-D full-context posterior mean; frozen, no new fitting',
        'task': 'Missing-interval interpolation, not causal forecasting',
        'physical_interval_per_case_s': [0, 3], 'sample_fps': 20, 'playback_fps': 10,
        'frames': 122, 'encoded_duration_s': 12.2,
        'frame_interpolation': False, 'displacement_amplification': False,
        'actual_panel': 'Saved Isaac RGB, known-rest per-camera crop, resized to192x160 then3x',
        'prediction_panel': 'Saved mean nodal positions, existing Gaussian renderer, same camera/crop',
        'qualification': 'Gaussian and Isaac appearance differ. Geometry is not retouched.',
        'appearance_calibration': record(appearance), 'source_script': record(Path(__file__)),
        'renderer': record(Path(__file__).with_name('gaussian_observation.py')),
        'cases': cases, 'video': record(output)}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest['video']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('runs/gaussian-boundary-neural-v01'))
    parser.add_argument('--predictions', type=Path, default=Path('runs/gaussian-final-test-fresh64-25d-rgbd-full-v01'))
    parser.add_argument('--appearance', type=Path, default=Path('runs/gaussian-observation-roi/train-appearance.json'))
    parser.add_argument('--output', type=Path, default=Path('examples/gaussian-neural-boundary/actual-vs-predicted.mp4'))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    render(args.dataset, args.predictions, args.appearance, args.output, args.device)
