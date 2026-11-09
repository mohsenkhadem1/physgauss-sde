"""Camera-0 renders with ONE shared appearance model: GIC's own fixed-particle Gaussians (trained on
cameras 1-10 of its fitting episode), driven by different motion sources. Differences in PSNR then
isolate motion error. Sources: true motion (appearance ceiling), GIC's own forecast from that model,
the oracle-parameter GIC forecast (D0 model only), and our posterior-mean fields (tracks / RGB-D,
passive-only and after the poke). Field sources are sampled at each Gaussian's rest position
(clamped into the true rest box, since GIC's reconstruction is slightly larger than the body).
Usage (cwd = GIC vendor dir): gic_shared_appearance.py MODEL_NAME WORLD [WORLD...]"""
import sys, json
from pathlib import Path
from argparse import Namespace, ArgumentParser
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gic_stem_render as GR
import numpy as np, torch
import fem
from gaussian_renderer import render
from train_gs_fixed_pcd import train_gs_with_fixed_pcd
from new_trajectory import load_pcd_file
from arguments import OptimizationParams

T = R/'runs/stem-trial-v01'


def main():
    name, worlds = sys.argv[1], sys.argv[2:]
    model = T/f'gic/{name}'
    cfg = eval((model/'cfg_args').read_text(), {'Namespace': Namespace})
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.tensor([0., 0, 0], device='cuda'); phys = json.loads((model/'0-pred.json').read_text())
    vol = load_pcd_file(str(model), 40000)
    parser = ArgumentParser(); op = OptimizationParams(parser); opt = op.extract(parser.parse_args(['--reg_scale', '--reg_alpha']))
    assert (model/'stem-fixpcd-trained.json').exists()
    xyz = torch.tensor(np.load(T/f'gic/fit-{name}.npz')['xyz'], device='cuda')
    scene = train_gs_with_fixed_pcd(vol, cfg, opt, pipe, [], [opt.iterations], [x-vol for x in xyz], phys['fps'],
                                    force_train=False, grid_size=phys['density_grid_size'])
    mesh = fem.Mesh((33, 6, 6))
    ref = vol.cpu().numpy().astype(float)
    eps = 1e-6
    clamped = np.clip(ref, [fem.X0+eps, fem.Y0+eps, fem.Z0+eps], [fem.X1-eps, fem.Y1-eps, fem.Z1-eps])
    I = mesh.interpolation(clamped).toarray()
    out = T/'renders/shared'; out.mkdir(parents=True, exist_ok=True)
    def gic_fc(ep, w):
        if name == 'D0':
            return T/('gic/forecast-D0.npz' if ep == 'D2' else 'gic/forecast-D2b-D0.npz')
        return T/(f'gic/forecast-{name}.npz' if ep == 'D2' else f'gic/forecast-D2b-{name}.npz')
    for w in worlds:
        for ep in ('D2', 'D2b'):
            src = {'truth motion': I@np.load(T/f'truth/{ep}-{w}.npz')['U'].transpose(1, 0, 2).reshape(mesh.n, -1)}
            src['truth motion'] = src['truth motion'].reshape(len(ref), -1, 3).transpose(1, 0, 2)
            p = np.load(gic_fc(ep, w))['xyz']
            src['GIC forecast'] = p-p[0]
            if name == 'D0':
                o = np.load(T/(f'gic/forecast-D2b-oracle-{w}.npz' if ep == 'D2b' else f'gic/forecast-oracle-{w}.npz'))['xyz']
                src['GIC oracle parameters'] = o-o[0]
            for s in ('tracks', 'rgbd'):
                z = np.load(T/f'ours-fields/{s}-{w}-{ep}.npz')
                for stage in ('D0', 'D0pD1'):
                    U = z[stage]
                    src[f'ours {s} {stage}'] = np.einsum('pn,fnj->fpj', I, U)
            views = GR.cam0(T/f'scenes/{ep}-{w}', cfg)
            with torch.no_grad():
                for label, d in src.items():
                    d = torch.tensor(d, device='cuda', dtype=torch.float32); assert len(d) == len(views)
                    imgs = []
                    for f, view in enumerate(views):
                        view.load2device()
                        imgs.append(GR.to_u8(render(view, scene.gaussians, pipe, bg, d[f], 0., 0., False)['render'])); view.load2device('cpu')
                    np.savez_compressed(out/f'{name}|{ep}|{w}|{label}.npz', images=np.stack(imgs))
            print(name, w, ep, list(src), flush=True)


if __name__ == '__main__':
    main()
