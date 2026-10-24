"""Local, orbiting-camera video of held-force static predictions.

Reads only a qualified, checksum-bound prefix posterior after its action seal.
Camera motion is animation; the displayed equilibria are static. Media/data
stay local. This script is separate from the frozen scientific study.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
from experiments import run_surface_p2_study_v02 as runner
from experiments import campaign_surface_p2_static_v02 as physics
from experiments.campaign_surface_posterior_field_v01 import PosteriorMeanField


def main(args):
    protocol = runner.validate(args.protocol_sha256)
    if protocol['numerical_engine'] != 'conditional_amplitude':
        raise ValueError('Explicit positive conditional-amplitude hypotheses required')
    directory = runner.LOCAL / f'object-{args.object:02d}-LOCAL'
    posterior = directory / f'{args.law}-prefix-LOCAL.json'
    seal = json.loads((directory / f'{args.law}-choices-seal-LOCAL.json').read_text())
    record = json.loads(posterior.read_text())
    if not record['qualified'] or seal['protocol_SHA256'] != args.protocol_sha256:
        raise ValueError('Qualified prefix and matching frozen action seal required')
    if seal['post_probe_outcomes_read'] or seal['truth_used_for_selection'] or seal['choices'] is None:
        raise ValueError('A causal, prefix-only action choice seal is required')
    if seal['prefix_checkpoint_SHA256'] != runner.sha(posterior):
        raise ValueError('Prefix seal changed')
    if record['raw_SHA256'] != runner.sha(posterior.with_suffix('.npz')):
        raise ValueError('Posterior raw checksum mismatch')
    with np.load(posterior.with_suffix('.npz')) as z:
        theta = runner.physical_parameters(z['unit_grid']); probability = z['quadrature_probability']
        sensor_mean = z['mean']
    truthmeta = json.loads((directory / 'truth-checkpoint-LOCAL.json').read_text())
    if not truthmeta['all_physical_valid'] or runner.sha(directory/'truth-LOCAL.npz') != truthmeta['raw_SHA256']:
        raise ValueError('Qualified checksum-bound truth required')
    ready = json.loads((runner.LOCAL/'cache-ready.json').read_text())
    if not ready['qualified'] or ready['protocol_SHA256'] != args.protocol_sha256 or record['model_fingerprint'] != ready['model_fingerprint']:
        raise ValueError('Posterior must match the frozen model reference')
    with np.load(directory/'truth-LOCAL.npz') as z:
        prefix = z[args.law][truthmeta['labels'].index(runner.study.PREFIX)] + z['prefix_noise']
    if record['observations_SHA256'] != runner.observation_fingerprint([(runner.study.PREFIX,prefix)]):
        raise ValueError('Posterior must match the acquired initial observation')
    labels = runner.study.TARGETS; fwd = physics.SparseStaticForward(4)
    catalog = runner.study.load_catalog(4)
    force = np.column_stack([catalog[label].ravel()[fwd.free] for label in labels])
    model = PosteriorMeanField(fwd.Kunit_free, fwd.W_free, force)
    mean = np.zeros((len(labels), *fwd.X.shape))
    mean.reshape(len(labels), -1)[:, fwd.free] = model.mean(theta, probability)
    query = physics.field_queries(4, mean, runner.study.observation_points()).reshape(-1)
    parity = float(np.max(abs(query-sensor_mean)))
    if parity > 2e-9:
        raise FloatingPointError('Full-field versus recorded posterior sensor-mean parity failed')
    actual = []
    stream = runner.private_streams(args.object)
    context = dict(protocol_SHA256=args.protocol_sha256, index=args.object,
        unit_SHA256=runner.array_sha(stream['unit']),
        force_SHA256={label:runner.array_sha(f) for label,f in catalog.items()})
    if args.law == 'matching_linear':
        path = directory / 'linear-fields-LOCAL.npz'; runner.verify_npz(path, context)
        with np.load(path) as z:
            actual = z['fields'][[truthmeta['labels'].index(label) for label in labels]]
    else:
        for label in labels:
            path = directory/f'truth-load-{label}-LOCAL.npz'
            runner.verify_npz(path, dict(context, label=label))
            with np.load(path) as z:actual.append(z['NH80_field'])
        actual = np.asarray(actual)
    actual_query = physics.field_queries(4, actual, runner.study.observation_points()).reshape(len(labels), -1)
    errors = np.sqrt(np.mean((query.reshape(len(labels), -1)-actual_query)**2, axis=1))*1e6

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.animation import FFMpegWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    # Face node order: corners0/1/2, edge01/02/12. Four display triangles.
    triangles = fwd.faces[:, np.array([[0,3,4],[3,1,5],[4,5,2],[3,5,4]])].reshape(-1,3)
    magnitude = np.linalg.norm(np.stack([actual, mean]), axis=-1)*1000
    color_norm = Normalize(0, float(magnitude.max())); cmap = plt.get_cmap('viridis')
    positions = (fwd.X[None,None]+args.scale*np.stack([actual,mean]))*1000
    low = positions.min(axis=(0,1,2))-15; high = positions.max(axis=(0,1,2))+15
    fig = plt.figure(figsize=(12,5.5)); axes = [fig.add_subplot(1,2,j+1,projection='3d') for j in range(2)]
    meshes = []
    for j,ax in enumerate(axes):
        mesh = Poly3DCollection(positions[j,0][triangles], linewidths=.12, edgecolors=(0,0,0,.18))
        ax.add_collection3d(mesh); meshes.append(mesh)
        ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]),xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)')
        ax.set_box_aspect(high-low)
        ax.set_title(('Actual '+args.law+' simulation','Predicted posterior mean')[j],fontsize=12)
    fig.subplots_adjust(left=.02,right=.88,bottom=.18,top=.83,wspace=.03)
    cax=fig.add_axes([.91,.27,.018,.40]);fig.colorbar(ScalarMappable(norm=color_norm,cmap=cmap),cax=cax,label='Displacement magnitude (mm; physical units)')
    heading=fig.text(.5,.94,'',ha='center',fontsize=14)
    fig.text(.5,.055,f'Static equilibria | displacement displayed ×{args.scale:g} | camera motion only; no time forecast',ha='center',fontsize=10)
    fig.text(.5,.015,'Held-force prediction from the initial observation only; object index fixed before seeing results.',ha='center',fontsize=9)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    writer=FFMpegWriter(fps=24,codec='libx264',extra_args=['-pix_fmt','yuv420p'])
    with writer.saving(fig,str(output),dpi=100):
        for k,label in enumerate(labels):
            heading.set_text(f'Object {args.object} · {label} · held-out sensor RMSE {errors[k]:.1f} µm')
            for j,mesh in enumerate(meshes):
                mesh.set_verts(positions[j,k][triangles]);mesh.set_facecolor(cmap(color_norm(magnitude[j,k][triangles].mean(axis=1))))
            for frame in range(48):
                for ax in axes:ax.view_init(elev=20+8*np.sin(frame*np.pi/47),azim=-65+110*frame/47)
                writer.grab_frame()
    plt.close(fig)
    receipt=dict(protocol_SHA256=args.protocol_sha256,object_index=args.object,law=args.law,
        posterior_checkpoint_SHA256=runner.sha(posterior),video_SHA256=runner.sha(output),
        full_field_sensor_parity_m=parity,held_force_sensor_RMSE_um=errors.tolist(),
        displacement_display_scale=args.scale,frames=192,static_equilibria=True,camera_motion_only=True,
        video_LOCAL=True,not_a_time_forecast=True)
    output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),sensor_parity_m=parity,frames=192)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol-sha256',required=True)
    p.add_argument('--object',type=int,choices=[0],default=0);p.add_argument('--law',choices=['NH80_stress'],default='NH80_stress')
    p.add_argument('--scale',type=float,default=20.)
    p.add_argument('--output',default='runs/operator-dev/surface-static-media-v01/held-force-actual-predicted.mp4')
    args=p.parse_args()
    if args.object not in range(16) or not np.isfinite(args.scale) or args.scale<=0:raise ValueError('Declared object and positive display scale required')
    main(args)
