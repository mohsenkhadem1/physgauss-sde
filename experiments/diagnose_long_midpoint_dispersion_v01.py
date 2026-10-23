"""Read saved LOCAL fields; fit a descriptive oscillator, never call FE/CUDA.

The fitted frequency is diagnostic, not a certified prestressed FEM eigenvalue.
No old outputs are changed. The exact modal reference is an independent Duhamel
integral; recurrence is the implicit-midpoint Cayley transform.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares


def midpoint_clocks(omega, steps):
    h = 1 / steps
    A = np.array([[0., 1.], [-omega**2, 0.]])
    left = np.eye(2) - h*A/2
    advance = np.linalg.solve(left, np.eye(2) + h*A/2)
    load = np.linalg.solve(left, [0., h])
    state = np.zeros(2)
    saved = [state.copy()]
    for k in range(steps):
        t = (k + .5)*h
        state = advance@state + load*(256*t**4*(1-t)**4)
        if (k + 1) % (steps//8) == 0:
            saved.append(state.copy())
    return np.array(saved)


def duhamel_clocks(omega):
    def pulse(s):
        return 256*s**4*(1-s)**4
    return np.array([[quad(lambda s: np.sin(omega*(t-s))/omega*pulse(s), 0, t,
                          epsabs=1e-14, epsrel=1e-12)[0],
                      quad(lambda s: np.cos(omega*(t-s))*pulse(s), 0, t,
                           epsabs=1e-14, epsrel=1e-12)[0]]
                     for t in np.arange(9)/8])


def diagnose(root):
    root = Path(root)
    saved = []
    pins = {}
    for n in (32, 64, 128):
        fields = []
        for k in range(9):
            path = root/f'fine40_h{n}'/f'clock-{k:03d}.npz'
            pins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path, allow_pickle=False) as f:
                fields.append({key: f[key].copy() for key in ('u', 'v')})
        saved.append(fields)
    initial_identity = all(np.array_equal(saved[0][0][k], a[0][k])
                           for a in saved[1:] for k in ('u', 'v'))
    basis = saved[1][8]['u'] - saved[2][8]['u']
    basis /= np.linalg.norm(basis, axis=1).max()
    B = np.sum(basis*basis)
    projected = np.empty((2, 9, 2))
    rows = []
    residual = np.zeros(2)
    total = np.zeros(2)
    for i in range(9):
        row = dict(time_s=i/8)
        for ik, key in enumerate(('u', 'v')):
            d = [saved[j][i][key] - saved[j+1][i][key] for j in (0, 1)]
            norms = [np.linalg.norm(x) for x in d]
            for j in (0, 1):
                projected[j, i, ik] = np.sum(d[j]*basis)/B
                residual[ik] += np.sum((d[j] - projected[j, i, ik]*basis)**2)
                total[ik] += np.sum(d[j]**2)
            row[key] = dict(pair_maxima=[float(np.linalg.norm(x, axis=1).max()) for x in d],
                            signed_cosine=float(np.sum(d[0]*d[1])/np.prod(norms)) if np.prod(norms) else None)
        rows.append(row)

    def differences(omega):
        return -np.diff(np.array([midpoint_clocks(omega, n) for n in (32, 64, 128)]), axis=0)
    scale = np.array([1e-6, 1e-4])
    def objective(z):
        return ((z[1]*differences(z[0]) - projected)/scale).ravel()
    candidates = []
    for omega0 in np.linspace(10, 300, 80):
        d = differences(omega0)
        gain0 = np.sum(d*projected/scale**2)/np.sum(d*d/scale**2)
        answer = least_squares(objective, [omega0, gain0], bounds=([1., -100.], [400., 100.]),
                               max_nfev=100, xtol=1e-10)
        candidates.append((float(np.linalg.norm(answer.fun)), *map(float, answer.x)))
    score, omega, gain = min(candidates)
    prediction = gain*differences(omega)
    reference = gain*duhamel_clocks(omega)
    ladders = {}
    for steps in ((32, 64, 128), (64, 128, 256), (128, 256, 512)):
        values = [gain*midpoint_clocks(omega, n) for n in steps]
        d = np.array([abs(values[i]-values[i+1]).max(0) for i in (0, 1)])
        ladders[str(steps)] = dict(pair_delta_q_v=d.tolist(), observed_order_q_v=np.log2(d[0]/d[1]).tolist(),
                                  finest_exact_error_q_v=abs(values[-1]-reference).max(0).tolist())
    return dict(scope='Saved-only descriptive fit; no FE/CUDA calls; no admission or eigenvalue certification',
                initial_states_bit_identical=initial_identity, clock_hashes=pins, clock_signed_differences=rows,
                spatial_template_relative_residual_q_v=np.sqrt(residual/total).tolist(),
                fitted_omega_rad_s=omega, fitted_frequency_Hz=omega/(2*np.pi), fitted_acceleration_gain=gain,
                fit_weighted_residual=score,
                projection_fit_relative_residual_q_v=[float(np.linalg.norm((prediction-projected)[:, :, i])/np.linalg.norm(projected[:, :, i])) for i in (0, 1)],
                midpoint_modified_frequency_rad_s={str(n):float(2*n*np.arctan(omega/(2*n))) for n in (32, 64, 128, 256, 512)},
                scalar_only_ladder_predictions=ladders)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    answer = diagnose(args.saved_root)
    with open(args.output, 'x') as f:
        json.dump(answer, f, indent=2, allow_nan=False)
        f.write('\n')
