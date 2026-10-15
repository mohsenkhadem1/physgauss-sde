"""Self-contained experiment report with plots and an animated comparison."""
import base64
import html
import io
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = {"physical": "#2563eb", "physical_sde": "#d97706"}
LABELS = {"physical": "Physical + parameter uncertainty", "physical_sde": "Physical + neural SDE"}


def encoded(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def animation(path, benchmark, results):
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    observed = {int(idx): i for i, idx in enumerate(benchmark.observations.indices)}
    frames = []
    for t in range(benchmark.config.frames):
        frame = Image.new("RGB", (800, 260), "#111827")
        draw = ImageDraw.Draw(frame)
        images = [benchmark.truth_rgb[t],
                  benchmark.observations.rgb[observed[t]] if t in observed else None,
                  results["physical"]["samples"]["rgb"][:, t].mean(0),
                  results["physical_sde"]["samples"]["rgb"][:, t].mean(0)]
        labels = ["Ground truth", "Available input", "Physical posterior", "Physical + SDE"]
        for column, (image, label) in enumerate(zip(images, labels)):
            left = column * 200
            draw.text((left + 10, 10), label, fill="white", font=font)
            if image is not None:
                array = (image.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
                thumb = Image.fromarray(array).resize((184, 184), Image.Resampling.BILINEAR)
                frame.paste(thumb, (left + 8, 40))
            else:
                draw.rectangle((left + 8, 40, left + 192, 224), fill="#1f2937")
                draw.text((left + 24, 120), "Missing frame", fill="#94a3b8", font=font)
        draw.text((10, 234), f"t={float(t * benchmark.config.duration / (benchmark.config.frames - 1)):.2f}s  |  Synthetic, known rest shape and camera", fill="#cbd5e1", font=font)
        frames.append(frame)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)


def make_report(output, benchmark, results, summary):
    output = Path(output)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.linspace(0, benchmark.config.duration, benchmark.config.frames)
    obs = benchmark.observations.indices.tolist()
    left, right = max(zip(obs[:-1], obs[1:]), key=lambda pair: pair[1] - pair[0])
    free = ~results["physical"]["model"].mesh.pinned
    node = int(np.flatnonzero(free.numpy())[-1])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    for method, result in results.items():
        x = result["samples"]["x"].numpy()
        lo, hi = np.quantile(x[:, :, node, 0], [.05, .95], axis=0)
        axes[0].plot(times, x[:, :, node, 0].mean(0), color=COLORS[method], label=LABELS[method])
        axes[0].fill_between(times, lo, hi, color=COLORS[method], alpha=.17)
        free_positions = x[:, :, free.numpy(), :]
        low, high = np.quantile(free_positions, [.05, .95], axis=0)
        axes[1].plot(times, (high - low).mean((1, 2)) * 1000,
                     color=COLORS[method], label=LABELS[method])
        history = np.asarray(result["fit"].history)
        axes[2].plot(history, color=COLORS[method], label=LABELS[method])
    truth = benchmark.truth.x[0, :, node, 0].numpy()
    axes[0].plot(times, truth, color="#111827", linestyle="--", label="Ground truth")
    axes[0].scatter(times[obs], truth[obs], s=18, color="#111827", zorder=5,
                    label="Observation times (images only)")
    axes[0].set(title="A free material point: x position", xlabel="Time (s)", ylabel="Position (m)")
    axes[1].set(title="Mean 90% interval width", xlabel="Time (s)", ylabel="Free-node coordinate width (mm)")
    axes[2].set(title="MAP objective", xlabel="Objective evaluations", ylabel="Negative log posterior + constant")
    for ax in axes[:2]:
        ax.axvspan(times[left], times[right], color="#64748b", alpha=.09)
    for ax in axes:
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("One moving camera, irregular observations | Shading: largest missing interval", fontsize=12)
    fig.savefig(output / "diagnostics.png", dpi=150)
    plt.close(fig)
    animation(output / "comparison.gif", benchmark, results)

    # Still contact sheet for review and repository previews.
    midpoint = (left + right) // 2
    sheet = Image.new("RGB", (800, 260), "#111827")
    gif = Image.open(output / "comparison.gif")
    gif.seek(midpoint)
    sheet.paste(gif.convert("RGB"))
    sheet.save(output / "preview.png")

    rows = []
    for key, result in results.items():
        metrics = result["metrics"]["long_gap"]
        rows.append("<tr><td>" + LABELS[key] + "</td>"
                    + f"<td>{metrics['position_rmse_m'] * 1000:.2f} mm</td>"
                    + f"<td>{metrics['rgb_psnr_db']:.2f}</td>"
                    + f"<td>{metrics['novel_view_psnr_db']:.2f}</td>"
                    + f"<td>{metrics['component_90_coverage']:.1%}</td>"
                    + f"<td>{metrics['component_90_mean_width_m'] * 1000:.2f} mm</td></tr>")
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>PhysGauss-SDE experiment</title>
<style>body{font:16px/1.6 system-ui,sans-serif;color:#182235;background:#f5f7fb;margin:0}main{max-width:1120px;margin:45px auto;padding:0 24px}h1{font-size:36px;line-height:1.2;margin-bottom:12px}h2{margin-top:32px;font-size:22px}.tag{color:#2563eb;font-size:13px;font-weight:700;letter-spacing:1px}p{max-width:950px}.notice{background:#fff7ed;border-left:4px solid #d97706;padding:15px 20px;border-radius:6px}img{width:100%;height:auto;background:white;border-radius:12px;border:1px solid #e2e8f0;margin-top:14px}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{text-align:left;padding:13px;border-bottom:1px solid #e2e8f0}th{color:#475569}pre{white-space:pre-wrap;font:12px/1.5 ui-monospace,monospace;background:#e8edf5;padding:20px;border-radius:8px}small{color:#475569}.scroll{overflow:auto}a{color:#2563eb}</style>
<main><div class="tag">RESEARCH PROTOTYPE / CPU BENCHMARK</div>
<h1>Physical motion across missing frames</h1>
<p>A single known camera moves around a clamped elastic solid. We fit only {{COUNT}} irregular RGB observations. Gaussian extent follows volumetric material deformation. The second model adds spatially correlated neural stochastic forcing.</p>
<div class="notice"><strong>Scope:</strong> known rest geometry, density, material family, appearance and camera poses. One synthetic scene; local Gaussian posterior approximation. The diffusion network uses separate simulator supervision. Coverage below is a diagnostic, not evidence of calibration across scenes. There is no real-video, unknown-geometry, contact-inference or PIDG baseline result yet.</div>
<h2>Observed and reconstructed motion</h2>
<p>Black input panels are missing observations. Both physical posteriors use observations before and after the gap. Each image shown is a posterior mean render.</p>
<img alt="Animated ground truth, input observations, physical reconstruction and SDE reconstruction" src="data:image/gif;base64,{{ANIMATION}}">
<h2>Largest missing interval</h2>
<p>3D and uncertainty metrics exclude clamped nodes. Novel-view images are evaluation-only, from a camera orbit offset by 100 degrees. Coverage and interval width should be interpreted together.</p>
<div class="scroll"><table><tr><th>Method</th><th>Position RMSE</th><th>PSNR (dB)</th><th>Novel PSNR (dB)</th><th>90% coverage</th><th>Mean width</th></tr>{{ROWS}}</table></div>
<img alt="Material-point trajectory, uncertainty interval widths and fitting objective" src="data:image/png;base64,{{PLOT}}">
<h2>Reproducibility and complete measurements</h2>
<p>Exact configuration, random seeds, dependency versions, source fingerprint, scores, and numerical diagnostics are recorded below. Gaussian shape covariance and posterior covariance are separate. The stochastic increments are inferred jointly with scene parameters; adding observation noise to a display does not count as trajectory uncertainty. MAP objectives show optimization progress within a model and are not model-evidence comparisons.</p>
<details><summary>Show complete experiment record</summary><pre>{{RECORD}}</pre></details>
<p><small>PhysGauss-SDE. No generated view is treated as a real observation. There are no hidden-frame targets in the fitting API.</small></p></main></html>"""
    document = (document.replace("{{ANIMATION}}", encoded(output / "comparison.gif"))
                .replace("{{PLOT}}", encoded(output / "diagnostics.png"))
                .replace("{{ROWS}}", "".join(rows))
                .replace("{{COUNT}}", str(len(obs)))
                .replace("{{RECORD}}", html.escape(json.dumps(summary, indent=2))))
    (output / "report.html").write_text(document, encoding="utf-8")
