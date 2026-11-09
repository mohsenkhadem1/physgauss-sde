# Literature positioning for the CVPR 2027 visual–physical world-model paper

Verified 9 November 2026. Scope: renderable 3D/4D scenes, physically grounded simulation, action-conditioned prediction, physical inference, active measurement, and physical-fidelity tests.

## Precise distinction behind our claim

A visually accurate reconstructed motion does **not** uniquely identify mechanics for a new action. This is distinct from saying that existing systems cannot take actions, cannot simulate, or never generalize. Several state-of-the-art systems **do** fit material properties, simulate counterfactual actions, and probe actively. Our proposed distinction is *task-conditioned prediction-error risk*: given candidate material/support/load explanations and a specified action, determine whether the predictions differ enough to matter and choose a new view or calibrated physical measurement only if its expected benefit exceeds its cost.

The proposed method uses known metric rest geometry and calibrated cameras. The current RGB-only branch uses tracked surface correspondences, not a full photometric posterior. Thus comparisons against methods that reconstruct unknown geometry from RGB must declare the different priors and inputs. The main RGB-only three-scenario GIC/PAC-NeRF/ours benchmark was specified at commit c610ef6e; its new aggregate scores were not present when this literature revision was prepared.

## 1. Dynamic 4D appearance reconstruction — time-conditioned, not general physical action-conditioned

- **4D Gaussian Splatting** (Wu et al., CVPR 2024): learns time-conditioned Gaussian deformation for high-speed novel-time and novel-view rendering. [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.html)
- **Deformable 3D Gaussians** (Yang et al., CVPR 2024): canonical Gaussians with learned deformation for monocular dynamic reconstruction. [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Deformable_3D_Gaussians_for_High-Fidelity_Monocular_Dynamic_Scene_Reconstruction_CVPR_2024_paper.html)
- **Spacetime Gaussian Feature Splatting** (Li et al., CVPR 2024): compact spatiotemporal Gaussians for view synthesis. [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Spacetime_Gaussian_Feature_Splatting_for_Real-Time_Dynamic_View_Synthesis_CVPR_2024_paper.html)

**Gap at end of category:** excellent reconstructions at new times/views do not constitute a measurable force-to-deformation transition under a new commanded load.

## 2. Physics-enabled rendering — physically simulated or generated motions

- **PhysGaussian** (Xie et al., CVPR 2024): 3D Gaussians and MPM share a physical forward representation; many materials, generative interventions. [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_PhysGaussian_Physics-Integrated_3D_Gaussians_for_Generative_Dynamics_CVPR_2024_paper.html)
- **Spring-Gaus** (Zhong et al., ECCV 2024): spring-mass 3D Gaussian simulation, action-capable dynamics. [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00271.pdf)
- **PhysDreamer** (Zhang et al., ECCV 2024): optimises physically plausible response of static 3D assets through generative-video priors and accepts external interactions. [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00270.pdf)
- **PhysFlow** (Liu et al., CVPR 2025): MLLM material initialization, MPM, and video/optical-flow guidance; this is more than a forward-only simulator. [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Unleashing_the_Potential_of_Multi-modal_Foundation_Models_and_Video_Diffusion_CVPR_2025_paper.html)
- **PhysSplat** (Zhao et al., ICCV 2025): physically animated segmented objects with MLLM-guided properties. [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_PhysSplat_Efficient_Physics_Simulation_for_3D_Scenes_via_MLLM-Guided_Gaussian_ICCV_2025_paper.html)
- **NGFF** (Li et al., ICLR 2026): learned neural Gaussian force fields integrating RGB perception and dynamic 4D video rendering. [ICLR](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8365c06948c91169bb74b1b0629c68a2-Abstract-Conference.html)
- **GaussianFluent** (Huang et al., CVPR 2026): mixed materials, fracture, Gaussian interior rendering and CD-MPM. [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_GaussianFluent_Gaussian_Simulation_for_Dynamic_Scenes_with_Mixed_Materials_CVPR_2026_paper.html)
- **PhysGM** (Lv et al., CVPR 2026): feed-forward image-to-physical-Gaussian asset with simulated 4D dynamics. [CVF paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Lv_PhysGM_Large_Physical_Gaussian_Model_for_Feed-Forward_4D_Synthesis_CVPR_2026_paper.pdf)

**Gap at end of category:** a unified or simulator-driven rendering representation does not itself measure uncertainty from incorrectly assigned forces, attachments or materials, nor certify an action-specific force/response tolerance.

## 3. Inverse rendering and physics estimation — strongest simulation-and-rendering precedents

- **PAC-NeRF** (Li et al., ICLR 2023): differentiable volume rendering and MPM material identification from multiview video. [Paper](https://openreview.net/forum?id=tVkrbkz42vc)
- **GIC** (Cai et al., NeurIPS 2024): Gaussian-informed continuum geometry, render masks, and material identification. [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/89379d5fc6eb34ff98488202fb52b9d0-Abstract-Conference.html)
- **MASIV** (Zhao et al., ICCV 2025): learned constitutive laws and dense motion guidance, an important generalization of fixed material families. [IEEE](https://doi.org/10.1109/ICCV51701.2025.00562)
- **PhysTwin** (Jiang et al., ICCV 2025): springs, generative shape, Gaussian appearance, real manipulation, and changed-interaction simulations. [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Jiang_PhysTwin_Physics-Informed_Reconstruction_and_Simulation_of_Deformable_Objects_from_Videos_ICCV_2025_paper.html)
- **M-PhyGs** (Wada et al., CVPR 2026 Findings): jointly infers material composition and parameters, evaluates changed flower manipulations. [arXiv](https://arxiv.org/abs/2512.16885)
- **ReconPhys** (Wang et al., ECCV 2026): feed-forward 3D Gaussian appearance and mass/stiffness/damping/friction predictions from single-view video. [arXiv](https://arxiv.org/abs/2604.07882)
- **PhysConvex** (Wang et al., ECCV 2026): boundary-driven convex continuum representation and explicit physical generalization. [Project](https://danalaina.github.io/physconvex_webpage/)

**Gap at end of category:** many models already do changed-action simulation. What is not established just by a reconstruction score or selected held-out transfers is whether plausible competing physical explanations lead to materially different outcomes for a specified next force. One fitted point estimate is not an epistemic-risk certificate.

## 4. Action-conditioned video and learned world models — actions exist, but may have different semantics

- **P3Sim** (Lee et al., CVPR 2026): probabilistic world variables with partial geometric transformation signals and persistent scene memory. [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Lee_Perceptual_3D_Simulation_With_Physical_World_Modeling_CVPR_2026_paper.html)
- **Physical Object Understanding with a Physically Controllable World Model** (Venkatesh et al., CVPR 2026): distributions over visual/dynamic variables support object understanding and 3D manipulation. [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Venkatesh_Physical_Object_Understanding_with_a_Physically_Controllable_World_Model_CVPR_2026_paper.html)
- **Dexterous World Models** (Kim et al., CVPR 2026): action-conditioned egocentric videos given hand trajectories and renderings. [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_Dexterous_World_Models_CVPR_2026_paper.html)
- **Goal Force** (Gillman et al., CVPR 2026): explicit force vectors and motion goals condition video generation. Do NOT claim all learned video models lack force inputs. [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Gillman_Goal_Force_Teaching_Video_Models_To_Accomplish_Physics-Conditioned_Goals_CVPR_2026_paper.html)
- **PhyCo** (Narayanan et al., CVPR 2026): physical-property-map-conditioned video diffusion. [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Narayanan_PhyCo_Learning_Controllable_Physical_Priors_for_Generative_Motion_CVPR_2026_paper.html)
- **tau0-WM** (Zhou et al., 2026 arXiv): multi-view video/action policy plus action-conditioned video simulator and task-progress scoring. [arXiv](https://arxiv.org/abs/2606.01027)

**Gap at end of category:** generating plausible action-conditioned video or progress scores does not imply calibrated metric deformation or reaction-force error for a new load and does not automatically choose what physical measurement to collect next.

## 5. Interaction calibration and active physical identification

- **Latent force GPs** (Alvarez et al., 2009; Nayek et al., 2019; Rogers et al., 2020) and **goal-oriented Bayesian design** (Attia et al., 2018): established prior foundations for integrating uncertain forces and choosing measurements for target quantities.
- **KnockGS** (Ge et al., August 2026 preprint): observes a known calibration probe and freezes inferred physical Gaussian scales to test a new interaction. Already demonstrates probe-to-probe transfer. [arXiv](https://arxiv.org/abs/2608.27365)
- **PhysCoRe** (Yin et al., CoRL 2026): material inference plus learned residual correction, per-region confidence and exploration; active probing is not new by itself. [Project](https://lunarlab-gatech.github.io/PhysCoRe-website/)

**Gap at end of category:** pre-selected calibration and exploring locally uncertain material are not equivalent to minimizing error for the *particular intended action* at a specified measurement cost. Use strong fixed, point-estimate, and Gaussian-risk controls; our previous 96-case study does not establish a large advantage over the latter.

## 6. Benchmarks for physical truth and model reliability

- **PhysGaia** (Kim et al., CVPR 2026): multi-body scenes, material diversity, 3D particle truth and physical metrics for dynamic view synthesis. [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_PhysGaia_A_Physics-aware_Benchmark_with_Multi-Body_Interactions_for_Dynamic_Novel_CVPR_2026_paper.html)
- **PhysInOne** (Zhou et al., CVPR 2026): millions of simulated physical videos, multiple phenomena, future prediction and property benchmarks. [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_PhysInOne_Visual_Physics_Learning_and_Reasoning_in_One_Suite_CVPR_2026_paper.html)
- **Deform360** (Li et al., ECCV 2026): multi-view visuotactile deformable world modeling benchmark. [arXiv](https://arxiv.org/abs/2607.05390)
- **GaussianBench** (Dumebi-Kachikwu, October 2026 preprint): conservation, heterogeneous material, counterfactual interaction and simulator-independent scoring; already exposes failures of Gaussian simulators. [arXiv](https://arxiv.org/abs/2610.10554)

**Gap at end of category:** simulation-fidelity benchmarks typically evaluate the resulting prediction or rollout; our planned decision test evaluates the probability of error *before* the next action and the prospective benefit of a new view versus a known-force measurement.

## Recommended CVPR evaluation framing

1. **Figure 1 — visual fit vs held-out action response:** same RGB observations, known commands, held camera, and unseen deformation/impulse; comparisons GIC, PAC-NeRF, ours, with relevant representational-capacity controls. Keep completed stem D2b distinct from the as-yet-unscored Fig. 1 D2c protocol.
2. **Figure 2 — what additional evidence should be acquired:** same observed prefix, different target actions; proceed/look/probe with equal budget; best fixed, random, point-MAP and Gaussian-risk baselines; report failure-risk calibration, not just mean RMSE.
3. **Figure 3 — different sources of ambiguity:** force/material scale, support compliance, spatial heterogeneity; distinguish unidentifiability from model/solver mismatch.
4. **Stress test — unmodeled contact:** the bowl study already shows 1.86% predicted failure versus 43.75% realized against a finer reference; do not imply the current posterior is calibrated under arbitrary model errors.

## Text hygiene for review

Do not say existing methods 'have no action inputs', 'cannot simulate novel interactions', or 'do not estimate physical properties'. Some do. Do not imply our current estimator jointly optimizes every rendered pixel; its measured input is image-derived tracks with supplied metric geometry. Do not present adapted mechanics-only baselines as official end-to-end reconstructions. Distinguish CVPR/ECCV accepted work from preprints. Cite official CVF/ECVA/ICLR sources where available, and avoid numerical comparisons across distinct datasets or sensor inputs.
