# Plan G — JEPA Predictive Planning for District Energy Management

> **Status: Initial experiment / preliminary results.**
> This is an ongoing research project. The world model training and evaluation
> pipeline are complete. Ablations, stronger baselines, and statistical validation
> are in progress.

---

## What this is?

An offline latent world model for multi-building district energy control,
built on JEPA-style predictive learning and evaluated with CEM-based MPC.

The core idea: instead of training a policy with online environment interaction,
learn a compressed latent model of microgrid dynamics purely from pre-collected
transition data, then plan over that latent space at test time. No reward shaping
during learning. No environment access during training.

This connects directly to the line of work in
[Colombini et al. (2024) — *Learning To Explore With Predictive World Model Via
Self-Supervised Learning*](https://arxiv.org/abs/XXXX), which uses JEPA-style
predictive representations to support exploration. Where that work focuses on
exploration policy, this applies the same representational backbone to offline
planning in a multi-agent energy management setting.

---

## Architecture
```
State Window (24h) → RevIN → Transformer Encoder → Latent z_t (64-dim)
                                                          ↓
                              Action a_t →→→ Latent Dynamics MLP → z_{t+1}
                                                          ↓
                                              VICReg Regularization
                                                          ↓
                                         CEM Planner (H=6, N=512 samples)
                                                          ↓
                                              CityLearn Environment
```

**Encoder:** 4-layer Temporal Transformer, 24-step input window, mean pooling
to 64-dim latent. RevIN normalization applied pre-embedding.

**Dynamics model:** 3-layer MLP with residual connection. Biased toward learning
incremental transitions: `z_{t+1} = z_t + Δ(z_t, a_t)`.

**Training loss:** VICReg (Variance-Invariance-Covariance Regularization) with
λ_inv=1, λ_var=25, λ_cov=1. Prevents dimensional collapse without a separate
target network.

**Planner:** Cross-Entropy Method. Iterative refinement of a Gaussian
distribution over H-step action trajectories. Top-k elite selection per
iteration.

**Masking:** Structured feature masking during training, grouped by physical
category (SOC/temperature: 40%, grid signals: 30%, weather: 10%).

---

## Data

CityLearn 2022 Challenge environment: 17 residential buildings, one full
year (8760 timesteps). Transition dataset extracted via random exploration:
148,920 `(s_t, a_t, s_{t+1})` tuples across 20 episodes.

`CityLearn_DataExtraction.ipynb` handles all data collection.
`plan_g_jepa__3_.ipynb` contains the full training and evaluation pipeline.

---

## Preliminary Results

Evaluated over 2 episodes against a random baseline. Results should be
treated as directional, not conclusive; statistical validation across
more episodes is ongoing.

| Metric | Random Baseline | JEPA-MPC | Change |
|---|---|---|---|
| Total Reward | 181882.1502 | 166244.0027 | 8.6% |
| Peak Demand (kW) | 14.5298 | 11.6458 | 19.8% |
| Carbon (kg CO₂) | 28624.3057 | 25551.3334 | 10.7% |
| Electricity Cost ($) | 28624.3057 | 25551.3334 | 3.7% |

**Latent space diagnostics:**
- All 64 latent dimensions active (no collapse under VICReg)
- t-SNE shows hour-of-day clustering. Temporal structure learned
- Rollout MSE over 5 steps: 0.39 → 0.61 (graceful degradation, no divergence)

---

## What's missing / what comes next?

This is a research-in-progress snapshot. Known gaps:

- [ ] Rule-based controller baseline (charge-on-solar / discharge-at-peak)
- [ ] SAC / PPO trained-from-scratch comparisons
- [ ] Ablations: no VICReg, no masking, K=1 vs K=5, latent dim 32 vs 64
- [ ] 10+ evaluation episodes for mean ± std reporting
- [ ] Comparison against CityLearn 2022 challenge leaderboard numbers
- [ ] SOC-rich training data (current dataset has mostly inactive storage)

---

## Setup

Designed for Google Colab free tier. RAM-constrained, lazy dataset
construction, fp16 mixed precision, gradient checkpointing.
```bash
pip install citylearn numpy==1.26.4 pandas==2.2.2
```

1. Run `CityLearn_DataExtraction.ipynb` to generate `transitions_all_buildings.pkl`
2. Upload to Google Drive
3. Run `plan_g_jepa__3_.ipynb` — all hyperparameters in Cell 2

---

## Connection to prior work

| Component | Inspiration |
|---|---|
| JEPA objective | LeCun (2022) — A Path Towards Autonomous Machine Intelligence |
| VICReg | Bardes et al. (2022) |
| Predictive world model + exploration | Colombini et al. (2024) |
| CEM-MPC | Chua et al. (2018) — PETS |
| District energy environment | Nweye et al. (2022) — CityLearn |

---

## Author

**Chidera** — MTech AI, Vivekananda Global University

Research interests: latent world models, Self-Supervised Learning, predictive representations, planning & control, hierarchical reasoning,
medical imaging, remote sensing.

---

## GitHub repo About section
```
Offline JEPA world model + CEM-MPC for multi-building district energy control. Preliminary experiment. CityLearn 2022.
```

---

## Pinned repo blurb
```
Plan G: trained a latent world model on 148K building energy transitions using JEPA + VICReg, then planned over it with CEM-MPC. 19.8% peak demand reduction vs random baseline. Ongoing.
