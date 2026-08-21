# Learning Environment Dynamics with Visual Neural World Models

A PyTorch project that learns action-conditioned dynamics of visual interactive environments using MiniGrid.

## Project question

> Can a compact neural network learn the dynamics of an interactive visual environment well enough to predict future trajectories on unseen environments?

## Scope

Core system:

```text
MiniGrid
  ↓
trajectory collection
  ↓
baseline dynamics
  ↓
CNN encoder
  ↓
latent action-conditioned dynamics
  ↓
decoder
  ↓
one-step prediction
  ↓
multi-step rollout
  ↓
generalization analysis
```

Planned extension:

```text
learned world model
  ↓
imagined trajectories
  ↓
model-predictive planning
```

## Repository structure

- `src/envs/` — MiniGrid environment construction
- `src/data/` — trajectory collection and datasets
- `src/models/` — baseline, encoder, decoder, and world-model components
- `src/training/` — training loops and losses
- `src/evaluation/` — prediction, rollout, and generalization metrics
- `src/planning/` — model-based planning extension
- `src/visualization/` — prediction and experiment visualizations
- `configs/` — reproducible experiment configurations
- `experiments/` — exploratory notebooks and experiment entry points
- `experiments/kaggle/` — Kaggle-specific workflow documentation
- `scripts/` — command-line experiment utilities
- `results/` — tracked figures/tables; large checkpoints stay out of normal Git history
- `demo/` — eventual interactive visualization
- `tests/` — automated checks

## Compute workflow

**GitHub is the source of truth. Kaggle is the compute environment.**

The reusable project code lives in GitHub. Kaggle notebooks are used for GPU execution, experiment orchestration, visualization, and notebook-level analysis. We do not depend on a persistent Colab filesystem or repeatedly copy the project into a notebook.

See [`experiments/kaggle/README.md`](experiments/kaggle/README.md) for the workflow and artifact policy.

## Checkpoints

- `Checkpoint 00` — repository and environment foundation
- `Checkpoint 01` — MiniGrid integration
- `Checkpoint 02` — trajectory collection
- `Checkpoint 03` — dataset and seed splits
- `Checkpoint 04` — persistence baseline
- `Checkpoint 05` — state-space dynamics model
- `Checkpoint 06` — CNN autoencoder
- `Checkpoint 07` — latent dynamics
- `Checkpoint 08` — multi-step rollout
- `Checkpoint 09` — pixel vs. latent experiments
- `Checkpoint 10` — unseen-layout generalization
- `Checkpoint 11` — planning extension
- `Checkpoint 12` — demo and final results

## Setup

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

The project configuration is also available through `pyproject.toml`.
