## RL Baseline (MiniGrid / BabyAI)

This folder contains the reinforcement learning (RL) baseline used for comparison in the SPCA paper.

The implementation is **derived from** the public repository:

> **lcswillems/rl-starter-files**
> [https://github.com/lcswillems/rl-starter-files](https://github.com/lcswillems/rl-starter-files)

Only files that differ from the original repository are included here.

---

## Setup

### 1. Clone the Original Repository

First, clone the upstream `rl-starter-files` repository:

```bash
git clone https://github.com/lcswillems/rl-starter-files.git
cd rl-starter-files
```

### 2. Merge Modified Files

Copy the modified files from this folder into the cloned repository, overwriting the originals:

```bash
# From the rl-starter-files clone directory:
cp -r /path/to/minigrid-crewai/rl-baselines/rl-starter-files/* .
```

Or if you're in the `minigrid-crewai/rl-baselines/rl-starter-files` directory:

```bash
cp -r ./* /path/to/rl-starter-files/
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Training a PPO Agent

Train a PPO agent on a BabyAI environment:

```bash
python3 -m scripts.train --algo ppo --env BabyAI-GoToRedBallGrey-v0 --model MyModel-v1 --save-interval 10 --frames 250000 --seed 1
```

**Key arguments:**
- `--algo`: Algorithm (`ppo` or `a2c`)
- `--env`: BabyAI environment name (e.g., `BabyAI-GoToRedBallGrey-v0`, `BabyAI-PickupLoc-v0`)
- `--model`: Name for the model (saved under `storage/<model>/`)
- `--frames`: Total training frames (default: 10M)
- `--seed`: Random seed for reproducibility
- `--text`: Add this flag for environments with language instructions

Training stops automatically after 5 consecutive evaluations with SR ≥ 99%.

### 2. Evaluating a Trained Model

```bash
python3 -m scripts.evaluate --env BabyAI-GoToRedBallGrey-v0 --model MyModel-v1 --episodes 512
```

**Key arguments:**
- `--episodes`: Number of evaluation episodes (default: 100)
- `--argmax`: Use deterministic actions (highest probability)
- `--text`: Required if model was trained with `--text`

### 3. Running Multiple Seeds (Ensemble Training)

Train 10 seeds for statistical analysis:

```bash
for s in {1..10}; do
  python3 -m scripts.train --algo ppo --env BabyAI-GoToRedBallGrey-v0 \
    --model GoToRedBall-v$s --save-interval 10 --frames 250000 --seed $s
done
```

---

## Analysis Scripts

### Analyze Success Rate Metrics

Compute SR metrics (SR at 500 episodes, episodes to 70% SR, episodes to 99% SR) across trials:

```bash
python3 analyze_sr_metrics.py --storage storage --levels PickupLoc GoToRedBall GoToRedBallGrey GoToLocal --csv sr_metrics_results.csv
```

### Plot Ensemble SR Curves

Generate mean SR vs episodes plot with confidence bands:

```bash
python3 plot_sr_ensemble.py --storage storage --match PickupLoc --tail-mode fill --tail-fill 0.99 --band std --out storage/PickupLoc-ensemble.png --show
```

**Options:**
- `--match`: Substring to match trial folder names
- `--tail-mode`: How to handle trials that finish early (`cut`, `hold`, `fill`)
- `--band`: Band type (`ci` for 95% CI, `std` for ±1 SD)

---

## Directory Structure

```
rl-starter-files/
├── scripts/
│   ├── train.py          # Training script
│   └── evaluate.py       # Evaluation script
├── model.py              # FiLM-based AC model (BabyAI architecture)
├── utils/                # Agent and utility functions
├── storage/              # Trained models and logs
├── analyze_sr_metrics.py # Metrics analysis across trials
├── plot_sr_ensemble.py   # Ensemble plotting
└── SR_run.sh             # Example commands
```
