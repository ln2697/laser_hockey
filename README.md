# Laser Hockey RL Agents

Two reinforcement learning approaches for competitive hockey: **Model Predictive Q (MPQ)** - tournament winner, and **Soft Actor-Critic with Random Network Distillation (SAC-RND)**.

## Quick Start

**Run pretrained model:**
```bash
python3 hockey_agent.py
```

**Train from scratch:**
```bash
pip install -r requirements.txt
bash scripts/001_baseline.sh
```

Training: ~5-7 hours on GTX 1080ti + 8-12 CPUs. Results save to `runs/001_baseline_<timestamp>`.

## Methods

### MPQ Agent (1st Place 🏆)
- **Approach**: Learned dynamics + 1-step minimax planning
- **Actions**: 25 discrete
- **Features**: Observation augmentation, puck forecasting, Mixture of Experts
- **Performance**: 96% vs strong bot, 99% vs weak bot

### SAC-RND Agent (28th Place)  
- **Approach**: Maximum entropy RL + intrinsic exploration
- **Actions**: 4 continuous 
- **Features**: Random Network Distillation, auto entropy tuning, PER
- **Performance**: 77% vs strong bot, 81% vs weak bot

## Environment

Hockey simulation with two competing agents. 17D state space (positions, velocities, angles). 250-step episodes. Binary rewards: +10 goal, -10 concede.

---
*Reinforcement Learning 2024/25 Final Project - Team Agent-Q*