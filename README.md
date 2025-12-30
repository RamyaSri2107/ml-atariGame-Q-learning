# Reinforcement Learning Experiments (Q-Learning, DQN, PPO)

This repository contains Jupyter notebooks exploring classic Reinforcement Learning algorithms using OpenAI Gym environments.

The focus is on:
- Tabular Q-Learning on a discrete environment (FrozenLake-v1)
- **Deep RL / Policy Gradient approaches** (DQN + PPO-style code) on (CartPole-v1)
- Comparing learning curves / total episode rewards

---

## Notebooks

### 1) Q_Learning_and_Policy_Iteration.ipynb
Goal: Train a tabular Q-learning agent on `FrozenLake-v1`.

What it includes:
- Q-table initialization
- Epsilon-greedy action selection
- Bellman Q-learning update
- Evaluation helper (`agent_performance`)
- Simple hyperparameter sweep over:
  - learning rate (alpha)
  - discount factor (gamma)
  - exploration probability (epsilon)

---

### 2) Atari_game.ipynb (CartPole)
Goal: Experiment with deep RL methods on `CartPole-v1` and compare performance.

What it includes (partially implemented):
- DQN-style components:
  - PyTorch network model
  - replay buffer / experience replay
  - target network structure (intended)
- PPO-style components:
  - Actor-Critic network
  - PPO class skeleton
- Plot section intended to compare reward curves

---

## Setup

### Requirements
- Python 3.9+ (3.10+ recommended)
- Jupyter Notebook / JupyterLab
- Core packages:
  - `numpy`
  - `gym` (or `gymnasium`, depending on your local setup)
  - `matplotlib`
  - `torch` (for DQN/PPO notebook)

### Install
```bash
pip install numpy matplotlib torch gym
