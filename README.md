# 🎮 Flappy Bird RL Agent: Deep Q-Network (DQN) Implementation

An end-to-end Reinforcement Learning pipeline designed to train an autonomous agent to play Flappy Bird using Deep Q-Networks (DQN). Built using PyTorch, this project demonstrates deep Q-learning fundamentals, environment vectorization, state preprocessing, experience replay, and target network stabilization for optimal reward convergence.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)
![Domain](https://img.shields.io/badge/Domain-Reinforcement%20Learning-green)
![Algorithm](https://img.shields.io/badge/Algorithm-Deep%20Q--Network-orange)

---

## 📌 Executive Summary

Reinforcement Learning in dynamic arcade environments presents unique challenges due to high-dimensional state spaces and temporal credit assignment. This repository implements a custom PyTorch RL pipeline where the agent learns frame-by-frame navigation strategies, optimal jump timing, and obstacle avoidance purely through trial-and-error interactions and scalar reward feedback.

---

## 🛠️ Reinforcement Learning Architecture & Pipeline

### 1. Environment & State Preprocessing
- **Frame Processing:** Extracted raw visual/game state vectors (player position, velocity, pipe distance, gap orientation).
- **State Stacking:** Stacked consecutive frames to capture temporal velocity dynamics and motion direction.

### 2. Deep Q-Network Architecture
- **Q-Function Approximation:** Implemented a Deep Neural Network in PyTorch to map state inputs to predicted Q-values for discrete actions (`Jump` vs. `Do Nothing`).
- **Target Network Synchronization:** Maintained a separate Target Network periodically updated from the policy network to stabilize temporal difference (TD) target updates.

### 3. Experience Replay Memory
- **Replay Buffer:** Stored `(state, action, reward, next_state, done)` tuples to break temporal correlation between consecutive training samples.
- **Minibatch Sampling:** Sampled uniform random batches during training to promote stable gradient steps.

### 4. Policy Optimization & Exploration
- **Epsilon-Greedy Policy:** Decay-based $\epsilon$-greedy strategy balancing exploration of novel flight paths with exploitation of learned high-reward trajectories.
- **Bellman Equation Minimization:** Trained network parameters via Mean Squared Error (MSE) loss between predicted Q-values and Bellman optimality targets.

---
