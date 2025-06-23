# 🧠 Blocks World RL Agents
[Screencast from 2025-06-23 15-06-15.webm](https://github.com/user-attachments/assets/7da4cdcf-dc15-444c-bc8c-c4c6af080afa)

This project implements Reinforcement Learning agents to solve the classic **Blocks World** environment using:

- **Q-Learning** (Tabular)
- **DQN** (Deep Q-Networks)
- **PPO** (Proximal Policy Optimization)

Agents learn to rearrange blocks to match goal configurations, navigating a symbolic planning space integrated with logic-based rules from Prolog.

---

## 🚀 Overview

**Blocks World** is a simplified environment often used in AI for symbolic reasoning and planning. Here, we integrate it with Gym-style RL interfaces, allowing deep RL agents to train and learn efficiently.

---

## 🧰 Algorithms

| Algorithm    | Type         | Description                                 |
|--------------|--------------|---------------------------------------------|
| Q-Learning   | Value-based  | Classic tabular method for small state spaces |
| DQN          | Value-based  | Neural network approximator for Q-values     |
| PPO          | Policy-based | Actor-Critic method using policy gradients   |

---

## 🏗️ Environment

- Custom **Blocks World** domain
- Backend: **Prolog** (via `swiplserver`)
- Frontend: Python `gymnasium`-style interface
- Optionally visualized with `pygame`
