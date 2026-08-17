# Hex (9x9) AI Benchmark: Heuristics vs CMA-ES vs Maskable PPO

An artificial intelligence benchmark for the strategic board game **Hex (9x9)**, evaluating and comparing graph-based heuristics, evolutionary optimization (CMA-ES), and deep reinforcement learning (Maskable PPO with a deep ResNet architecture).

> 📄 **Project Report (in Polish):** The full, detailed academic report is available in the repository: [`Sprawozdanie_Zachowanie botów optymalizacja vs uczenie vs heurystyka.pdf`](./Sprawozdanie_Zachowanie%20botów%20optymalizacja%20vs%20uczenie%20vs%20heurystyka-1.pdf).

---

## 🏆 Key Highlights & Results

* **Maskable PPO (3.5M steps)**: Emerged as the undisputed tournament winner with a **92.9% overall win rate** across a 90,000-game round-robin tournament. The model naturally learned advanced Hex concepts, including distance blocking, bridge connections, and optimal swap-rule utilization.
* **CMA-ES Optimization**: Successfully evolved linear evaluation weights for 12 engineered topological features, matching mid-stage deep RL agents (50.9% win rate vs PPO 1.5M).
* **Advanced RL Training Pipeline**:
  * **Imitation Learning (Behavioral Cloning)**: Pretrained on 8,000 heuristic matches (`bc_resnet_deep_9x9.pth`) with cross-entropy loss to accelerate early exploration.
  * **Critic Warmup**: 200,000 steps with frozen feature extraction and policy heads to stabilize value estimation.
  * **League Training**: Dynamic matchmaking (50% heuristics, 25% past checkpoints, 25% self-play) to prevent catastrophic forgetting and narrow-strategy over-exploitation.
  * **Action Masking & Rotational Symmetry**: Used action masking for illegal moves alongside 180° board rotation data augmentation.

### 📊 Tournament Results Matrix

| Rank | Agent / Model | Win Rate (%) |
| :--- | :--- | :--- |
| **#1** | **Maskable PPO (3.5M steps)** | **92.90%** |
| **#2** | Maskable PPO (3.0M steps) | 85.13% |
| **#3** | Maskable PPO (2.0M steps) | 71.99% |
| **#4** | Maskable PPO (2.5M steps) | 69.72% |
| **#5** | Maskable PPO (1.5M steps) | 56.87% |
| **#6** | CMA-ES (12-feature linear) | 51.62% |
| **#7** | Maskable PPO (1.0M steps) | 35.11% |
| **#8** | Maskable PPO (500k steps) | 22.93% |
| **#9** | Heuristic Agent (0-1 BFS) | 12.51% |
| **#10** | Blind Heuristic Baseline | 1.22% |

---

## 📂 Repository Structure

* `GUI.py` — Interactive graphical user interface (GUI) to play against any trained bot or human player.
* `RL_PPO.py` — Deep Reinforcement Learning pipeline (ResNet architecture, Maskable PPO agent, league system, and evaluation).
* `cma_es.py` — Evolutionary optimization script using Covariance Matrix Adaptation Evolution Strategy.
* `best_cma_es.npy` — Optimized weight vector obtained via CMA-ES.
* `bc_resnet_deep_9x9.pth` — Initial network weights from the Behavioral Cloning phase.
* `league_checkpoints/` — Directory containing historical and trained PPO model checkpoints (`.zip`).
* `Sprawozdanie_Zachowanie botów optymalizacja vs uczenie vs heurystyka.pdf` — Complete project report (in Polish).

---

## 🚀 Quickstart

### 1. Requirements & Setup
Clone the repository and install the dependencies:
```bash
git clone https://github.com/damiankrzyzelewski/HEX9x9_bot.git
cd HEX9x9_bot
pip install torch torchvision stable-baselines3 sb3-contrib cma numpy pygame
```

### 2. Download Model Checkpoints
1. Open the [Releases](../../releases/latest) section.
2. Download the model `.zip` files from the release assets.
3. Place the downloaded `.zip` files into the `league_checkpoints/` directory:

```text
league_checkpoints/
├── PPO_500000_steps.zip
├── PPO_1000000_steps.zip
├── PPO_1500000_steps.zip
├── PPO_2000000_steps.zip
├── PPO_2500000_steps.zip
├── PPO_3000000_steps.zip
└── PPO_3500000_steps.zip
```

### 3. Launch the Game GUI
To play against the bots, simply run:
```bash
python GUI.py
```
