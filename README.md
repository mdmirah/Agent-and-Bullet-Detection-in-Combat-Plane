<h1 align="center">Agent and Bullet Detection in Combat: Plane</h1>

In this project Ayden Kemp [@aydensairplanes](https://github.com/aydensairplanes) and I have implemented a **multi-agent reinforcement learning system** for the Combat Plane environment using **PettingZoo** and **Stable-Baselines3**. This project augments the Combat:Plane environment by adding cutom functions for feature extraction including agent and bullet detection and target ranging with the goal of supporting the design of custom heuristics for offensive, defensive and hybrid agents. By using these custom functions, unique reward functions for **offensive**, **defensive**, and **hybrid** agents are created, tested and validated. We have seen our custom offensive and defensive agents both outperform (As per their designed heuristic) the baseline agent of the Combat: Plane environment. A **hybrid** agent has also been designed in the codebase which is to be tested in future work.

---

## Overview

The project trains intelligent agents to play the **Combat Plane game (bi-plane version)** from the Atari environment. The agents use **PPO (Proximal Policy Optimization)** and incorporate **custom reward functions** to learn specific combat strategies.

---

## Features

### Core Components

- **Plane Detection:** Identifies agent and enemy positions using R-channel analysis (Blue and Red circles).
<img width="488" height="374" alt="image" src="https://github.com/user-attachments/assets/38472cc6-7ae6-4322-aa3d-4f904e37905b" />

- **Bullet Detection:** Computer vision-based bullet tracking with filtering (Green circles).
<img width="488" height="381" alt="image" src="https://github.com/user-attachments/assets/ed27f755-29fa-4b52-a3fa-aafe0f37c5f2" />

- **Toroidal Distance Calculations:** Handles wrap-around game mechanics for the Atari environment.
- **Custom Reward Engineering:** Domain-specific heuristic driven reward shaping for offensive and defensive agents as well as an experimental hybrid agent.

### Agent Types

- **Offensive Agent (`OffensiveRewardWrapper`)**
  - Rewards for shooting actions and proximity to enemy
  - Amplifies hit rewards with multiplier
  - Encourages aggressive gameplay

- **Defensive Agent (`DefensiveRewardWrapper`)**
  - Rewards for maintaining distance from enemy
  - Penalizes firing actions
  - Provides bullet avoidance rewards
  - Applies damage penalties when hit

- **Hybrid Agent (`HybridRewardWrapper`)** (Optional and Experimental)
  - Combines offensive and defensive strategies
  - Tracks own bullets to distinguish from enemy bullets
  - Balanced reward system for versatile gameplay

---

## Installation

```bash
# Install dependencies
pip install pettingzoo[atari] supersuit stable-baselines3[extra] moviepy gymnasium==1.1.1 imageio ffmpeg-python -q
pip install autorom[accept-rom-license] -q
AutoROM --accept-license
pip install shimmy>=2.0 -q
pip install --upgrade pettingzoo
```

## Usage
Basic Setup
```
import numpy as np
import imageio
from PIL import Image
import matplotlib.pyplot as plt

from pettingzoo.atari import combat_plane_v2
import supersuit as ss
from stable_baselines3 import PPO

from IPython.display import HTML
from base64 import b64encode

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os

import cv2
from pettingzoo.utils.conversions import aec_to_parallel
from pettingzoo.utils.wrappers.base import BaseWrapper
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper
import pettingzoo.utils.env

# Create base environment
env = combat_plane_v2.env(render_mode='rgb_image', game_version='bi-plane', guided_missile=False)
env.reset()
```
Agent Selection
```
# Choose your agent type:
# env = OffensiveRewardWrapper(env)  # Offensive Agent
env = DefensiveRewardWrapper(env)    # Defensive Agent
# env = HybridRewardWrapper(env)     # Hybrid Agent (Optional and Experimental)

# Apply standard preprocessing
env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 4)
env = ss.black_death_v3(env)
```
Training
```
from stable_baselines3.common.callbacks import CheckpointCallback

# Setup checkpointing
checkpoint_callback = CheckpointCallback(
    save_freq=100,
    save_path="./models/",
    name_prefix="ppo_agent"
)

# Initialize and train model
model = PPO("CnnPolicy", env, n_steps=768, batch_size=256, verbose=1)
model.learn(total_timesteps=10000, callback=checkpoint_callback)
```
## Custom Functions
Image Extraction and Filtering to Red (R) Channel Only
```
def get_cleaned_R(observation):
    """
    Extracts the R channel from the observation and zeroes out known scoreboard regions.
    Parameters:
        observation (ndarray): A (256, 160, 3) observation image.
    Returns:
        R_cleaned (ndarray): A (256, 160) array with scoreboard regions set to 0.
    """
    R = observation[:, :, 0].copy()
    # Zero out first_0's scoreboard (rows 4:14, cols 32:44)
    R[4:14, 32:44] = 0
    # Zero out second_0's scoreboard (rows 4:14, cols 87:99)
    R[4:14, 112:124] = 0
    return R
```

Toroidal Distance and Mean Functions
```
def toroidal_distance(pos1, pos2, max_row, max_col):
    if pos1 is None or pos2 is None:
        return None  # Distance undefined if any position missing
    d_row = abs(pos1[0] - pos2[0])
    d_col = abs(pos1[1] - pos2[1])
    # Wrap around for rows
    if d_row > max_row / 2:
        d_row = max_row - d_row
    # Wrap around for cols
    if d_col > max_col / 2:
        d_col = max_col - d_col
    return np.sqrt(d_row**2 + d_col**2)

def toroidal_mean(coords, max_row, max_col):
    """
    Compute the mean (row, col) on a toroidal (periodic) grid.
    Handles wraparound when averaging coordinates.
    """
    if not coords:
        return None
    rows = np.array([c[0] for c in coords])
    cols = np.array([c[1] for c in coords])
    # Convert to complex numbers on the unit circle
    row_angles = rows / max_row * 2 * np.pi
    col_angles = cols / max_col * 2 * np.pi
    mean_row_angle = np.angle(np.mean(np.exp(1j * row_angles)))
    mean_col_angle = np.angle(np.mean(np.exp(1j * col_angles)))
    # Map back to 0–max range
    mean_row = (mean_row_angle % (2 * np.pi)) / (2 * np.pi) * max_row
    mean_col = (mean_col_angle % (2 * np.pi)) / (2 * np.pi) * max_col
    return (int(round(mean_row)) % max_row, int(round(mean_col)) % max_col)
```

Agent Detection
```
def find_plane_position(R, target_value):
    """Finds plane positions using 4x2 uniform block detection"""
    # Returns (row, col) coordinates or None
```
Bullet Detection
```
def find_bullet_positions(R, step_count=None, visualize=False):
    """Detects bullet positions with edge filtering and visualization"""
    # Returns list of bullet coordinates
```
Distance Calculations
```
def toroidal_distance(pos1, pos2, max_row, max_col):
    """Calculates wrap-around distance on toroidal grid"""
    # Returns Euclidean distance with periodic boundaries
```
## Visualization

The project includes comprehensive visualization tools:
- RGB Image Display: Shows full game state
- R-Channel Analysis: Displays processed red channel data
- Plane Position Tracking: Visualizes detected agent and enemy positions
- Bullet Detection Overlay: Shows identified bullets on game frame

## File Structure
```
├── Agent_and_Bullet_Detection_in_Combat_Plane_Oct_1_2025.ipynb
├── models/
│   └── ppo_agent_*_steps.zip          # Trained model checkpoints
└── outputs/
    └── combat_plane_ai_vs_random.mp4  # Generated gameplay videos
```
## Citation
```
@misc{combat_plane_agent_detection_2025,
  author = {Rahman, Md Mijanur and Kemp, Ayden},
  title = {Agent and Bullet Detection in Combat Plane},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mdmirah/Agent-and-Bullet-Detection-in-Combat-Plane}}
}
```
