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
    """
    Finds the average (row, col) position of all 4x2 uniform blocks of a given target_value in R.
    Parameters:
        R (ndarray): 2D array of R channel values with scoreboard regions removed.
        target_value (int): The value to search for (223 for first_0, 111 for second_0).
    Returns:
        (row, col): Tuple of integer coordinates (averaged) or None if not found.
    """
    positions = []

    for row in range(R.shape[0] - 3):      # Stop at row 252 to allow 4-row box
        for col in range(R.shape[1] - 1):  # Stop at col 158 to allow 2-col box
            submatrix = R[row:row+4, col:col+2]

            if np.all(submatrix == target_value):
                # Take the (3rd row, 2nd col) = (row + 2, col + 1)
                positions.append((row + 2, col + 1))

    if positions:
        return toroidal_mean(positions, R.shape[0], R.shape[1])
    else:
        return None
```
Bullet Detection
```
def find_bullet_positions(
    R,
    step_count=None,
    white_thresh_low=245,
    white_thresh_high=255,
    min_bullet_area=0.5,
    max_bullet_area=1,
    edge_margin=2,  # Parameter to ignore detections near edges
    visualize=False
):
    """
    Detect bullet positions in R-channel by excluding background, clouds, and edges.

    Parameters:
        R (np.ndarray): Red channel image.
        step_count (int): Optional debug step.
        white_thresh_low (int): Minimum R value to consider a bullet.
        white_thresh_high (int): Maximum R value (255).
        min_bullet_area (int): Minimum contour area to consider.
        max_bullet_area (int): Maximum contour area to consider.
        edge_margin (int): Number of pixels to exclude near the edge.
        visualize (bool): Whether to show debug visualizations.

    Returns:
        List of (x, y) bullet positions.
    """
    height, width = R.shape[:2]

    # Apply intensity filter to extract only very bright pixels
    bullet_mask = cv2.inRange(R, white_thresh_low, white_thresh_high)

    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    bullet_mask = cv2.morphologyEx(bullet_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(bullet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bullet_positions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_bullet_area <= area <= max_bullet_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Skip detections near the edge
                if (edge_margin <= cx < width - edge_margin) and (edge_margin <= cy < height - edge_margin):
                    bullet_positions.append((cx, cy))

    # Optional visualization
    if visualize:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        axs[0].imshow(R, cmap="gray")
        axs[0].set_title(f"R-Channel (Step {step_count})")

        axs[1].imshow(bullet_mask, cmap="gray")
        axs[1].set_title("Binary Mask (Thresholded)")

        R_copy = cv2.cvtColor(R, cv2.COLOR_GRAY2BGR)
        for (cx, cy) in bullet_positions:
            cv2.circle(R_copy, (cx, cy), 3, (0, 255, 0), -1)  # Green dot

        axs[2].imshow(R_copy)
        axs[2].set_title("Detected Bullets (Filtered)")

        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    # Text log
    if step_count is not None:
        if bullet_positions:
            print(f"[BULLET DETECTION] Step {step_count}: Found {len(bullet_positions)} bullet(s) → {bullet_positions}")
        else:
            print(f"[BULLET DETECTION] Step {step_count}: No bullets detected.")

    return bullet_positions
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
```
def visualize_rgb_image(rgb_image, step=None, title="RGB Image"):
    """
    Displays the full RGB image.

    Parameters:
        rgb_image (ndarray): A (H, W, 3) RGB image (in standard numpy uint8 format).
        step (int, optional): Optional step counter to annotate the plot title.
        title (str): Custom title for the plot.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_image)
    plot_title = f"{title} (Step {step})" if step is not None else title
    plt.title(plot_title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
```
- R-Channel Analysis: Displays processed red channel data
```
def visualize_r_channel(R, step=None, center=None):
    import matplotlib.pyplot as plt
    plt.imshow(R, cmap='gray')
    if center:
        plt.scatter(center[1], center[0], color='red', s=40, label='Agent')
        plt.legend()
    if step:
        plt.title(f"R-Channel at Step {step}")
    plt.show()
```
- Plane Position Tracking: Visualizes detected agent and enemy positions
```
def visualize_planes(R, step=None):
    """
    Visualizes the red channel and the detected positions of the two planes.

    Parameters:
        R (ndarray): 2D grayscale red channel image.
        step (int, optional): Step count to label the figure.
    """
    # Detect plane positions
    first_pos = find_plane_position(R, target_value=223)
    second_pos = find_plane_position(R, target_value=111)

    # Convert grayscale R to BGR for color annotations
    R_bgr = cv2.cvtColor(R, cv2.COLOR_GRAY2BGR)

    if first_pos:
        cv2.circle(R_bgr, (first_pos[1], first_pos[0]), 5, (255, 0, 0), -1)  # Blue for first_0
        cv2.putText(R_bgr, "1", (first_pos[1]+6, first_pos[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    if second_pos:
        cv2.circle(R_bgr, (second_pos[1], second_pos[0]), 5, (0, 0, 255), -1)  # Red for second_0
        cv2.putText(R_bgr, "2", (second_pos[1]+6, second_pos[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(R_bgr)
    title = f"Plane Positions (Step {step})" if step is not None else "Plane Positions"
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
```
- Bullet Detection Overlay: Shows identified bullets on game frame
```
# Subsection of previous bullet detection code...
"""
def find_bullet_positions(
    R,
    step_count=None,
    white_thresh_low=245,
    white_thresh_high=255,
    min_bullet_area=0.5,
    max_bullet_area=1,
    edge_margin=2,  # Parameter to ignore detections near edges
    visualize=False
):
    height, width = R.shape[:2]

    # Apply intensity filter to extract only very bright pixels
    bullet_mask = cv2.inRange(R, white_thresh_low, white_thresh_high)

    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    bullet_mask = cv2.morphologyEx(bullet_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(bullet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bullet_positions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_bullet_area <= area <= max_bullet_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Skip detections near the edge
                if (edge_margin <= cx < width - edge_margin) and (edge_margin <= cy < height - edge_margin):
                    bullet_positions.append((cx, cy))
"""
    # Optional visualization
    if visualize:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        axs[0].imshow(R, cmap="gray")
        axs[0].set_title(f"R-Channel (Step {step_count})")

        axs[1].imshow(bullet_mask, cmap="gray")
        axs[1].set_title("Binary Mask (Thresholded)")

        R_copy = cv2.cvtColor(R, cv2.COLOR_GRAY2BGR)
        for (cx, cy) in bullet_positions:
            cv2.circle(R_copy, (cx, cy), 3, (0, 255, 0), -1)  # Green dot

        axs[2].imshow(R_copy)
        axs[2].set_title("Detected Bullets (Filtered)")

        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()
"""
    # Text log
    if step_count is not None:
        if bullet_positions:
            print(f"[BULLET DETECTION] Step {step_count}: Found {len(bullet_positions)} bullet(s) → {bullet_positions}")
        else:
            print(f"[BULLET DETECTION] Step {step_count}: No bullets detected.")

    return bullet_positions
"""
```

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
