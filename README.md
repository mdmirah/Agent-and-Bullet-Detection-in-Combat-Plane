<h1 align="center">Agent and Bullet Detection in Combat: Plane</h1>

In this project Ayden Kemp [@aydensairplanes](https://github.com/aydensairplanes) and I have implemented a **multi-agent reinforcement learning system** for the Combat Plane environment using **PettingZoo** and **Stable-Baselines3**. This project augments the Combat:Plane environment by adding cutom functions for feature extraction including agent and bullet detection and target ranging with the goal of supporting the design of custom heuristics for offensive, defensive and hybrid agents. By using these custom functions, unique reward functions for **offensive**, **defensive**, and **hybrid** agents are created, tested and validated. We have seen our custom offensive and defensive agents both outperform (As per their designed heuristic) the baseline agent of the Combat: Plane environment. A **hybrid** agent has also been designed in the codebase which is to be tested in future work.

---

## Overview

The project trains intelligent agents to play the **Combat: Plane game (bi-plane version)** from the Atari environment. The agents use **PPO (Proximal Policy Optimization)** and incorporate **custom reward functions** to learn specific combat strategies.

---

## Features

### Core Components

- **Plane Detection:** Identifies agent and enemy positions using R-channel analysis (Blue and Red circles).
<img width="488" height="374" alt="image" src="https://github.com/user-attachments/assets/38472cc6-7ae6-4322-aa3d-4f904e37905b" />

- **Bullet Detection:** Computer vision-based bullet tracking with filtering (Green circles).
<img width="488" height="381" alt="image" src="https://github.com/user-attachments/assets/ed27f755-29fa-4b52-a3fa-aafe0f37c5f2" />

- **Toroidal Distance Calculations:** Handles wrap-around game mechanics for the Atari environment.
- **Reduced Action Space Mapping:** The default Combat: Plane environment from **PettingZoo** contains 18 discrete action choices. However, these only correspond to 6 unique actions (fly straight, fly clockwise, fly counterclockwise, fly straight while firing a projectile, fly clockwise while firing a projectile, fly counterclockwise while firing a projectile). A wrapper to reduce the action space to only the 6 unique actions was developed to streamline agent training.
- **Feature Extraction:** MLP-based agent training requires a 1D vector input as compared to the multidimensional image data provided to an agent undergoing CNN-based training. A pipeline for extracting key features for use as an input to MLP-based agents was developed.
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
Action Space Reduction
```
import gymnasium as gym
from pettingzoo.utils.wrappers import BaseWrapper

mapping = [0, 1, 2, 5, 10, 13]

class ReduceActionSpaceWrapper(BaseWrapper):
    """
    Maps a small discrete action space (0..K-1) to the original ALE action ids.
    """
    def __init__(self, env, mapping):
        super().__init__(env)
        self.mapping = list(mapping)
        self._reduced_space = gym.spaces.Discrete(len(self.mapping))

    def action_space(self, agent):
        return self._reduced_space

    def step(self, action):
        # PettingZoo expects None for dead steps
        if action is None:
            return super().step(None)
        # Map reduced action -> original ALE action
        return super().step(self.mapping[int(action)])
```

Feature Extraction
```
def wrapped_delta(a, b, size):
    """
    Signed shortest wrapped difference from a to b on periodic domain [0, size).
    Result is in [-size/2, size/2].
    """
    d = b - a
    if d > size / 2:
        d -= size
    elif d < -size / 2:
        d += size
    return d

def nearest_bullet_relative(bullets, self_pos, max_row, max_col):
    """
    Returns signed wrapped relative position (dr, dc) of nearest bullet to self.
    If no bullets, returns (0, 0, 0) where final 0 is a 'has_bullet' flag.
    """
    if self_pos is None or not bullets:
        return 0.0, 0.0, 0.0

    best = None
    best_dist = None

    for (bx, by) in bullets:
        # bullet positions are (x, y), planes are (row, col)
        # so convert bullet to (row, col) = (y, x)
        br, bc = by, bx

        dr = wrapped_delta(self_pos[0], br, max_row)
        dc = wrapped_delta(self_pos[1], bc, max_col)
        dist = np.sqrt(dr**2 + dc**2)

        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = (dr, dc)

    return best[0], best[1], 1.0

def extract_features_from_obs(observation, self_target=223, opp_target=111):
    """
    observation: raw RGB image, shape (256, 160, 3)

    Returns feature vector as np.float32
    """
    max_row = observation.shape[0]   # 256
    max_col = observation.shape[1]   # 160

    R = get_cleaned_R(observation)

    self_pos = find_plane_position(R, self_target)
    opp_pos = find_plane_position(R, opp_target)
    bullets = find_bullet_positions(R, step_count=None, visualize=False)

    # Defaults if detection fails
    if self_pos is None:
        self_pos = (0, 0)
        self_found = 0.0
    else:
        self_found = 1.0

    if opp_pos is None:
        opp_pos = (0, 0)
        opp_found = 0.0
    else:
        opp_found = 1.0

    # Absolute normalized positions
    self_r = self_pos[0] / max_row
    self_c = self_pos[1] / max_col
    opp_r = opp_pos[0] / max_row
    opp_c = opp_pos[1] / max_col

    # Wrapped relative position from self -> opponent
    dr = wrapped_delta(self_pos[0], opp_pos[0], max_row) / (max_row / 2)
    dc = wrapped_delta(self_pos[1], opp_pos[1], max_col) / (max_col / 2)

    # Toroidal distance normalized
    dist = toroidal_distance(self_pos, opp_pos, max_row, max_col)
    if dist is None:
        dist = 0.0
    max_dist = np.sqrt((max_row / 2)**2 + (max_col / 2)**2)
    dist = dist / max_dist

    # Nearest bullet relative to self
    bdr, bdc, has_bullet = nearest_bullet_relative(bullets, self_pos, max_row, max_col)
    bdr /= (max_row / 2)
    bdc /= (max_col / 2)

    bullet_count = min(len(bullets), 10) / 10.0

    features = np.array([
        self_r, self_c,
        opp_r, opp_c,
        dr, dc,
        dist,
        bdr, bdc,
        has_bullet,
        bullet_count,
        self_found,
        opp_found,
    ], dtype=np.float32)

    return features
```

Offensive Reward Wrapper
```
# Optional print statements for troubleshooting are presently commented out
class OffensiveRewardWrapper(BaseWrapper):
    def __init__(self, env, shoot_actions=None, reward_multiplier=5, shoot_bonus=0.1):
        super().__init__(env)
        self.shoot_actions = shoot_actions or {1, 10, 11, 12, 13, 14, 15, 16, 17}
        self.reward_multiplier = reward_multiplier
        self.shoot_bonus = shoot_bonus
        self.first_0_step_count = 0  # Counter for first_0 steps

    def step(self, action):
        agent = self.agent_selection
        super().step(action)

        if agent == "first_0" and agent in self.agents:
            self.first_0_step_count += 1  # Increment step count

            original = self.rewards[agent]
            if original > 0:
                self.rewards[agent] = original * self.reward_multiplier
                #print(f"[REWARD DEBUG] {agent} hit: reward {original:.3f} → {self.rewards[agent]:.3f}")

            if self.first_0_step_count % 5 == 0:
                observation = self.env.observe(agent)
                R = get_cleaned_R(observation)

                first_pos = find_plane_position(R, 223)
                second_pos = find_plane_position(R, 111)

                #print(f"[DEBUG] first_0 position: {first_pos}, second_0 position: {second_pos}")
                #print(f"[REWARD DEBUG] {agent} used action {action}")

                if action in self.shoot_actions:
                    self.rewards[agent] += self.shoot_bonus
                    #print(f"[REWARD DEBUG] {agent} used action {action} on step {self.first_0_step_count}, +{self.shoot_bonus:.3f} → total: {self.rewards[agent]:.3f}")

                # Proximity reward
                max_toroidal_dist = 151.6
                min_dist = 10
                max_score = 0.5

                dist = toroidal_distance(first_pos, second_pos, R.shape[0], R.shape[1])
                if dist is not None:
                    clamped_dist = max(min_dist, min(dist, max_toroidal_dist))
                    proximity_reward = max_score * (max_toroidal_dist - clamped_dist) / (max_toroidal_dist - min_dist)
                    self.rewards[agent] += proximity_reward
                    #print(f"[REWARD DEBUG] {agent} proximity distance {dist:.2f}, reward +{proximity_reward:.3f} → total: {self.rewards[agent]:.3f}")
                else:
                    #print(f"[REWARD DEBUG] {agent} proximity distance unknown (position obscured)")
                    pass

            if self.first_0_step_count % 5 == 0:
                print(f"[OFFFENSIVE] Step: {self.first_0_step_count}, Reward: {self.rewards[agent]}")
```

Defensive Reward Wrapper
```
class DefensiveRewardWrapper(BaseWrapper):
    def __init__(self, env, bullet_value=255, bullet_block_size=2):
        super().__init__(env)
        self.bullet_value = bullet_value
        self.bullet_block_size = bullet_block_size
        self.first_0_step_count = 0

    def step(self, action):
        agent = self.agent_selection
        super().step(action)

        if agent == "first_0" and agent in self.agents:

            self.first_0_step_count += 1
            action_label = self.action_labels.get(action, "Unknown")


            # Calculate rewards only every 10 steps
            if self.first_0_step_count % 10 != 0:
                return

            # Get red channel from observation
            observation = self.env.observe(agent)
            R = get_cleaned_R(observation)

            # Detect agent and opponent
            first_pos = find_plane_position(R, 223)
            second_pos = find_plane_position(R, 111)

            # # Visualize the observation
            #visualize_rgb_image(observation, self.first_0_step_count)
            #visualize_planes(R, self.first_0_step_count)

            # Detect all bullets (treat all as dangerous)
            bullets = find_bullet_positions(
            R,
            step_count=self.first_0_step_count,
            white_thresh_low=100,
            white_thresh_high=255,
            min_bullet_area=0.5,
            max_bullet_area=1.0,
            edge_margin=2,
            visualize=False
            )

            # print(f"[DEFENSIVE AGENT ACTION] Step {self.first_0_step_count}: Agent {agent} took action {action} ({action_label})")

            # Initialize reward components
            distance_reward = 0.0
            bullet_avoidance_reward = 0.0
            damage_penalty = 0.0
            firing_penalty = 0.0

            if first_pos:
                # Distance-from-enemy reward
                if second_pos:
                    max_dist = 151.6
                    dist = toroidal_distance(first_pos, second_pos, R.shape[0], R.shape[1])
                    if dist is not None:
                        distance_reward = 0.4 * (dist / max_dist)
                        self.rewards[agent] += distance_reward

                # Bullet avoidance reward
                if bullets:
                    bullet_distances = [
                        toroidal_distance(first_pos, bpos, R.shape[0], R.shape[1])
                        for bpos in bullets if bpos is not None
                    ]
                    bullet_distances = [d for d in bullet_distances if d is not None]
                    if bullet_distances:
                        closest = min(bullet_distances)
                        max_bullet_dist = 151.6
                        bullet_avoidance_reward = 0.6 * min(closest / max_bullet_dist, 1.0)
                        self.rewards[agent] += bullet_avoidance_reward

                    # Damage penalty for getting hit
                    hit_radius = 3
                    for bpos in bullets:
                        dist_to_self = toroidal_distance(bpos, first_pos, R.shape[0], R.shape[1])
                        if dist_to_self is not None and dist_to_self <= hit_radius:
                            damage_penalty = -0.6
                            self.rewards[agent] += damage_penalty
                            break

            # Penalty for firing action
            if action in {1, 10, 11, 12, 13, 14, 15, 16, 17}:
                firing_penalty = -0.4
                self.rewards[agent] += firing_penalty

            # Final reward report
            print(f"Step: {self.first_0_step_count}, "
              f"First Pos: {first_pos}, Second Pos: {second_pos}, "
              f"\nDistance Reward: {distance_reward:.4f}, "
              f"Bullet Avoidance Reward: {bullet_avoidance_reward:.4f}, "
              f"Damage Penalty: {damage_penalty:.4f}, "
              f"Firing Penalty: {firing_penalty:.4f}, "
              f"Total Reward: {self.rewards[agent]:.4f}\n")
```

Hybrid Reward Wrapper
```
class HybridRewardWrapper(BaseWrapper):
    def __init__(self, env, shoot_actions=None,
                 proximity_threshold=264, max_proximity_score=0.25,
                 hit_multiplier=0.5,
                 bullet_value=255, hit_radius=3,
                 bullet_proximity_threshold=15, bullet_proximity_penalty_scale=-0.3,
                 own_bullet_memory=5):
        super().__init__(env)
        self.env = env
        self.shoot_actions = shoot_actions or {1, 10, 11, 12, 13, 14, 15, 16, 17}
        self.proximity_threshold = proximity_threshold
        self.max_proximity_score = max_proximity_score
        self.hit_multiplier = hit_multiplier
        self.bullet_value = bullet_value
        self.hit_radius = hit_radius
        self.bullet_proximity_threshold = bullet_proximity_threshold
        self.bullet_proximity_penalty_scale = bullet_proximity_penalty_scale
        self.own_bullet_memory = own_bullet_memory
        self.first_0_step_count = 0
        self.own_bullet_positions = []

    def step(self, action):
        agent = self.agent_selection
        super().step(action)

        if agent != "first_0" or agent not in self.agents:
            return

        self.first_0_step_count += 1
        observation = self.env.observe(agent)
        R = get_cleaned_R(observation)

        # Detect plane positions
        first_pos = find_plane_position(R, 223)
        second_pos = find_plane_position(R, 111)

        # Visualize the observation
        visualize_rgb_image(observation, self.first_0_step_count)
        visualize_planes(R, self.first_0_step_count)

        # Detect all bullets
        bullets = find_bullet_positions(
            R,
            step_count=self.first_0_step_count,
            white_thresh_low=100,
            white_thresh_high=255,
            min_bullet_area=0.5,
            max_bullet_area=1.0,
            edge_margin=2,
            visualize=True
        )

        # Reward components
        proximity_reward = 0.0
        hit_reward = 0.0
        bullet_proximity_penalty = 0.0
        damage_penalty = 0.0

        # Track own bullet firing position
        if action in self.shoot_actions and first_pos:
            self.own_bullet_positions.append({
                "pos": first_pos,
                "step": self.first_0_step_count
            })

        # Remove expired own bullets
        self.own_bullet_positions = [
            b for b in self.own_bullet_positions
            if self.first_0_step_count - b["step"] <= self.own_bullet_memory
        ]

        # Filter out self bullets
        enemy_bullets = []
        for bpos in bullets:
            if all(toroidal_distance(bpos, ob["pos"], R.shape[0], R.shape[1]) > 5
                   for ob in self.own_bullet_positions):
                enemy_bullets.append(bpos)

        # Proximity reward (offensive)
        if first_pos and second_pos:
            dist = toroidal_distance(first_pos, second_pos, R.shape[0], R.shape[1])
            if dist is not None and dist < self.proximity_threshold:
                proximity_reward = self.max_proximity_score * (self.proximity_threshold - dist) / self.proximity_threshold
                self.rewards[agent] += proximity_reward

        # Hit reward (amplified)
        original = self.rewards[agent]
        if original > 0:
            hit_reward = original * self.hit_multiplier
            self.rewards[agent] += hit_reward

        # Bullet proximity penalty (enemy bullets only)
        if first_pos and enemy_bullets:
            distances = [toroidal_distance(first_pos, bpos, R.shape[0], R.shape[1]) for bpos in enemy_bullets]
            distances = [d for d in distances if d is not None]
            if distances:
                closest = min(distances)
                if closest <= self.bullet_proximity_threshold:
                    bullet_proximity_penalty = self.bullet_proximity_penalty_scale * (
                        (self.bullet_proximity_threshold - closest) / self.bullet_proximity_threshold
                    )
                    self.rewards[agent] += bullet_proximity_penalty

        # Damage penalty (enemy bullets only)
        if first_pos:
            for bpos in enemy_bullets:
                dist_to_self = toroidal_distance(first_pos, bpos, R.shape[0], R.shape[1])
                if dist_to_self is not None and dist_to_self <= self.hit_radius:
                    damage_penalty = -1.0
                    self.rewards[agent] += damage_penalty
                    break

        # Final reward breakdown
        action_label = self.action_labels.get(action, "Unknown")
        print(f"[HYBRID] Step {self.first_0_step_count}, Agent: {agent} took action {action} ({action_label})")
        print(f"→ First Pos: {first_pos}, Second Pos: {second_pos}")
        print(f"→ Proximity Reward: +{proximity_reward:.4f}")
        print(f"→ Hit Reward Multiplier: +{hit_reward:.4f}")
        print(f"→ Bullet Proximity Penalty: {bullet_proximity_penalty:.4f}")
        print(f"→ Damage Penalty: {damage_penalty:.4f}")
        print(f"→ Total Reward: {self.rewards[agent]:.4f}\n")
        print(f"[DEBUG] Own bullets tracked: {[ob['pos'] for ob in self.own_bullet_positions]}")
        print(f"[DEBUG] Enemy bullets: {enemy_bullets}")
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
# Subsection of previous bullet detection code, other sections commented out...
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
