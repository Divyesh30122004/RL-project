import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from minigrid.wrappers import ImgObsWrapper
from minigrid.core.world_object import Key, Door, Goal


class DRQN(nn.Module):
    def __init__(self, n_actions, hidden_size=64):
        super(DRQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.feature_size = 32 * 3 * 3
        self.lstm = nn.LSTM(self.feature_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_actions)

    def forward(self, x, hidden=None):
        B, T, H, W, C = x.shape
        x = x.reshape(B * T, H, W, C).float() / 10.0
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(B, T, -1)
        lstm_out, hidden = self.lstm(x, hidden)
        q_values = self.fc(lstm_out)
        return q_values, hidden


def scan_grid(env):
    key_pos = door_pos = goal_pos = None
    grid = env.unwrapped.grid
    for idx, obj in enumerate(grid.grid):
        if obj is None:
            continue
        x = idx % grid.width
        y = idx // grid.width
        if isinstance(obj, Key):
            key_pos = (x, y)
        elif isinstance(obj, Door):
            door_pos = (x, y)
        elif isinstance(obj, Goal):
            goal_pos = (x, y)
    return key_pos, door_pos, goal_pos


def run_episode(env, model):
    obs, _ = env.reset()
    hidden = None
    terminated = False
    truncated = False
    step = 0

    key_pos, door_pos, goal_pos = scan_grid(env)
    key_pickup_step = None
    door_open_step = None
    has_key = False
    door_opened = False

    trajectory = [tuple(env.unwrapped.agent_pos)]

    while not (terminated or truncated):
        state_tensor = torch.tensor(obs).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            q_values, hidden = model(state_tensor, hidden)
            action = q_values[:, -1].argmax(dim=1).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        step += 1
        trajectory.append(tuple(env.unwrapped.agent_pos))

        if not has_key and env.unwrapped.carrying and isinstance(env.unwrapped.carrying, Key):
            has_key = True
            key_pickup_step = step

        if not door_opened and door_pos:
            door_obj = env.unwrapped.grid.get(*door_pos)
            if door_obj is None or (hasattr(door_obj, 'is_open') and door_obj.is_open):
                door_opened = True
                door_open_step = step

    success = reward > 0
    return trajectory, key_pos, door_pos, goal_pos, key_pickup_step, door_open_step, success, step


def normalize_trajectory(trajectory, key_pos):
    """Shift all positions so key is at origin."""
    kx, ky = key_pos
    return [(x - kx, y - ky) for (x, y) in trajectory]


def accumulate_heatmap(norm_trajectories, map_range=8):
    """
    Accumulate visit counts in key-centered space.
    map_range: grid extends this many cells in each direction from key.
    Returns 2D array and the offset used.
    """
    size = 2 * map_range + 1
    heatmap = np.zeros((size, size), dtype=np.float32)
    offset = map_range  # key sits at (offset, offset)

    for traj in norm_trajectories:
        for (rx, ry) in traj:
            xi = int(rx) + offset
            yi = int(ry) + offset
            if 0 <= xi < size and 0 <= yi < size:
                heatmap[yi, xi] += 1

    return heatmap, offset


def phase_trajectories(trajectory, key_pickup_step, door_open_step):
    """Split trajectory into 3 phases."""
    phase1 = trajectory[:key_pickup_step + 1] if key_pickup_step else trajectory
    phase2 = trajectory[key_pickup_step:door_open_step + 1] if (key_pickup_step and door_open_step) else []
    phase3 = trajectory[door_open_step:] if door_open_step else []
    return phase1, phase2, phase3


def draw_heatmap(ax, heatmap, offset, title, cmap='hot', landmark_offsets=None):
    """Draw a heatmap in key-centered space."""
    size = heatmap.shape[0]

    im = ax.imshow(heatmap, cmap=cmap, origin='upper', interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Visit count')

    # mark key at origin
    ax.plot(offset, offset, '*', color='gold', markersize=14, zorder=5, label='Key (origin)')

    # mark door and goal if provided
    if landmark_offsets:
        if landmark_offsets.get('door'):
            dx, dy = landmark_offsets['door']
            ax.plot(dx + offset, dy + offset, 's', color='saddlebrown',
                    markersize=10, zorder=5, label='Door (avg)')
        if landmark_offsets.get('goal'):
            gx, gy = landmark_offsets['goal']
            ax.plot(gx + offset, gy + offset, '^', color='limegreen',
                    markersize=10, zorder=5, label='Goal (avg)')

    # grid lines
    for i in range(size + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.3, alpha=0.5)
        ax.axvline(i - 0.5, color='gray', linewidth=0.3, alpha=0.5)

    # axis labels relative to key
    ticks = list(range(0, size, 2))
    ax.set_xticks(ticks)
    ax.set_xticklabels([t - offset for t in ticks], fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels([t - offset for t in ticks], fontsize=7)
    ax.set_xlabel('X offset from key', fontsize=8)
    ax.set_ylabel('Y offset from key', fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=6, loc='upper right')


def main(n_episodes=100):
    env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="rgb_array")
    env = ImgObsWrapper(env)
    n_actions = env.action_space.n

    model = DRQN(n_actions)
    model.load_state_dict(torch.load("minigrid_drqn_perfect_8x8.pth", map_location='cpu'))
    model.eval()

    # --- collect episodes ---
    all_norm_full = []
    all_norm_phase1 = []
    all_norm_phase2 = []
    all_norm_phase3 = []
    door_offsets = []
    goal_offsets = []
    success_count = 0
    best_traj = None
    best_steps = float('inf')
    best_meta = None

    print(f"Running {n_episodes} episodes...")
    for ep in range(n_episodes):
        traj, key_pos, door_pos, goal_pos, key_pickup_step, door_open_step, success, n_steps = run_episode(env, model)

        if not success or key_pos is None:
            continue

        success_count += 1
        norm = normalize_trajectory(traj, key_pos)
        all_norm_full.append(norm)

        p1, p2, p3 = phase_trajectories(norm, key_pickup_step, door_open_step)
        all_norm_phase1.append(p1)
        if p2: all_norm_phase2.append(p2)
        if p3: all_norm_phase3.append(p3)

        kx, ky = key_pos
        if door_pos:
            door_offsets.append((door_pos[0] - kx, door_pos[1] - ky))
        if goal_pos:
            goal_offsets.append((goal_pos[0] - kx, goal_pos[1] - ky))

        if n_steps < best_steps:
            best_steps = n_steps
            best_traj = norm
            best_meta = (key_pickup_step, door_open_step, success, n_steps)

        if ep % 10 == 0:
            print(f"  ep {ep}/{n_episodes} | successes so far: {success_count}")

    print(f"\nDone. Success rate: {success_count}/{n_episodes}")

    # average landmark offsets
    avg_door = tuple(np.mean(door_offsets, axis=0).astype(int)) if door_offsets else None
    avg_goal = tuple(np.mean(goal_offsets, axis=0).astype(int)) if goal_offsets else None
    landmark_offsets = {'door': avg_door, 'goal': avg_goal}

    MAP_RANGE = 8
    heatmap_full, offset = accumulate_heatmap(all_norm_full, MAP_RANGE)
    heatmap_p1,   _      = accumulate_heatmap(all_norm_phase1, MAP_RANGE)
    heatmap_p2,   _      = accumulate_heatmap(all_norm_phase2, MAP_RANGE)
    heatmap_p3,   _      = accumulate_heatmap(all_norm_phase3, MAP_RANGE)

    # --- plot ---
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    draw_heatmap(axes[0], heatmap_full, offset,
                 f'Overall Visit Frequency\n({success_count} successful eps)',
                 cmap='hot', landmark_offsets=landmark_offsets)

    draw_heatmap(axes[1], heatmap_p1, offset,
                 'Phase 1: Start → Key',
                 cmap='Blues', landmark_offsets=landmark_offsets)

    draw_heatmap(axes[2], heatmap_p2, offset,
                 'Phase 2: Key → Door',
                 cmap='Oranges', landmark_offsets=landmark_offsets)

    draw_heatmap(axes[3], heatmap_p3, offset,
                 'Phase 3: Door → Goal',
                 cmap='Greens', landmark_offsets=landmark_offsets)

    # overlay best trajectory on overall heatmap
    if best_traj:
        key_pickup_step, door_open_step, _, _ = best_meta
        xs = [p[0] + offset for p in best_traj]
        ys = [p[1] + offset for p in best_traj]
        for i in range(len(best_traj) - 1):
            if key_pickup_step and door_open_step and i >= door_open_step:
                color = 'cyan'
            elif key_pickup_step and i >= key_pickup_step:
                color = 'orange'
            else:
                color = 'white'
            axes[0].plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                         color=color, linewidth=1.5, alpha=0.8, zorder=6)
        axes[0].plot(xs[0], ys[0], 'o', color='white', markersize=6, zorder=7)
        axes[0].set_title(
            f'Overall Visit Frequency\n({success_count} eps) + best traj ({best_steps} steps)',
            fontsize=10)

    plt.suptitle("DRQN Trajectory Analysis — Key-Centered Heatmaps (MiniGrid DoorKey 8x8)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("drqn_heatmap_analysis.png", dpi=150, bbox_inches='tight')
    print("Saved: drqn_heatmap_analysis.png")
    env.close()


if __name__ == "__main__":
    main(n_episodes=1000)
