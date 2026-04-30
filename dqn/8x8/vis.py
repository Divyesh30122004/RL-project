import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from minigrid.wrappers import ImgObsWrapper
from minigrid.core.world_object import Key, Door, Goal

# --- 1. DQN Architecture ---
class DQN(nn.Module):
    def __init__(self, n_actions, hidden_size=128):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.feature_size = 32 * 3 * 3 
        self.fc1 = nn.Linear(self.feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.float() / 10.0
        x = x.permute(0, 3, 1, 2) 
        x = self.conv(x)
        x = x.reshape(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- 2. Helper Functions (Scanning & Normalization) ---
def scan_grid(env):
    key_pos = door_pos = goal_pos = None
    grid = env.unwrapped.grid
    for idx, obj in enumerate(grid.grid):
        if obj is None: continue
        x, y = idx % grid.width, idx // grid.width
        if isinstance(obj, Key): key_pos = (x, y)
        elif isinstance(obj, Door): door_pos = (x, y)
        elif isinstance(obj, Goal): goal_pos = (x, y)
    return key_pos, door_pos, goal_pos

def run_episode(env, model):
    obs, _ = env.reset()
    terminated = False
    truncated = False
    step = 0
    key_pos, door_pos, goal_pos = scan_grid(env)
    key_pickup_step = door_open_step = None
    has_key = door_opened = False
    trajectory = [tuple(env.unwrapped.agent_pos)]

    while not (terminated or truncated):
        state_tensor = torch.tensor(obs).unsqueeze(0) # DQN expects (B, H, W, C)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = q_values.argmax(dim=1).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        step += 1
        trajectory.append(tuple(env.unwrapped.agent_pos))

        if not has_key and env.unwrapped.carrying:
            has_key = True
            key_pickup_step = step
        if not door_opened and door_pos:
            door_obj = env.unwrapped.grid.get(*door_pos)
            if door_obj is None or (hasattr(door_obj, 'is_open') and door_obj.is_open):
                door_opened = True
                door_open_step = step

    return trajectory, key_pos, door_pos, goal_pos, key_pickup_step, door_open_step, (reward > 0), step

def normalize_trajectory(trajectory, key_pos):
    return [(x - key_pos[0], y - key_pos[1]) for (x, y) in trajectory]

def accumulate_heatmap(norm_trajectories, map_range=8):
    size = 2 * map_range + 1
    heatmap = np.zeros((size, size), dtype=np.float32)
    offset = map_range
    for traj in norm_trajectories:
        for (rx, ry) in traj:
            xi, yi = int(rx) + offset, int(ry) + offset
            if 0 <= xi < size and 0 <= yi < size:
                heatmap[yi, xi] += 1
    return heatmap, offset

def phase_trajectories(trajectory, k_step, d_step):
    p1 = trajectory[:k_step + 1] if k_step else trajectory
    p2 = trajectory[k_step:d_step + 1] if (k_step and d_step) else []
    p3 = trajectory[d_step:] if d_step else []
    return p1, p2, p3

def draw_heatmap(ax, heatmap, offset, title, cmap='hot', landmark_offsets=None):
    im = ax.imshow(heatmap, cmap=cmap, origin='upper', interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.plot(offset, offset, '*', color='gold', markersize=12, label='Key')
    if landmark_offsets:
        if landmark_offsets['door']:
            dx, dy = landmark_offsets['door']
            ax.plot(dx+offset, dy+offset, 's', color='saddlebrown', markersize=8, label='Door')
        if landmark_offsets['goal']:
            gx, gy = landmark_offsets['goal']
            ax.plot(gx+offset, gy+offset, '^', color='limegreen', markersize=8, label='Goal')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=6)

# --- 3. Main Logic ---
def main(n_episodes=1000):
    env = gym.make("MiniGrid-DoorKey-8x8-v0")
    env = ImgObsWrapper(env)
    model = DQN(env.action_space.n)
    model.load_state_dict(torch.load("minigrid_dqn_perfect_8x8.pth", map_location='cpu'))
    model.eval()

    all_norm_full, p1_list, p2_list, p3_list = [], [], [], []
    door_offs, goal_offs = [], []
    success_count = 0

    print(f"Analyzing {n_episodes} DQN episodes...")
    for ep in range(n_episodes):
        res = run_episode(env, model)
        traj, k_pos, d_pos, g_pos, k_step, d_step, success, n_steps = res

        if success and k_pos:
            success_count += 1
            norm = normalize_trajectory(traj, k_pos)
            all_norm_full.append(norm)
            p1, p2, p3 = phase_trajectories(norm, k_step, d_step)
            p1_list.append(p1)
            if p2: p2_list.append(p2)
            if p3: p3_list.append(p3)
            if d_pos: door_offs.append((d_pos[0]-k_pos[0], d_pos[1]-k_pos[1]))
            if g_pos: goal_offs.append((g_pos[0]-k_pos[0], g_pos[1]-k_pos[1]))

    # Plotting
    avg_l = {
        'door': tuple(np.mean(door_offs, axis=0).astype(int)) if door_offs else None,
        'goal': tuple(np.mean(goal_offs, axis=0).astype(int)) if goal_offs else None
    }
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    maps = [all_norm_full, p1_list, p2_list, p3_list]
    titles = ['Overall', 'P1: Start->Key', 'P2: Key->Door', 'P3: Door->Goal']
    cmaps = ['hot', 'Blues', 'Oranges', 'Greens']

    for i in range(4):
        h, off = accumulate_heatmap(maps[i])
        draw_heatmap(axes[i], h, off, titles[i], cmaps[i], avg_l)

    plt.suptitle(f"DQN Performance Analysis (8x8) - Success Rate: {success_count}/{n_episodes}")
    plt.tight_layout()
    plt.savefig("dqn_path_analysis.png")
    print("Analysis saved to dqn_path_analysis.png")

if __name__ == "__main__":
    main()
