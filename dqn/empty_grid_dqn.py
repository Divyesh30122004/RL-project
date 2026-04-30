
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time
from minigrid.wrappers import ImgObsWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# DQN Architecture
# -----------------------
class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        x = x.float() / 255.0
        x = x.permute(0, 3, 1, 2)   # NHWC -> NCHW
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


# -----------------------
# Manual Random Spawn
# -----------------------
def randomize_agent_position(env):
    grid = env.unwrapped.grid
    width = env.unwrapped.width
    height = env.unwrapped.height

    while True:
        x = np.random.randint(1, width - 1)
        y = np.random.randint(1, height - 1)

        # Only place on empty cell
        if grid.get(x, y) is None:
            env.unwrapped.agent_pos = np.array([x, y])
            env.unwrapped.agent_dir = np.random.randint(0, 4)
            break


# -----------------------
# Environment
# -----------------------
env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="human")
env = ImgObsWrapper(env)

n_actions = env.action_space.n

# -----------------------
# Load Model
# -----------------------
policy_net = DQN(n_actions).to(device)
policy_net.load_state_dict(torch.load("dqn_minigrid_8x8.pth", map_location=device))
policy_net.eval()

# -----------------------
# Run Episodes
# -----------------------
NUM_EPISODES = 5

for episode in range(NUM_EPISODES):

    obs, _ = env.reset()

    # ---- Manually randomize spawn ----

    # Generate correct wrapped observation
    raw_obs = env.unwrapped.gen_obs()
    obs = env.observation(raw_obs)

    done = False
    total_reward = 0
    steps = 0

    print(f"\nEpisode {episode}")

    while not done:
        state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = policy_net(state)
            action = q_values.argmax(dim=1).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        steps += 1

        time.sleep(0.15)

    print(f"Finished in {steps} steps | Reward: {total_reward}")
    time.sleep(1)

env.close()
