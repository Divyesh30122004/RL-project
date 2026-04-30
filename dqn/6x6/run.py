import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from minigrid.wrappers import ImgObsWrapper

# --- 1. Re-define the DQN Architecture (Must match exactly) ---
class DQN(nn.Module):
    def __init__(self, n_actions, hidden_size=128):
        super(DQN, self).__init__()
        
        # Convolutional layers (same as training)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 7x7 -> 5x5 -> 3x3
        self.feature_size = 32 * 3 * 3  # 288 features
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)
    
    def forward(self, x):
        # x: (batch, H, W, C) - single state, no sequences
        batch_size = x.shape[0]
        
        # Normalize and reshape for conv layers
        x = x.float() / 10.0
        x = x.permute(0, 3, 1, 2)  # NCHW
        
        # Conv layers
        x = self.conv(x)
        x = x.reshape(batch_size, -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        
        return q_values

# --- 2. Setup Environment and Load Model ---
env = gym.make("MiniGrid-DoorKey-6x6-v0", render_mode="human")
env = ImgObsWrapper(env)

n_actions = env.action_space.n
model = DQN(n_actions, hidden_size=128)
model.load_state_dict(torch.load("minigrid_dqn_perfect_6x6.pth", map_location=torch.device('cpu')))
model.eval()

# --- 3. Visualization Loop ---
def watch_agent(episodes=5):
    for ep in range(episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_steps = 0
        print(f"Starting Episode {ep+1}")
        
        while not (terminated or truncated):
            # Prepare observation (no hidden state needed)
            state_tensor = torch.tensor(obs).unsqueeze(0)
            
            with torch.no_grad():
                q_values = model(state_tensor)
                action = q_values.argmax(dim=1).item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()
            
            time.sleep(0.1)  # Slow down so you can see the moves
            total_steps += 1
            
        if reward > 0:
            print(f"Goal Reached in {total_steps} steps!")
        else:
            print("Mission Failed.")
        time.sleep(1)

if __name__ == "__main__":
    watch_agent()
    env.close()
