
import gymnasium as gym
import torch
import torch.nn as nn
import time
from minigrid.wrappers import ImgObsWrapper

# --- 1. Re-define the Architecture (Must match exactly) ---
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
        x = x.reshape(B * T, H, W, C).float() / 10.0 # Match your x/10 change
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(B, T, -1)
        lstm_out, hidden = self.lstm(x, hidden)
        q_values = self.fc(lstm_out)
        return q_values, hidden

# --- 2. Setup Environment and Load Model ---
env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="human")
env = ImgObsWrapper(env)

n_actions = env.action_space.n
model = DRQN(n_actions)
model.load_state_dict(torch.load("minigrid_drqn_perfect_8x8.pth", map_location=torch.device('cpu')))
model.eval()

# --- 3. Visualization Loop ---
def watch_agent(episodes=5):
    for ep in range(episodes):
        obs, _ = env.reset()
        hidden = None
        terminated = False
        truncated = False
        total_steps = 0

        print(f"Starting Episode {ep+1}")
        
        while not (terminated or truncated):
            # Prepare observation
            state_tensor = torch.tensor(obs).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                q_values, hidden = model(state_tensor, hidden)
                action = q_values[:, -1].argmax(dim=1).item()
            
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
