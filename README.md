# RL Project

This repository contains trained DQN and DRQN agents for MiniGrid DoorKey environments, plus scripts to run visual simulations.

## Repository Structure

```text
RL-project/
├── requirements.txt
├── dqn/
└── drqn/

## Requirements

- Python **3.13.11** is required.
- `pip` is required.

## Setup

1. Clone the repository:

```bash
git clone <REPO_URL>
cd RL-project
```

2. Verify Python version:

```bash
python3 --version
```

Expected output should show `Python 3.13.11`.

3. Create a separate virtual environment:

```bash
python3 -m venv .venv
```

4. Activate the virtual environment:

```bash
source .venv/bin/activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Simulations

Run each simulation from the folder that contains its `run.py` and model file.

### DQN (6x6)

```bash
cd dqn/6x6
python run.py
```

### DRQN (6x6)

```bash
cd drqn/6x6
python run.py
```

### DRQN (8x8)

```bash
cd drqn/8x8
python run.py
```

## Notes

- Keep the virtual environment activated while running scripts.
- If you want to switch between simulations from any nested folder, return to repo root first:

```bash
cd /path/to/RL-project
```
