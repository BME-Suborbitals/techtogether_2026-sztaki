# techtogether_2026-sztaki

This repository includes a real-time demo script at `real_time.py`. The script runs a MuJoCo simulation in real-time. The simulation is based on the Unitree Go1 robot, which follows the movement of the user real-time through the camera.

The steps below focus on launching that script inside a Python virtual environment.

## Prerequisites

- Python 3.10+ installed
- Git (optional, if you are cloning the repo)

## Quick start (virtual environment)

### 1) Create and activate a venv

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux (bash)**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see your prompt change to include `.venv`.

### 2) Install dependencies

The dependencies live in the simulation package folder. Install them from the repo root:

```bash
python -m pip install --upgrade pip
python -m pip install -r unitree-g1-mujoco/requirements.txt
```

### 3) Run the real-time demo

From the repo root:

```bash
python real_time.py
```
