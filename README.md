PUMA560 Trajectory Analysis (改进 DH 参数法)
=============================================

Overview
--------
This folder contains `PUMA560.py`, a self-contained script that:

- builds a Puma560 DH robot model (robust DH handling),
- computes forward/inverse kinematics for example Cartesian via-points,
- plans joint trajectories using segmented cubic splines and LSPB (hybrid),
- prints DH parameters and per-link 4x4 transforms, and
- saves static visualizations and animated GIFs for MATLAB-style simulation.

Requirements
------------
Install the Python packages listed in `requirements.txt`. The script uses `roboticstoolbox`, `spatialmath`, `numpy`, `scipy`, `matplotlib`, `imageio`, and `Pillow`.

Quick setup (Conda recommended)
-------------------------------
1. Create and activate environment:

```bash
conda create -n arms python=3.8 -y
conda activate arms
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) For MP4 animations on Linux install ffmpeg:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

Run
---
From this directory run:

```bash
python PUMA560.py
```

This will print DH and transform information to the console and save visualization files in the current directory.

Generated outputs
-----------------
The script produces the following files (examples):

- `arm_config_via0.png`, `arm_config_via1.png`, `arm_config_via2.png`, `arm_config_via3.png` — robot snapshots for each via-point.
- `cubic_ee_path.png`, `cubic_ee_xyz_vs_time.png` — end-effector path and X/Y/Z vs time for the cubic trajectory.
- `hybrid_ee_path.png`, `hybrid_ee_xyz_vs_time.png` — path and time plots for the LSPB hybrid trajectory.
- `cubic_traj.gif`, `hybrid_traj.gif` — MATLAB-style animated GIFs for the two trajectories.
- `transforms_via0.txt` / `transforms_via0.tex`, ... — per-via-point textual transforms and LaTeX `equation*` files for report inclusion.

Notes & tips
------------
- GIF frame-rate and resolution: the script computes `fps` from the time-step near the bottom of `PUMA560.py`. To change it, edit the `fps` assignment or the `figsize` / `dpi` in `save_animation_gif()` in [arm/PUMA560.py](arm/PUMA560.py).
- Older CSV files were produced during intermediate runs but the current script no longer writes CSVs. To remove any leftover CSVs run:

```bash
rm -f transforms_via*_T*.csv
```

- If you see font/glyph warnings from `matplotlib` while running (non-ASCII labels), they are harmless for the saved images.

Troubleshooting
---------------
- "ModuleNotFoundError": ensure you installed packages with `pip install -r requirements.txt` inside the `arms` environment.
- GIFs look jumpy: the LSPB implementation was updated; re-run `python PUMA560.py` after editing `fps` if you want a different temporal sampling.

Next steps
----------
- I can delete leftover CSV files for you, or merge all `transforms_via*.tex` into a single LaTeX file for your report—tell me which you prefer.

Contact
-------
Reply here with any requested changes (custom DH/MDH parameters, MP4 output, combined LaTeX), and I'll update the code.
