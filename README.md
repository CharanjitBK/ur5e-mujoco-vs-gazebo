# UR5e in MuJoCo vs Gazebo: a controlled simulator comparison

The same UR5e arm, workcell, motions and success tests are run in **MuJoCo 3** and
**Gazebo Classic 11 (ROS 2 Humble)**. Three tasks, 20 seeded trials per task per simulator:

| Task | What it stresses |
|---|---|
| Pick-and-place | trajectory tracking, grasp friction |
| Peg-in-hole (3 mm clearance per side) | jamming and wall contact |
| Repetitive tapping (20 taps per trial) | repeated impact, contact force |

Metrics: success, placement/tip error, path length, tracking error, contact force,
real-time factor (RTF), CPU and RAM. See `REPORT.md` for results and caveats.

> **Read the limitations first.** The two arms use different controllers (Gazebo: velocity loop,
> MuJoCo: position actuators), Gazebo is capped at real time (RTF = 1), and CPU/RAM cover
> different process sets. Details in `REPORT.md`.

## Repository layout

```
common/                 trial, analysis and plotting scripts (+ waypoints_*.json)
mujoco_ur5/models/      MuJoCo scenes (UR5e from MuJoCo Menagerie, see Credits)
ur5_experiment/         ROS 2 package: gripper/peg xacro, worlds, controller YAML
make_gz_peg.py          generates the Gazebo peg-in-hole world + peg xacro
make_gz_tap.py          generates the Gazebo tapping world
setup_controllers.sh    spawns/switches controllers after every Gazebo launch
results/                figures and tables (small files only)
```

Large raw data (`*.npy`, contact logs) is git-ignored. Re-create it by running the trials.

## Requirements

- Ubuntu 22.04 (a Docker container works), ROS 2 Humble, Gazebo Classic 11
- `ros-humble-gazebo-ros-pkgs`, `ros-humble-gazebo-ros2-control`, `ros-humble-ros2-control`,
  `ros-humble-ros2-controllers`
- UR packages cloned into your workspace `src/` (not included here):
  `Universal_Robots_ROS2_Description`, `Universal_Robots_ROS2_Driver` (branch `humble`),
  `ur_msgs` (branch `humble-devel`), `Universal_Robots_ROS2_Gazebo_Simulation`
  (use the Humble branches; check each repo's README)
- Python 3.10 with `mujoco`, `numpy`, `psutil` for the trial scripts
- A venv for analysis (`pandas`, `scipy`, `matplotlib`, `mujoco`):
  ```bash
  python3 -m venv venv && venv/bin/pip install numpy pandas scipy matplotlib mujoco
  ```

**Paths:** scripts assume the project lives at `/root/ur5_project` (the Docker mount).
If you clone elsewhere, either mount/symlink it there or edit the `/root/ur5_project`
constants at the top of each script.

**Interpreters:** ROS scripts use system `python3` (they need `rclpy`); analysis scripts use
`venv/bin/python`. Mixing them gives a broken-pandas error.

## Setup

```bash
# 1. put ur5_experiment/ into <ws>/src/ next to the UR repos, then
cd <ws> && colcon build --symlink-install && source install/setup.bash

# 2. generate the generated worlds / xacro
python3 make_gz_peg.py      # keep H = 0.013 in the script for the 3 mm baseline
python3 make_gz_tap.py
colcon build --packages-select ur5_experiment --symlink-install && source install/setup.bash
```

## Running MuJoCo

```bash
cd common
python3 solve_ik.py                       # writes waypoints.json (pick-and-place)
python3 trials_mujoco.py 20               # pick-and-place

python3 make_peg_scene.py; python3 solve_ik_peg.py
python3 trials_mujoco_peg.py 20           # add `view` for a live window

python3 make_tap_scene.py; python3 solve_ik_tap.py; python3 solve_ik_tap_jitter.py
python3 trials_mujoco_tap.py 20           # add `view` for a live window
```

Results go to `data/raw/mujoco/`.

## Running Gazebo

Launch (Terminal 1). Only `description_file` and `world` change between tasks:

| Task | `description_file` | `world` |
|---|---|---|
| Pick-and-place | `ur5e_gripper.urdf.xacro` | `ur5e_experiment.world` |
| Peg-in-hole | `ur5e_peg.urdf.xacro` | `ur5e_peg.world` |
| Tapping | `ur5e_peg.urdf.xacro` | `ur5e_tap.world` |

```bash
pkill -9 gzserver; pkill -9 gzclient
ros2 launch ur_simulation_gazebo ur_sim_control.launch.py \
  ur_type:=ur5e launch_rviz:=false \
  runtime_config_package:=ur5_experiment \
  controllers_file:=ur5e_gripper_controllers.yaml \
  description_package:=ur5_experiment \
  description_file:=<DESCRIPTION_FILE> \
  world:=<PATH_TO>/ur5_experiment/worlds/<WORLD>
```

Then (Terminal 2), **after every relaunch**:

```bash
./setup_controllers.sh        # spawns finger + velocity controllers and switches the arm
ros2 control list_controllers # expect forward_velocity_controller + finger_effort_controller active
```

Run the trials:

```bash
python3 common/trials_gazebo.py 20          # pick-and-place (add `nosamp` to skip CPU sampler)
python3 common/trials_gazebo_peg.py 20
python3 common/trials_gazebo_tap.py 20 force   # `force` records contacts (~50 MB per trial)
```

Sanity check on any Gazebo run: the printed `above_hole tool0` should be near (0.5, 0.2, 0.60).
A frozen value such as (0.0008, 0.2329, 1.0794) means the arm did not move (controllers missing).

## Analysis and figures

```bash
V=venv/bin/python
$V common/analyze_fixed.py          # pick-and-place metrics (supersedes analyze.py)
$V common/plot.py; $V common/plot_tracking.py
$V common/analyze_peg.py; $V common/plot_peg_descent.py; $V common/plot_peg_tip.py
$V common/tap_force_all.py          # parses Gazebo contact logs
$V common/analyze_tap.py
```

Outputs: `data/processed/*.csv`, `results/*.csv`, `results/figures/report/`.

## Fairness design

- Same robot model geometry, masses, friction (0.5), timestep (0.002 s).
- Same joint waypoints and quintic trajectories, solved once with MuJoCo IK.
- Same seeded variations (seed = 42 + trial): cube start offset, hole offset, tap depth.
- Same success criteria in both simulators.

## Known issues

- After each Gazebo relaunch the extra controllers must be spawned again.
- `sequence size exceeds remaining buffer` (DDS) and `Moved backwards in time` warnings are harmless.
- Gazebo's arm needed the **velocity** command interface: the position interface moves links
  kinematically and friction cannot lift objects.
- Gazebo peg/wall surfaces must use `min_depth` 0; 0.001 widened the hole and gave a false 20/20.

## Credits and licences

- UR5e MuJoCo model: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
  (BSD-3-Clause), modified here. Keep its licence if you redistribute it.
- UR ROS 2 packages: Universal Robots / ROS-Industrial repositories (own licences).
- Code in this repository: MIT licence (see `LICENSE`).
