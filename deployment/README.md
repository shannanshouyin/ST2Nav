# ST2Nav: Unitree G1 deployment

Real-world deployment of ST2Nav with two fixed RGBD cameras. The robot captures observations and executes routes; a separate GPU computer reconstructs the panorama and runs the policy.

## Layout and hardware

```text
deployment/
├── README.md
├── robot/             # G1 capture, motion, client and private configuration templates
└── server/            # stitching, inference, source overlays and offline checks
    ├── vendor/ST2Nav/
    ├── vendor/habitat-lab/
    └── weights/       # Checkpoint configuration and checksums
```

`server/vendor/ST2Nav/` contains the model code, graph builder, waypoint predictor and XLM-R tokenizer used by the inference server. Run the installation commands below from `deployment/`.

| Component | Configuration |
|---|---|
| Robot | Unitree G1, three-waist-joint normal locomotion, official high-level controller |
| Upper camera | Intel RealSense D435i, rigid chest mount, portrait orientation |
| Lower camera | Intel RealSense D435i, rigid abdominal mount, portrait orientation |
| Capture | Each: aligned RGB8/Z16, 1280 × 720 at 6 Hz; D435i USB 3, D435i USB 2 tested |
| Inference | NVIDIA CUDA GPU |

Keep the camera mounts and waist attitude consistent with the installation calibration.

## Installation

**Server — Python 3.10, Linux or Windows, compatible NVIDIA driver.** Create an environment outside the repository; select CUDA wheels supported by the GPU. The deployment environment uses PyTorch 2.8.0 / torchvision 0.23.0 with CUDA 12.9:

```bash
python3.10 -m venv "$HOME/.venvs/st2nav-server"
source "$HOME/.venvs/st2nav-server/bin/activate"
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r server/requirements.txt
python -m pip check
```

On Windows, use `py -3.10 -m venv` and the environment's `Scripts/Activate.ps1`; subsequent Python commands are the same. Configure the local model checkpoint and waypoint predictor paths and checksums using [checkpoint setup](server/weights/README.md).

**Robot — Ubuntu on G1, Python 3.10.** Use the vendor-supported locomotion service and SDK. Install RealSense/librealsense with Python bindings for the onboard architecture; when a wheel is available, run `python -m pip install "pyrealsense2>=2.54,<3"` in the robot environment. Otherwise build librealsense with Python bindings for the system Python and expose them through `--system-site-packages` (or set `CAMERA_PYTHON` to that interpreter). Camera bindings are installed separately from `robot/requirements.txt`.

```bash
python3 -m venv --system-site-packages "$HOME/.local/share/etpr1/robot-env"
source "$HOME/.local/share/etpr1/robot-env/bin/activate"
python -m pip install -r robot/requirements.txt
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git "$HOME/unitree_sdk2_python"
git -C "$HOME/unitree_sdk2_python" checkout 814556d15970dd2ecf1c9984e845ca02ab07e206
python -m pip install -e "$HOME/unitree_sdk2_python"
python -m pip check
python -c "import numpy, scipy, pyrealsense2, unitree_sdk2py, cyclonedds"
chmod +x robot/run.sh robot/plan.sh
cp robot/config.example.json robot/config.json
```

SDK2 uses CycloneDDS 0.10.2. If it cannot find a system library, build CycloneDDS `releases/0.10.x` using the SDK's installation instructions and set `CYCLONEDDS_HOME` to its install prefix before installing SDK2. Edit `robot/config.json` with the local network interface, server DNS name and camera serials. `ROBOT_PYTHON` overrides the launcher interpreter; `CAMERA_PYTHON` overrides the camera-worker interpreter. Complete [installation calibration](robot/CALIBRATION.md), then place identical private calibration files on both computers.

## Start inference and navigation

After configuring the checkpoint path and checksum, install OpenSSL and create new credentials once in the **server environment**:

```bash
cd server
python init_auth.py --hostname inference.example
python tests/check_package.py
python tests/check_storage.py
python server.py --host YOUR_BIND_HOST --port 18766 --model st2nav
```

Replace `inference.example` with a DNS name reachable from the robot; use that same name in `robot/config.json` and the certificate. `YOUR_BIND_HOST` is the server's local bind hostname. Credentials default to `~/.local/share/etpr1/`. Securely copy **only** `server.crt` and `token` to that directory on the robot; never copy `server.key` or commit credentials. The server handles one navigation session at a time. Run one model service at a time on an 8 GB GPU. ST2Nav uses port 18766; RPGNav uses 18767.

On the robot, with a verified calibration, clear route, correct locomotion mode and a nearby operator holding the remote:

```bash
cd robot
./run.sh check
./run.sh navigate --model st2nav --execute --human-clear-path --clear-stop
```

Enter the instruction when prompted, or add `--instruction "Go forward and stop beside the chair."`. `--execute` authorizes motion; `--human-clear-path` records the operator's inspection; `--clear-stop` explicitly arms a new task after a previous stop. Interrupt with Ctrl+C or `./run.sh stop`, and verify the physical stop with the remote available. Remote input also interrupts execution. The model's STOP ends the task. Faults, manual stops and decision limits are recorded separately.

## Observations and reconstruction

Each decision uses **12 headings spaced 30° apart × 2 cameras = 24 RGBD frames**. The robot stops at each heading, records aligned metric depth and time-matched body pose, and returns to its initial heading. Intrinsics and calibrated camera-to-body transforms project frames into a common panorama; bounded image alignment, exposure compensation and shared seam selection reduce overlaps and seams. Twelve virtual **90° × 90°** views are rendered in 30° counterclockwise order, relative to the scan-end pose. Model inputs are RGB **224 × 224** and axial depth **256 × 256**, clipped to 10 m and normalized to [0,1].

Valid measured depths, including values beyond 10 m, are retained in the metric output. Missing values are estimated from neighboring inverse depth and measured support. Measurement and estimate masks remain separate; obstacle checks use measured depth. The waypoint predictor uses depth, while RGB and instruction features guide the navigation decision.

## Outputs and offline reproduction

The model returns `status`, `path_xy_m`, candidate waypoints, action probabilities and the selected node. World XY is expressed in the robot's odometry frame; the controller computes each target relative to the latest measured pose using forward/left coordinates and counterclockwise yaw.

Experiment records are persistent and stored outside the repository at **`~/navigation_results/st2nav/`** by default. The robot writes `robot/episodes/`, `robot/dual-scans/`, `robot/scans/`, `robot/navigation/` and `robot/plans/`; the inference computer writes `server/incoming-dual/` (or `server/incoming/` for legacy capture) and `server/plans/`. Each computer saves its own files. Reconstructed scans include `model-input.npz`, `depth-meters.npz`, images and `reconstruction.json`.

Set `NAV_RESULTS_DIR` to an absolute directory on **each computer** to change the experiment root; `robot/` and `server/` are appended automatically. For example, `export NAV_RESULTS_DIR="$HOME/navigation_results/st2nav-experiment"` (PowerShell: `$env:NAV_RESULTS_DIR = "$HOME/navigation_results/st2nav-experiment"`). Alternatively, pass `--results-dir /path/to/experiment` to `./run.sh navigate` or `python server.py`; this takes precedence and is inherited by robot child processes.

Library caches are stored in `~/.cache/etpr1/`; credentials and session state are stored in `~/.local/share/etpr1/`.

For a stationary, disconnected replay of one saved input:

```bash
cd server
python replay.py --input /path/to/model-input.npz --instruction "Go forward." --output /path/to/result.json
```

`--pose X Y YAW_RAD` sets the scan reference pose; the default is a local zero pose. See [checks](server/VALIDATION.md) for the available verification commands and [third-party notices](server/THIRD_PARTY.md) for source attribution.
