# LabOS Robot Runtime

LabOS Robot Runtime is a YAML-driven robotics runtime for UFactory xArm systems. It combines recorded robot locations, RealSense camera calibration, vision-guided object motion, and a remote LabOS MCP server so agents can run protocols on a real bench robot or in mock mode.

## Quickstart

Use mock mode first if you want to verify the install before connecting robot hardware:

```bash
git clone https://github.com/cong-lab/LabOS-Robot-Runtime.git
cd LabOS-Robot-Runtime
pip install -r requirements.txt
cp .env.example .env
# Fill in LABOS_URL, LABOS_API_KEY, and LABOS_DEVICE_ID in .env.
./run.sh --mock  # --mock => do not connect to robot and headless/no vision processing
```

## Install

Run commands from the repository root. The runtime expects Python 3.10 or newer, `git`, and `pip`.

The normal install path is:

```bash
git clone https://github.com/cong-lab/LabOS-Robot-Runtime.git
cd LabOS-Robot-Runtime
git lfs pull
python -m pip install -r requirements.txt
```

The segmentation model at `weights/robot-segmentation.pt` is stored with Git LFS. If that file is missing or looks like a small text pointer after cloning, install Git LFS and fetch the model:

```bash
git lfs install
git lfs pull
```

`requirements.txt` installs the LabOS Python API directly from GitHub:

```text
labos-tool-client @ git+https://github.com/cong-lab/LabOS-Python-API.git
```

If you want to work on the LabOS SDK locally, clone it into `deps/` and install it in editable mode after installing the runtime requirements:

```bash
git clone https://github.com/cong-lab/LabOS-Python-API.git deps/LabOS-Python-API
python -m pip install -e ./deps/LabOS-Python-API
```

Some vision workflows may also need GPU-specific PyTorch or CUDA packages depending on the machine. Install those according to the camera workstation's environment before running training or model-heavy vision scripts.

## Hardware Setup

The runtime is intended for a single UFactory xArm or a bimanual xArm setup. If you are testing without hardware, skip to [Mock Mode](#mock-mode-optional).

Configure robot IPs and end effectors in `configs/robot_mapping.json`. The current layout uses:

- `right`: xArm at `192.168.0.2`, RealSense camera enabled, xArm gripper end effector.
- `left`: xArm at `192.168.0.3`, no camera, linear rail enabled, ZWHAND DM17 end effector.

Mount the Intel RealSense camera on the xArm camera plate for the arm marked with `has_camera: true`. Keep the `camera_device`, calibration file paths, and `tare` path in `configs/robot_mapping.json` aligned with the camera you actually use.

For ZWHAND DM17 setup, scanning, calibration, and replay details, see `docs/zwhand.md`.

## Calibration

Calibrate vision before running vision-guided protocols. The normal order is:

1. Capture checkerboard images for RealSense intrinsics.
2. Solve camera intrinsics.
3. Collect hand-eye samples with an ArUco marker.
4. Solve hand-eye calibration.
5. Confirm runtime config points to the generated calibration files.
The full calibration guide is in `docs/calibration.md`

## Protocols: Record And Run

Protocols live in `protocols/` as YAML files. They are built from declarative steps such as recorded `go_to` movements, end-effector commands, sleeps, sub-protocol calls, `parallel` blocks, and `repeat` loops. Recorded robot poses live in `locations/`.

Common recording and replay entry points are:

- `record_location.py`: record robot poses into `locations/`.
- `record_hand.py`: record ZWHAND or end-effector positions.
- `goto.py`: replay a recorded robot location.
- `gotoee.py`: replay a recorded end-effector location.

Run a YAML protocol from the command line:

```bash
python run_protocol.py vortexing
```

Vision-guided steps such as `move_to_object`, `wait_until_visible`, `handoff`, and `qr_align` let a protocol adapt to detected objects instead of relying only on fixed recorded positions. Object presets are defined in `configs/objects.yaml`.

For the full recording workflow, location format, YAML step reference, and protocol examples, see `docs/protocols.md`.

## Run The MCP Server

The MCP server connects this runtime to LabOS over WebSocket. Once connected, a LabOS agent can call tools such as `get_protocols`, `describe_protocol`, `start_protocol`, `stop_robot`, `get_status`, `move_to_object`, and other robot or vision helpers.

First create a machine definition in LabOS:

1. Navigate to https://labos.stella-agent.com and log in.
2. Click **Machines**.
3. Click **Add Machine**.
4. Label it something useful, such as `Mock Robot` or `Bench Robot`.
5. Copy the machine URL, API key, and device ID into `.env`:

```text
LABOS_URL=wss://...
LABOS_API_KEY=...
LABOS_DEVICE_ID=...
```

Then launch the server:

```bash
./run.sh
./run.sh --headless
```

By default, the real server opens the vision display when camera support is active. Use `--headless` or set `HEADLESS=1` to disable the OpenCV window.

The server fails fast if `LABOS_URL` or `LABOS_API_KEY` are missing from `.env` or the environment. At startup it logs the resolved URL, device ID, and a masked API key so you can verify that the right LabOS machine is being used.

For the full MCP architecture, tool surface, progress events, fatal error flow, and agent usage examples, see `docs/mcp.md`.

## Mock Mode Optional

Mock mode is useful for install checks, agent integration tests, demos, and development without a robot or camera. However, this still requires defining a valid API key and url as described in [Run The MCP Server](#run-the-mcp-server).

Run:

```bash
./run.sh --mock
```

You can also run the mock entry point directly:

```bash
python run_mock_mcp.py
```

Mock mode sets `MOCK_MODE=1` and `HEADLESS=1`. Protocol steps are replaced with short sleeps, no xArm or RealSense hardware is opened, and no vision control is performed. The protocol runner still emits realistic progress updates, including `protocol.progress` events through the MCP server. Mock logs are written to `logs/run_mock_mcp.log`.

## Repository Layout

- `aira/`: protocol runner, robot wrapper, and vision helpers.
- `protocols/`: YAML protocols and reusable protocol steps.
- `locations/`: recorded robot and end-effector poses.
- `configs/`: robot mapping, calibration outputs, object presets, and vision config.
- `scripts/`: calibration, capture, training, scanning, and test utilities.
- `docs/`: detailed guides for protocols, calibration, MCP, and ZWHAND.
- `weights/`: local model weights used by vision workflows.
- `run_mcp.py`, `run_mock_mcp.py`, `run.sh`, `run.bat`: MCP server entry points.

## Troubleshooting

- `Missing required LabOS settings`: check `.env` or exported shell variables for `LABOS_URL` and `LABOS_API_KEY`.
- xArm SDK fatal error: the MCP server locks further robot commands. Restart the script after the robot is in a safe state.
- Camera not detected: check the RealSense connection, `camera_device` in `configs/robot_mapping.json`, and the calibration guide.
- Object movement misses: confirm `configs/objects.yaml`, the model weights in `weights/`, and the camera calibration files.

Further reading:

- `docs/protocols.md`: recording positions and writing YAML protocols.
- `docs/calibration.md`: RealSense intrinsics and hand-eye calibration.
- `docs/mcp.md`: MCP server setup, tools, progress, and fatal handling.
- `docs/zwhand.md`: ZWHAND DM17 setup and calibration.
