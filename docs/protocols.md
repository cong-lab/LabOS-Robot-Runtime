# Protocols and Recorded Positions

This repository runs robot procedures from YAML files through `run_protocol.py` and `aira.protocol_runner`. Most protocols are built from recorded locations in `locations/`: teach the robot a pose once, verify it, then reference that location from YAML.

The usual workflow is:

1. Confirm the arm and end-effector are configured in `configs/robot_mapping.json`.
2. Record a location with `record_location.py` or `record_hand.py`.
3. Verify the saved pose slowly with `goto.py` or `gotoee.py`.
4. Use the location from a YAML protocol with `go_to`, `hand_position`, or `grip`.
5. Rerun the protocol and rerecord or offset locations as needed.

## Quick Start

Run a protocol by name:

```bash
python run_protocol.py vortexing
python run_protocol.py vortex_handoff
python run_protocol.py plate_test
python run_protocol.py miniprep
python run_protocol.py repeater_full
python run_protocol.py transfection
python run_protocol.py transformation
```

Useful flags:

```bash
python run_protocol.py vortexing --no-wait
python run_protocol.py vortexing --vision
python run_protocol.py --status
```

Protocol names resolve under `protocols/`. For example, `python run_protocol.py vortexing` loads `protocols/vortexing.yaml`. Sub-protocols can be run directly by path, such as `steps/rack/pick_up_from_rack`, but top-level user-facing protocols should live at `protocols/<name>.yaml`.

## Robot Configuration

Robot names, IP addresses, cameras, linear rails, and end-effectors are defined in `configs/robot_mapping.json`. Recording and replay commands use that mapping unless an `--ip` override is provided.

The current mapping uses:

- `right`: xArm gripper with camera.
- `left`: DM17 dexterous hand on a linear rail.

Use the configured arm names consistently in commands and YAML:

```bash
python record_location.py rack_watch --arm right
python record_location.py hand/repeater-grip --arm left --visual
```

## Recording Locations

Use `record_location.py` for normal robot poses. It puts the selected arm or arms into manual mode, waits for you to physically position them, restores position mode, reads the current pose and joint angles, then writes `locations/<name>.json`.

Record one arm:

```bash
python record_location.py rack_watch --arm right
```

Record both configured arms at once:

```bash
python record_location.py shared_handoff
```

`--arm both` is the default. Both arms enter manual mode together, and the saved JSON contains one sub-entry per arm.

Useful options:

- `--arm left`, `--arm right`, or `--arm both`: choose what to record.
- `--zdown`: run `z_down()` before saving so the tool Z axis points down in base coordinates.
- `--visual`: open the 3D hand editor for dexterous end-effectors before saving the end-effector state.
- `--port 8080`: choose the viser port for `--visual`.
- `--ip 192.168.0.2`: override the IP from `configs/robot_mapping.json`.

For arms on a linear rail, `record_location.py` also saves `linear_rail_mm`.

## Location Files

Locations live under `locations/`. The location name is the path without `.json`, so `locations/hand/repeater-grip.json` is referenced as `hand/repeater-grip`.

A single-arm location looks like this:

```json
{
  "pose": [119.233475, 146.701553, 435.516724, -179.998301, -0.001833, -3.520482],
  "position_mm": [119.233475, 146.701553, 435.516724],
  "orientation_deg": [-179.998301, -0.001833, -3.520482],
  "joint_angles_deg": [50.897904, -47.153453, -65.77693, -0.000573, 112.927919, 54.418156, 0.0],
  "arm": "right"
}
```

Fields:

- `pose`: `[x, y, z, roll, pitch, yaw]` in millimeters and degrees.
- `position_mm`: first three values from `pose`.
- `orientation_deg`: last three values from `pose`.
- `joint_angles_deg`: recorded xArm joint angles. When present and no offset is used, replay uses joint-space motion.
- `arm`: target arm name, usually `left` or `right`.
- `linear_rail_mm`: optional rail target for arms configured with `on_linear_rail: true`.
- `end_effector`: optional recorded gripper or dexterous hand state.

A bimanual location has `arm: both` and nested per-arm entries:

```json
{
  "arm": "both",
  "left": {
    "pose": [-80.0, -240.0, 340.0, 178.0, -19.0, -102.0],
    "position_mm": [-80.0, -240.0, 340.0],
    "orientation_deg": [178.0, -19.0, -102.0],
    "joint_angles_deg": [-108.7, -63.8, -35.3, 0.2, 80.0, -6.2, 0.0],
    "linear_rail_mm": 120.0
  },
  "right": {
    "pose": [111.0, 290.1, 165.4, -179.9, -0.0, -1.9],
    "position_mm": [111.0, 290.1, 165.4],
    "orientation_deg": [-179.9, -0.0, -1.9],
    "joint_angles_deg": [69.0, -31.0, -29.0, 0.0, 60.0, 71.0, 0.0]
  }
}
```

An end-effector location can include the arm pose plus an `end_effector` block:

```json
{
  "arm": "left",
  "end_effector": {
    "type": "zw-dm17",
    "angles": [61, 754, 591, 1000, 743, 0, 1000, 978, 0, 1000, 886, 0, 713, 112, 214, 112, 1000]
  }
}
```

End-effector formats:

- `{"type": "xarm-gripper2", "position": 450.0}` for the xArm gripper. Position is roughly `0` closed to `850` open.
- `{"type": "zw-dm17", "angles": [...]}` for the DM17 dexterous hand.

## Replaying Locations From the CLI

Use `goto.py` to move an arm to a saved location:

```bash
python goto.py rack_watch --arm right --speed 150 --acc 300
```

Use low speed while testing a newly recorded location:

```bash
python goto.py rack_watch --arm right --speed 50 --acc 200
```

Use `--offset` for a cartesian offset from the saved pose:

```bash
python goto.py rack_watch --arm right --offset 0 0 30
python goto.py rack_watch --arm right --offset 0 0 30 0 0 10
```

Offsets are `[dx, dy, dz]` or `[dx, dy, dz, droll, dpitch, dyaw]` in millimeters and degrees. Any non-null offset forces cartesian replay.

Use `--zdown` to reorient the tool after arrival:

```bash
python goto.py rack_watch --arm right --zdown
```

## Joint-Mode vs Cartesian-Mode Replay

`go_to` can replay a location in joint mode or cartesian mode:

- Default replay: if the location has `joint_angles_deg` and no `offset`, replay uses joint-space motion with `set_servo_angle`.
- Offset replay: if `offset` is provided, replay uses cartesian motion with `move_to_absolute`; joint angles are ignored and the offset is added to `pose`.
- Pose-only replay: if the location has no `joint_angles_deg`, replay uses cartesian motion with `pose`.

Prefer default joint-mode replay for taught positions because it follows the recorded joint configuration. Use `offset` for small repeatable changes, such as moving above a taught pickup point:

```yaml
- step: go_to
  arm: right
  location: miniprep_rack
  offset: [0, 0, 30]
  speed: 100
  acc: 300
```

Do not expect joint mode and offset to combine. In both `goto.py` and YAML, any `offset` switches `go_to` to cartesian mode.

## Bimanual and Linear-Rail Replay

If a location file has `"arm": "both"` and the YAML step does not specify `arm`, both saved arm entries move. They move in parallel by default:

```yaml
- step: go_to
  location: shared_handoff
```

Set `parallel: false` to move bimanual entries sequentially:

```yaml
- step: go_to
  location: shared_handoff
  parallel: false
```

If the step specifies one arm, only that arm's sub-entry is used:

```yaml
- step: go_to
  arm: left
  location: shared_handoff
```

For linear-rail arms, replay starts the rail move from `linear_rail_mm` before the arm move, so the rail and arm can move together.

## Recording and Replaying End-Effectors

`record_location.py` can record end-effector state while recording an arm pose. For dexterous hands, add `--visual` to use the browser-based editor:

```bash
python record_location.py hand/repeater-grip --arm left --visual
```

Use `record_hand.py` when you mainly want to author a hand pose:

```bash
python record_hand.py --arm left locations/hand/pipette-grip.json
python record_hand.py --arm left --visual locations/hand/pipette-grip.json
python record_hand.py --arm left --visual --start locations/hand/repeater-grip.json locations/hand/pipette-grip.json
```

Use `gotoee.py` to move only the end-effector:

```bash
python gotoee.py hand/repeater-grip --arm left
```

Update only the end-effector field in an existing location:

```bash
python gotoee.py hand/repeater-grip --arm left --save
python gotoee.py hand/repeater-grip --arm left --visual
```

In YAML, use `hand_position` for dexterous hands:

```yaml
- step: hand_position
  arm: left
  location: hand/repeater-grip
```

For the xArm gripper, use `grip`:

```yaml
- step: grip
  arm: right
  state: 200
```

## Creating YAML From Recorded Locations

A protocol file usually has these top-level keys:

```yaml
brief: "Human-readable summary"

args:
  travel_speed: 800
  travel_acc: 1000

protocol:
  - step: go_to
    location: home
    description: Start at home

failure:
  - step: grip
    state: 800
  - step: go_to
    location: home
```

`brief` is shown by protocol listing tools. `args` contains defaults. `protocol` is the ordered list of steps. `failure` is an optional cleanup sequence that runs if a step fails or the protocol is stopped.

Minimal recorded-location protocol:

```yaml
brief: "Pick up a tube from a recorded rack position"

args:
  travel_speed: 100
  travel_acc: 300

protocol:
  - step: go_to
    arm: right
    location: rack_watch
    offset: [0, 0, 30]
    speed: "{{travel_speed}}"
    acc: "{{travel_acc}}"
    description: Move above the recorded rack position

  - step: go_to
    arm: right
    location: rack_watch
    speed: "{{travel_speed}}"
    acc: "{{travel_acc}}"
    description: Move to the recorded rack position

  - step: grip
    arm: right
    state: 200
    description: Close gripper on the tube

  - step: go_to
    arm: right
    location: rack_watch
    offset: [0, 0, 30]
    speed: "{{travel_speed}}"
    acc: "{{travel_acc}}"
    description: Lift back above the rack

failure:
  - step: grip
    arm: right
    state: 800
  - step: go_to
    arm: right
    location: home
```

Bimanual YAML example:

```yaml
protocol:
  - step: go_to
    location: shared_handoff
    description: Move both arms to the recorded handoff pose

  - step: parallel
    branches:
      - arm: left
        steps:
          - step: hand_position
            location: hand/repeater-grip
      - arm: right
        steps:
          - step: grip
            state: 200
```

## Template Variables

Use `{{name}}` to reference values from `args` or values created by `repeat` and `random_choice`.

```yaml
args:
  travel_speed: 900

protocol:
  - step: go_to
    location: miniprep_vortexer
    speed: "{{travel_speed}}"
```

When a field is exactly `"{{name}}"`, the original type is preserved. This is why list-valued offsets work:

```yaml
args:
  hover_offset: [0, 0, 30]

protocol:
  - step: go_to
    location: miniprep_rack
    offset: "{{hover_offset}}"
```

When `{{name}}` appears inside a larger string, it becomes text. This is useful for generated location names:

```yaml
- step: random_choice
  var: roll_suffix
  choices: ['', '_1', '_2']
- step: go_to
  location: "trans_left_agar_plate_roll{{roll_suffix}}"
```

## Rerecording and Refining Positions

Rerun `record_location.py` with the same name to overwrite that location:

```bash
python record_location.py rack_watch --arm right
```

For small position changes, prefer a cartesian offset over editing JSON by hand:

```bash
python goto.py rack_watch --arm right --offset 0 0 5 --speed 50 --acc 200
```

Then move the offset into YAML:

```yaml
- step: go_to
  arm: right
  location: rack_watch
  offset: [0, 0, 5]
```

Rerecord the location when the offset becomes large, when the orientation changes, or when the joint configuration matters. Use `gotoee.py --save` or `gotoee.py --visual` to update only an end-effector state without changing the saved arm pose.

## Common Step Fields

Most steps support these fields:

- `step`: required operation name.
- `description`: optional short status text.
- `major_description`: optional higher-level status text for top-level protocol progress.
- `arm`: optional target arm, usually `left` or `right`. If omitted, the default arm is used, except bimanual saved locations can move both arms.
- `speed` and `acc`: motion speed and acceleration, when supported by the underlying robot command.

## Advanced YAML Step Reference

The sections below list the supported YAML step types. The most common recorded-location steps are `go_to`, `grip`, and `hand_position`.

### `load_home`

Loads a reference frame file.

```yaml
- step: load_home
  file: home.json
```

Fields: `file`, defaulting to `home.json`.

### `home`

Moves to the loaded home reference frame.

```yaml
- step: home
```

### `set_tcp_limits`

Sets Cartesian TCP motion limits.

```yaml
- step: set_tcp_limits
  arm: right
  max_speed: 400
  max_acc: 800
  jerk: 1300
```

Fields: `max_speed`, `max_acc`, `jerk`. All are optional.

### `set_joint_limits`

Sets joint-space motion limits.

```yaml
- step: set_joint_limits
  arm: right
  max_speed: 60
  max_acc: 800
  jerk: 1300
```

Fields: `max_speed`, `max_acc`, `jerk`. All are optional.

### `start_vision`

Warms up the camera, YOLO model, and calibration, then starts the vision display.

```yaml
- step: start_vision
  description: Start camera and vision display
```

### `go_to`

Moves to a saved location in `locations/`. See [Joint-Mode vs Cartesian-Mode Replay](#joint-mode-vs-cartesian-mode-replay) for how recorded joint angles and offsets affect motion mode.

```yaml
- step: go_to
  arm: right
  location: rack_watch
  offset: [0, 0, 20]
  speed: 150
  acc: 300
```

Fields: `location` or `go_to`, `offset`, `speed`, `acc`, `parallel`.

Replay behavior:

- With `joint_angles_deg` and no `offset`, `go_to` uses joint-space replay.
- With `offset`, `go_to` uses cartesian replay and adds the offset to the saved `pose`.
- Without `joint_angles_deg`, `go_to` uses cartesian replay.
- If the location JSON has `"arm": "both"` and no explicit `arm` is provided, both arms move to their saved poses. Set `parallel: false` to move those bimanual entries sequentially.

### `move`

Moves relative to the current pose. By default this is tool-frame motion. Use `frame: base` for base-frame relative motion.

```yaml
- step: move
  arm: right
  frame: tool
  relative: [0, 0, -6, 18, 0, 0]
  speed: 100
  acc: 500
```

```yaml
- step: move
  arm: right
  frame: base
  relative: [-5, 0, 8]
  speed: 150
  acc: 200
```

Fields: `relative` as `[dx, dy, dz]` or `[dx, dy, dz, roll, pitch, yaw]`, `frame`, `speed`, `acc`.

### `move_joint`

Moves joints by relative joint deltas.

```yaml
- step: move_joint
  relative: [0, 0, 0, 0, 30, 0, 0]
  speed: 60
  acc: 500
```

### `move_joint_absolute`

Moves to absolute xArm joint angles. This is useful for importing recorded one-off scripts that were written directly against `set_servo_angle`.

```yaml
- step: move_joint_absolute
  angles: [-81.1, 11.2, -72.6, -31.9, 70.3, 30.4]
  speed: 15
  acc: 500
```

If fewer than all joint angles are provided, set `preserve_current: true` to keep the remaining current joint values:

```yaml
- step: move_joint_absolute
  angles: [-109.0, 7.2, -76.8, 0.3, 70.9]
  preserve_current: true
  speed: 20
```

Fields: `angles`, `speed`, `acc` or `mvacc`, `preserve_current`, `is_radian`.

### `tool_position`

Moves relative to the current tool frame using the xArm `set_tool_position` command.

```yaml
- step: tool_position
  relative: [0, 30, 0]
  speed: 50
  acc: 50
```

`relative` can be `[x, y, z]` or `[x, y, z, roll, pitch, yaw]`.

### `z_level`

Moves the current arm to an absolute Z height while keeping X, Y, and orientation.

```yaml
- step: z_level
  arm: left
  height: 89
  speed: 100
  acc: 200
```

Fields: `height` or `z_level`, `speed`, `acc`.

### `move_to_object`

Uses vision to move to an object preset from `configs/objects.yaml`.

```yaml
- step: move_to_object
  object: "14ml tube"
  offset: [3.5, 2.5]
  pick_type: ranked_vertical
  repeat: 3
  average_frames: 5
  min_frames: 3
  repeat_skip_mm: 2.0
```

Fields: `object`, `camera_arm`, `pick_type`, `offset`, `speed`, `acc`, `average_frames`, `repeat`, `repeat_skip_mm`, `min_frames`, `iou_threshold`, `display`, `raise_on_not_found`.

### `z_level_object`

Moves Z relative to a detected object's estimated height.

```yaml
- step: z_level_object
  object: "14ml tube"
  z_offset: 10
  average_frames: 5
```

### `grip`

Sets the xArm gripper position. Smaller values are more closed; larger values are more open.

```yaml
- step: grip
  arm: right
  state: 200
  speed: 1000
```

Fields: `state`, `speed`.

### `hand_position`

Sets a dexterous hand pose either from a saved hand location or raw servo angles.

```yaml
- step: hand_position
  arm: left
  location: hand/repeater-grip
```

```yaml
- step: hand_position
  arm: left
  angles: [0, 0, 0, 0, 0, 0]
```

### `handoff`

Uses YOLOE text-prompt detection and velocity-mode Cartesian control to align the gripper with a hand-held object, then closes the gripper when the object reaches the target image region. This step is intended for controlled handoff tasks where a person presents a tube to the robot.

```yaml
- step: handoff
  object: orange cap
  object_classes: [orange circle, orange cap, 50ml orange cap centrifuge tube]
  yolo_classes: [orange circle, orange cap, hand, person]
  hand_class: hand
  desired_position: [581, 656]
  desired_area: 16000
  area_tolerance: 2000
  camera_device: 2
  model: yoloe-11l-seg.pt
  confidence: 0.03
  grip_state: 200
```

Fields:

- `object`: target object name to prioritize.
- `object_classes`: YOLOE class names considered valid handoff targets.
- `yolo_classes`: text classes loaded into YOLOE with `set_classes`.
- `hand_class`: class used to choose the target closest to a detected hand.
- `desired_position`: target image pixel `[x, y]`.
- `desired_area`: target bounding-box area in pixels.
- `area_tolerance` and `position_tolerance_px`: on-target tolerances.
- `camera_device`, `frame_width`, `frame_height`: OpenCV camera settings.
- `model`: YOLOE model path or model name.
- `confidence`: YOLO confidence threshold.
- `pixel_to_mm_factor`, `area_to_mm_factor`, `max_xy_per_frame`, `max_z_per_frame`: visual-servo scaling and clamps.
- `velocity_duration`, `settle_seconds`: velocity command duration and per-frame settle time.
- `history_frames`, `required_hits`, `timeout_seconds`: success and timeout criteria.
- `grip_state`: gripper position to apply after a successful handoff.
- `display`: whether to show the annotated OpenCV handoff window.
- `require_hand`: if true, the target must be near a detected hand; otherwise the largest valid target can be used when no hand is visible.

### `qr_align`

Collects multiple QR-code angle measurements with OpenCV, filters for a consistent angle cluster, and rotates the tool around its Z axis to a target orientation.

```yaml
- step: qr_align
  camera_device: 0
  target_degrees: -90
  num_measurements: 9
  min_count: 6
  max_deviation: 5.0
  min_area: 400
  max_area: 8000
  speed: 10
  acc: 5
  display: true
```

Fields:

- `camera_device` or `source`: OpenCV camera index.
- `target_degrees`: desired QR edge orientation.
- `num_measurements`: number of valid QR angle measurements to collect.
- `min_count` and `max_deviation`: filtering criteria for a stable angle cluster.
- `min_area` and `max_area`: QR bounding polygon area filter.
- `max_rotation_step`: maximum tool-Z rotation per command, default `90`.
- `tolerance_degrees`: remaining rotation tolerance, default `0.5`.
- `speed` and `acc`: Cartesian rotation speed and acceleration.
- `timeout_seconds`: maximum collection time.
- `display`: whether to show the annotated QR collection window.

### `sleep`

Pauses execution.

```yaml
- step: sleep
  seconds: 0.25
```

Fields: `seconds` or `sleep`.

### `wait_until_visible`

Waits until an object is visible for a number of consecutive frames.

```yaml
- step: wait_until_visible
  object: "14ml tube"
  min_frames: 3
  delay: 2.0
  poll_interval: 0.2
```

### `move_world`

Moves to a world-frame coordinate transformed into the target arm base frame.

```yaml
- step: move_world
  arm: left
  position: [100, 200, 300]
  orientation: [-180, 0, 0]
```

### `move_other`

Moves to a coordinate expressed in another arm's base frame.

```yaml
- step: move_other
  arm: right
  reference_arm: left
  position: [100, 0, 200]
```

### `run`

Runs another YAML protocol or sub-protocol. Paths resolve relative to the current file first, then relative to `protocols/`.

```yaml
- step: run
  file: steps/rack/pick_up_from_rack.yaml
  args:
    object: "50ml eppendorf"
```

Sub-protocols can define their own `args`; caller-provided `args` override those defaults.

### `repeat`

Runs a nested protocol multiple times.

Linear offset mode:

```yaml
- step: repeat
  count: 5
  var: i
  offset_step: [0, -40, 0]
  offset_var: tube_offset
  protocol:
    - step: go_to
      location: repeater_first_tube
      arm: left
      offset: "{{tube_offset}}"
```

Grid offset mode:

```yaml
- step: repeat
  count: 20
  var: i
  offsets:
    right_offset:
      step: [40, 0, 0]
      y_step: [0, 60, 0]
      columns: 3
    left_offset:
      step: [25, 0, 0]
      y_step: [0, 25, 0]
      columns: 3
  protocol:
    - step: run
      file: steps/miniprep/right_rack_pickup.yaml
      args:
        loop_offset: "{{right_offset}}"
```

Fields: `count`, `var`, `protocol`, `offset_step`, `offset_var`, `columns`, `y_offset_step`, `offsets`, `stop_on_not_found`.

### `parallel`

Runs multiple branches concurrently and waits for all branches to finish. If any branch fails, the step fails after branches join.

```yaml
- step: parallel
  description: Remove covers and initialize pipette
  branches:
    - arm: left
      description: Remove covers
      protocol:
        - step: run
          file: steps/transfection/remove_covers.yaml
    - arm: right
      description: Initialize pipette
      protocol:
        - step: run
          file: steps/transfection/initialize_pipette.yaml
```

Each branch can set `arm`, `description`, `args`, and `protocol`. The branch `arm` becomes the default arm for steps inside that branch and for sub-protocols called from that branch.

Use `parallel` for arbitrary concurrent sub-procedures. Use bimanual `go_to` with `"arm": "both"` locations when both arms are simply moving to saved paired poses.

### `random_choice`

Sets a template variable to a random value from a list.

```yaml
- step: random_choice
  var: roll_suffix
  choices: ['', '_1', '_2', '_3', '_4', '_5', '_6']
```

The chosen value is available to following steps in the same args context.

### `dispense_circle`

High-level helper for synchronized dual-arm dispensing at random points in a circular region.

```yaml
- step: dispense_circle
  arms: [right, left]
  n_steps: 9
  radius_mm: 30.0
  z_top: 252.7
  z_bottom: 249.5
  speed: 80
  acc: 150
```

`arms[0]` is the pipette/tip arm. `arms[1]` is the plunger arm. The step moves both arms in base-frame XY together and ramps the plunger arm from `z_top` to `z_bottom`.

### `python_call`

Calls a helper function from `protocols/helpers/<module>.py`. Use this sparingly for logic that would make YAML unreadable.

```yaml
- step: python_call
  module: transformation
  function: grab_pipette
  args:
    which_one: middle
```

Helper functions receive one argument: the merged args dictionary.

```python
def grab_pipette(args):
    ...
```

### `stop`

Fails the protocol intentionally with a message.

```yaml
- step: stop
  description: "Protocol stopped by cleanup guard"
```

## Object Presets

Vision targets are defined in `configs/objects.yaml`. A preset includes:

- `shape`: physical geometry for 3D estimation. Supported shapes include `circle`, `square`, and `rect`.
- `shape.location`: target point inside the detected shape, such as `center`, `tl`, `tr`, `bl`, or `br`.
- `yolo_class`: class name from the active model/dataset.
- `confidence`: optional per-object confidence threshold.
- `pick_type`: object selection strategy.

Currently defined object presets include `50ml eppendorf`, `vortex genie hole`, `rack hole`, `14ml tube`, and `14ml rack hole`.

Common `pick_type` values include `toolhead_close`, `camera_center`, `largest`, `ranked`, `ranked_vertical`, `highest_confidence`, `tl`, `tr`, `bl`, and `br`.

## Locations

Saved locations live in `locations/*.json`. A single-arm location usually contains:

```json
{
  "pose": [119.23, 146.70, 435.51, -179.99, -0.00, -3.52],
  "position_mm": [119.23, 146.70, 435.51],
  "orientation_deg": [-179.99, -0.00, -3.52],
  "joint_angles_deg": [50.89, -47.15, -65.77, 0.0, 112.92, 54.41, 0.0],
  "arm": "right"
}
```

Bimanual locations use `"arm": "both"` and contain per-arm entries. `go_to` can move both arms for those locations when the step does not specify an explicit `arm`.

Use `record_location.py` to record arm poses and `record_hand.py` to record arm plus end-effector poses. Hand-only poses are stored under `locations/hand/` and can be applied with `hand_position`.

## Failure Blocks

A `failure` block runs when a protocol errors or is stopped. Keep it conservative:

```yaml
failure:
  - step: grip
    arm: right
    state: 700
  - step: go_to
    arm: right
    location: home
```

Good failure cleanup releases or relaxes grippers, moves arms to known safe locations, and turns off equipment. Failure blocks support core motion, gripper, hand, sub-protocol, parallel, and `python_call` cleanup steps. Avoid high-level random or dispensing steps in failure cleanup.

## Protocol Organization

Top-level runnable protocols live directly under `protocols/`:

- `vortexing.yaml`: pick a 50 ml tube from a rack, vortex it, and place it back.
- `miniprep.yaml`: loop over 14 ml tubes, vortex each tube, and place it into a destination rack.
- `repeater_demo.yaml`: short repeater pipette demo.
- `repeater_full.yaml`: full repeater pickup, handoff, dispense, and return workflow.
- `transfection.yaml`: transfection workflow.
- `transformation.yaml`: bacterial transformation workflow.
- `wave.yaml`: simple wave gesture.
- `test.yaml`: minimal smoke-test protocol.

Reusable pieces live under `protocols/steps/`:

- `steps/rack/`: generic rack pickup/place operations.
- `steps/vortex/`: vortex tube operation.
- `steps/miniprep/`: miniprep-specific 14 ml rack and vortexer steps.
- `steps/repeater/`: reusable repeater workflow steps.
- `steps/transfection/`: reusable transfection workflow steps.
- `steps/transformation/`: reusable transformation workflow steps.

Python helper escape hatches live under `protocols/helpers/`.

## Full Example: Vortexing

`protocols/vortexing.yaml` composes reusable steps:

```yaml
brief: "Vortex a 50ml eppendorf tube in the vortex genie."

protocol:
  - step: start_vision
    description: Start camera + vision display
  - step: go_to
    location: home
    description: Start at home
  - step: go_to
    location: rack_watch
    description: Move to rack watch position
  - step: run
    file: steps/rack/pick_up_from_rack.yaml
    description: Pick tube up from rack
    args:
      object: "50ml eppendorf"
  - step: go_to
    location: vortex_watch
    description: Move to vortex watch position
  - step: run
    file: steps/vortex/vortex_tube.yaml
    description: Vortex tube
    args:
      tube_object: "50ml eppendorf"
  - step: go_to
    location: rack_watch
  - step: run
    file: steps/rack/place_into_rack.yaml
    args:
      object: "rack hole"
```

## Full Example: Parallel + Dispense

The transfection protocol uses `parallel` for independent left/right tasks and `dispense_circle` for circular random dispensing:

```yaml
- step: parallel
  description: Remove dish covers and initialize pipette
  branches:
    - arm: left
      protocol:
        - step: run
          file: steps/transfection/remove_covers.yaml
    - arm: right
      protocol:
        - step: run
          file: steps/transfection/initialize_pipette.yaml

- step: dispense_circle
  arms: [right, left]
  n_steps: 9
  radius_mm: 30.0
  z_top: 252.7
  z_bottom: 249.5
  speed: 80
  acc: 150
```

## Authoring Guidance

Design protocols to stay declarative whenever possible. Use `parallel`, `repeat`, `random_choice`, and reusable `run` sub-protocols first. Use `python_call` sparingly only when the logic would otherwise become too hard to audit in YAML.
