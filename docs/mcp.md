# Remote MCP Server

`run_mcp.py` exposes the LabOS Robot Runtime as a remote MCP device using the LabOS Python API (`labos.mcp.RemoteMCP`). The robot opens an outbound WebSocket connection to a LabOS relay. Remote agents call registered tools through the relay, while the local runtime streams progress, logs, structured events, and fatal robot errors back to the agent.

`run_mock_mcp.py` uses the same tool definitions but sets `MOCK_MODE=1`, so the protocol runner emits fake step progress and never touches the arm, camera, or end-effectors.

## Configuration

Copy the template and fill in the relay credentials:

```bash
cp .env.example .env
```

Required settings:

```text
LABOS_URL=wss://example.com/remote
LABOS_API_KEY=replace-with-tool-key
LABOS_DEVICE_ID=labos-robot-runtime
```

Optional settings:

```text
LABOS_HEARTBEAT_INTERVAL=15
MOCK_MODE=0
HEADLESS=0
```

`RemoteMCP()` reads `.env` from the current working directory. `HEADLESS=0` shows the OpenCV camera/YOLO vision display by default. Set `HEADLESS=1` or pass `--headless` to skip the vision window.

## Running

Real robot:

```bash
./run.sh
```

Headless real robot:

```bash
./run.sh --headless
```

Mock mode:

```bash
./run.sh --mock
```

Windows equivalents:

```bat
run.bat
run.bat --headless
run.bat --mock
```

Direct Python entrypoints:

```bash
python run_mcp.py
python run_mcp.py --headless
python run_mock_mcp.py
```

Runtime logs are written under `logs/`:

```text
logs/run_mcp.log
logs/run_mock_mcp.log
```

The same logger also writes to the terminal. In mock mode the logs include connection, disconnection, heartbeat sends, and fake protocol step progress.

## Tool Reference

Status and discovery:

- `get_status()`: current protocol state as a short human-readable string.
- `get_protocols()`: available YAML protocols with brief descriptions and major steps.
- `describe_protocol(protocol_name)`: details for one protocol.
- `get_object_definitions()`: exact object names and shapes from `configs/objects.yaml`.

Protocol execution:

- `start_protocol(protocol_name, blocking=false)`: starts a YAML protocol. By default it returns immediately. With `blocking=true`, the tool call stays open and streams progress until the protocol finishes or fails.
- `stop_robot()`: requests protocol stop and returns the robot to position control mode.

Manual control:

- `manual_mode()`: puts the default arm in teaching/manual mode.
- `go_home()`: moves the default arm to the recorded `home` location.
- `gripper(position)`: accepts `close`, `midway`, `open`, or a numeric `0..800` position.
- `z_level(level)`: accepts `low`, `medium`, `high`, or a numeric Z height in millimeters.
- `is_holding_something()`: gripper-position heuristic.

Vision:

- `list_objects()`: latest visible detections with pixel coordinates, color, and depth when available.
- `see_object(object_name)`: boolean visibility check.
- `move_to_object(object_name, target_px_x=null, target_px_y=null)`: vision-guided move to a configured object, optionally targeting a specific pixel coordinate.

## Progress Updates

Blocking calls use `ctx.progress(value, message)`, where `value` is `progress_pct / 100` from `aira.protocol_runner.get_protocol_status()` and `message` is the current step description when available.

The server also logs each status change locally, including protocol name, step number, step type, description, and percent complete.

Non-blocking `start_protocol(..., blocking=false)` returns immediately after emitting:

```text
protocol.started
```

After that, a background websocket watcher emits structured events:

```text
protocol.progress
```

The event payload is the protocol runner status dict:

```json
{
  "state": "running",
  "protocol_name": "miniprep",
  "current_step_index": 3,
  "current_step_name": "go_to",
  "current_step_description": "Move above rack",
  "total_steps": 10,
  "progress_pct": 30.0,
  "error": null
}
```

## Fatal Robot Errors

If a protocol step raises because the xArm SDK returned an error code, the protocol runner marks the status as failed. The MCP server then:

1. Best-effort clears robot errors and returns the arm to position control.
2. Sends `ctx.fatal("xarm_error", ...)` for blocking calls, or a websocket `tool.fatal` event for background progress.
3. Locks the MCP server process.
4. Prints this message in the terminal:

```text
MCP locked due to fatal robot error. Please restart this script to clear.
```

The same message is saved to the active log file.

After a fatal lock, every later tool call returns:

```text
Error: MCP server locked due to a prior fatal robot error. Please restart this script.
```

Restart `run.sh`, `run.bat`, or the Python entrypoint after checking the robot.

## Mock Mode

Mock mode is useful for testing agent behavior without hardware:

```bash
./run.sh --mock
```

Mock behavior:

- `MOCK_MODE=1`.
- `HEADLESS=1` by default.
- Protocol steps sleep briefly and emit realistic progress.
- Arm, gripper, hand, and vision tools return `[mock] success` strings.
- Status and protocol discovery still read the real YAML files.

## Agent Usage Example

A typical agent flow:

1. Call `get_protocols()` to see what is available.
2. Call `describe_protocol("miniprep")` to summarize the selected protocol.
3. Call `start_protocol("miniprep", blocking=true)` when the operator is ready.
4. Watch streamed progress messages until the tool returns.
5. If the operator asks to stop, call `stop_robot()`.

For non-blocking operation, call `start_protocol("miniprep")`, then use `get_status()` or subscribe to `protocol.progress` events through the relay.

## Troubleshooting

### Missing LabOS settings

If startup fails with missing URL or API key, confirm `.env` exists in the repository root and contains `LABOS_URL` and `LABOS_API_KEY`.

### Vision window does not open

Use `HEADLESS=1` or `--headless` when running over SSH or on a headless machine. If the vision display fails, the MCP server logs the error and continues.

### Mock mode unexpectedly active

Check:

```bash
echo "$MOCK_MODE"
```

Unset it or set `MOCK_MODE=0` for real hardware.

### Server is locked after fatal error

This is intentional. Inspect the robot, clear any physical issue, then restart the MCP script.
