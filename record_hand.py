#!/usr/bin/env python3
"""
Record arm pose + end-effector (DM17 hand) position to a location JSON file.

Two modes:
  (default)   The operator physically positions the arm/hand, then presses Enter.
  --visual    Opens a browser-based 3D URDF visualizer (viser) with interactive
              joint sliders, draggable fingertip IK gizmos, per-joint rotation
              rings, and tier-linked finger group controls.

Usage:
    python record_hand.py --arm left locations/hand/pipette/hold.json
    python record_hand.py --arm left --visual locations/hand/open.json
    python record_hand.py --visual locations/hand/fist.json          # no robot
    python record_hand.py --visual --start locations/hand/pgrip-1.json locations/hand/pgrip-2.json
"""

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_arm_pose(a, arm_label):
    """Read pose + joints from an arm. Returns dict or None."""
    code, pose = a.get_position()
    if code != 0:
        print(f"[{arm_label}] Failed to read current position.")
        return None
    code_j, joints = a.get_joint_angles()
    data = {
        "pose": [float(x) for x in pose],
        "position_mm": [float(pose[0]), float(pose[1]), float(pose[2])],
        "orientation_deg": [float(pose[3]), float(pose[4]), float(pose[5])],
    }
    if code_j == 0 and joints:
        data["joint_angles_deg"] = [float(j) for j in joints]
    return data


def _connect_hand(arm_proxy, ee_cfg):
    """Connect and return ZWDM17XArmController, or None on failure."""
    ee_type = ee_cfg.get("type")
    if ee_type != "zw-dm17":
        print(f"Error: Unsupported end-effector type {ee_type!r}")
        return None
    _zw = importlib.import_module("aira.endeffector.zw-dm17.controller")
    ctrl = _zw.ZWDM17XArmController(
        device_id=ee_cfg.get("device_id", 1),
        baudrate=ee_cfg.get("baudrate", 115200),
    )
    print("Connecting to DM17 hand...")
    if not ctrl.connect(arm_proxy.arm):
        print("Error: DM17 hand did not initialise.")
        return None
    print("Hand connected.")
    return ctrl


def _save_location(out_path, data):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved location to {out_path}")
    if "arm" in data:
        print(f"  arm: {data['arm']}")
    if "position_mm" in data:
        print(f"  position_mm: {data['position_mm']}")
    if "orientation_deg" in data:
        print(f"  orientation_deg: {data['orientation_deg']}")
    if "joint_angles_deg" in data:
        print(f"  joint_angles_deg: {data['joint_angles_deg']}")
    if "end_effector" in data:
        ee = data["end_effector"]
        print(f"  end_effector type: {ee.get('type')}")
        print(f"  end_effector angles: {ee.get('angles')}")


# ── Visual mode ──────────────────────────────────────────────────────


def _run_visual(args, out_path, arm_proxy, ee_ctrl, ee_cfg, start_angles=None):
    """Launch the viser-based 3D URDF hand visualizer via the shared module."""
    _vis = importlib.import_module("aira.endeffector.zw-dm17.visual")

    if arm_proxy is not None:
        print(f"[{args.arm}] Entering manual mode (arm is free to move).")
        arm_proxy.set_manual_mode()

    port = getattr(args, "port", 8080)
    angles = _vis.dm17_visual_edit(
        ee_ctrl=ee_ctrl,
        start_angles=start_angles,
        port=port,
    )

    if angles is None:
        if arm_proxy is not None:
            arm_proxy.set_position_mode()
        return 1

    data: dict = {}
    if arm_proxy is not None:
        arm_data = _read_arm_pose(arm_proxy, args.arm)
        if arm_data is not None:
            data.update(arm_data)
        data["arm"] = args.arm
        arm_proxy.set_position_mode()

    data["end_effector"] = {"type": "zw-dm17", "angles": angles}
    _save_location(out_path, data)
    return 0


# ── CLI mode (physical) ─────────────────────────────────────────────


def _run_physical(args, out_path, arm_proxy, ee_ctrl):
    arm_name = args.arm
    print(f"[{arm_name}] Entering manual mode.")
    arm_proxy.set_manual_mode()

    try:
        input(f"\n[{arm_name}] Position the arm and hand, then press Enter to record... ")
    except KeyboardInterrupt:
        print("\nCancelled.")
        arm_proxy.set_position_mode()
        return 1

    data = _read_arm_pose(arm_proxy, arm_name)
    arm_proxy.set_position_mode()

    if data is None:
        return 1

    data["arm"] = arm_name

    try:
        ee_state = ee_ctrl.state_dict()
        data["end_effector"] = ee_state
        print(f"  end_effector: {ee_state}")
    except Exception as e:
        print(f"Warning: Failed to read end-effector state: {e}")

    _save_location(out_path, data)
    return 0


# ── Main ─────────────────────────────────────────────────────────────


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Record arm + end-effector position to a location JSON file",
    )
    parser.add_argument("output", type=str, help="Output file path")
    parser.add_argument("--arm", type=str, default=None, help="Arm name (e.g. left)")
    parser.add_argument("--ip", type=str, default=None, help="Robot IP override")
    parser.add_argument("--visual", action="store_true", help="3D URDF visualizer mode")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--start", type=str, default=None,
                        help="Load initial hand position from a location JSON file")
    args = parser.parse_args()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    start_angles = None
    if args.start:
        start_path = Path(args.start)
        if not start_path.is_absolute():
            start_path = ROOT / start_path
        if not start_path.exists():
            print(f"Error: Start file not found: {start_path}")
            return 1
        with open(start_path) as f:
            start_data = json.load(f)
        ee = start_data.get("end_effector", {})
        if "angles" in ee:
            start_angles = [max(0, min(1000, int(v))) for v in ee["angles"]]
            print(f"Loaded start position from {start_path} ({len(start_angles)} joints)")
        else:
            print(f"Warning: No end_effector.angles in {start_path}, ignoring --start")

    arm_proxy = None
    ee_ctrl = None
    ee_cfg = None

    if args.arm:
        from aira.robot import arm, load_robot_mapping
        arm_name = args.arm.strip()
        mapping = load_robot_mapping()
        if arm_name not in mapping:
            print(f"Error: Unknown arm {arm_name!r}. Available: {list(mapping.keys())}")
            return 1
        arm_cfg = mapping[arm_name]
        ee_cfg = arm_cfg.get("end_effector")
        print(f"Connecting to arm '{arm_name}'...")
        try:
            arm_proxy = arm(name=arm_name, ip=args.ip)
        except Exception as e:
            print(f"Connection failed: {e}")
            return 1
        if ee_cfg and ee_cfg.get("type") not in (None, "xarm-gripper", "xarm-gripper2"):
            ee_ctrl = _connect_hand(arm_proxy, ee_cfg)
            if ee_ctrl is None:
                return 1
    elif not args.visual:
        print("Error: --arm is required unless --visual is used.")
        return 1

    if args.visual:
        return _run_visual(args, out_path, arm_proxy, ee_ctrl, ee_cfg, start_angles=start_angles)
    else:
        if arm_proxy is None or ee_ctrl is None:
            print("Error: --arm with a dexterous end-effector is required for physical mode.")
            return 1
        return _run_physical(args, out_path, arm_proxy, ee_ctrl)


if __name__ == "__main__":
    sys.exit(main())
