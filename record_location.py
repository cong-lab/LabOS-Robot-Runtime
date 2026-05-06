#!/usr/bin/env python3
"""
Record the current robot pose as a named location (joint + base cartesian) into locations/.

Supports multi-arm setups via --arm left|right|both (default: both).
When both arms are active, all arms enter manual mode simultaneously so
the operator can position them before pressing Enter once.

With --visual, a browser-based 3D editor opens for dexterous end-effectors
so the operator can set hand joint positions interactively.

Usage:
    python record_location.py my_pose                  # default: both arms
    python record_location.py --name pick_above --arm left
    python record_location.py my_pose --arm right
    python record_location.py my_pose --arm left --visual
    python record_location.py my_pose --zdown           # z_down() before saving
    python record_location.py my_pose --ip 192.168.1.195
"""

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_arm_pose(a, arm_label):
    """Read pose + joints from an arm that is already in manual mode. Returns dict or None."""
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
    if getattr(a, "on_linear_rail", False):
        try:
            code_r, rail_pos = a.get_linear_rail_pos()
            data["linear_rail_mm"] = float(rail_pos) if code_r == 0 else 0.0
            print(f"[{arm_label}] Linear rail position: {data['linear_rail_mm']} mm")
        except Exception:
            data["linear_rail_mm"] = 0.0
            print(f"[{arm_label}] Could not read linear rail position, defaulting to 0.")
    return data


def _restore_all(arms):
    """Restore position mode on all arm instances (best-effort)."""
    for name, a in arms.items():
        try:
            a.set_position_mode()
        except Exception:
            pass


def _connect_ee(arm_proxy, ee_cfg, arm_name):
    """Connect to an end-effector. Returns (controller, type) or (None, type)."""
    ee_type = ee_cfg.get("type", "")
    if ee_type == "zw-dm17":
        mod = importlib.import_module("aira.endeffector.zw-dm17.controller")
        ctrl = mod.ZWDM17XArmController(
            device_id=ee_cfg.get("device_id", 1),
            baudrate=ee_cfg.get("baudrate", 115200),
        )
        print(f"[{arm_name}] Connecting to DM17 hand...")
        if not ctrl.connect(arm_proxy.arm):
            print(f"[{arm_name}] Warning: DM17 hand did not initialise.")
            return None, ee_type
        return ctrl, ee_type
    if ee_type == "xarm-gripper2":
        from aira.endeffector.xarm_gripper2 import XArmGripper2
        ctrl = XArmGripper2(speed=ee_cfg.get("speed", 5000))
        print(f"[{arm_name}] Connecting to xArm gripper...")
        if not ctrl.connect(arm_proxy.arm):
            print(f"[{arm_name}] Warning: xArm gripper did not initialise.")
            return None, ee_type
        return ctrl, ee_type
    print(f"[{arm_name}] Unsupported end-effector type {ee_type!r}, skipping.")
    return None, ee_type


def _try_record_end_effector(arm_proxy, arm_name, visual=False, port=8080):
    """If the arm has a dexterous end-effector, record its state.

    With *visual=True*, opens a 3D editor instead of reading raw state.
    Returns a state_dict or None.
    """
    from aira.robot import load_robot_mapping
    mapping = load_robot_mapping()
    arm_cfg = mapping.get(arm_name, {})
    ee_cfg = arm_cfg.get("end_effector")
    if ee_cfg is None or ee_cfg.get("type") == "xarm-gripper":
        return None

    resp = input(f"[{arm_name}] Record end-effector position? [Y/n] ").strip().lower()
    if resp in ("n", "no"):
        return None

    ctrl, ee_type = _connect_ee(arm_proxy, ee_cfg, arm_name)
    if ctrl is None:
        print(f"[{arm_name}] Skipping end-effector recording.")
        return None

    if visual:
        print(f"[{arm_name}] Opening visual editor for {ee_type} hand...")
        result = ctrl.visual_edit(port=port)
        if result is not None:
            print(f"[{arm_name}] End-effector recorded via visual editor: {result}")
        else:
            print(f"[{arm_name}] Visual editor cancelled.")
        return result
    else:
        try:
            state = ctrl.state_dict()
            print(f"[{arm_name}] End-effector recorded: {state}")
            return state
        except Exception as e:
            print(f"[{arm_name}] Warning: Failed to read end-effector state: {e}")
            return None


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Record current robot pose to locations/<name>.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "location",
        nargs="?",
        type=str,
        default=None,
        help="Location name (saved as locations/<name>.json)",
    )
    parser.add_argument("--name", "-n", type=str, default=None,
                        help="Location name (alternative to positional)")
    parser.add_argument("--ip", type=str, default=None,
                        help="Robot IP (default: from robot_mapping.json or handeye_calibration_data.json)")
    parser.add_argument("--arm", type=str, default="both", choices=["left", "right", "both"],
                        help="Which arm to record (default: both)")
    parser.add_argument("--zdown", action="store_true",
                        help="Run z_down() after positioning to align tool Z with base Z before saving")
    parser.add_argument("--visual", action="store_true",
                        help="Open 3D visual editor for dexterous end-effector positioning")
    parser.add_argument("--port", type=int, default=8080,
                        help="Viser server port for --visual (default: 8080)")
    args = parser.parse_args()

    location_name = (args.name or args.location or "").strip()
    if not location_name:
        parser.error("Location name required (e.g. record_location.py my_pose or --name my_pose)")

    try:
        from aira.robot import arm, get_arm_names
        from aira.utils.paths import get_project_root
    except ImportError as e:
        print(f"Error: {e}")
        return 1

    locations_dir = get_project_root() / "locations"
    locations_dir.mkdir(parents=True, exist_ok=True)
    out_path = locations_dir / f"{location_name}.json"

    arm_choice = args.arm

    if arm_choice == "both":
        arm_names = get_arm_names()
        if len(arm_names) < 2:
            arm_choice = arm_names[0] if arm_names else None
            print(f"Only one arm configured ({arm_choice}), recording single arm.")

    if arm_choice == "both":
        arm_names = get_arm_names()
        arms = {}
        for name in arm_names:
            print(f"Connecting to arm '{name}'...")
            try:
                arms[name] = arm(name=name, ip=args.ip)
            except Exception as e:
                print(f"Connection to '{name}' failed: {e}")
                _restore_all(arms)
                return 1

        for name, a in arms.items():
            print(f"[{name}] Entering manual mode.")
            a.set_manual_mode()

        try:
            input("\nMove all arms to desired poses, then press Enter... ")
        except KeyboardInterrupt:
            print("\nCancelled.")
            _restore_all(arms)
            return 1

        _restore_all(arms)

        if args.zdown:
            for name, a in arms.items():
                print(f"[{name}] Running z_down()...")
                a.z_down()

        result = {"arm": "both"}
        ok = True
        for name, a in arms.items():
            data = _read_arm_pose(a, name)
            if data is None:
                ok = False
                break
            ee_state = _try_record_end_effector(
                a, name, visual=args.visual, port=args.port,
            )
            if ee_state is not None:
                data["end_effector"] = ee_state
            result[name] = data

        if not ok:
            return 1

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved bimanual location '{location_name}' to {out_path}")
        for name in arm_names:
            sub = result[name]
            print(f"  [{name}] position_mm: {sub['position_mm']}")
            if "linear_rail_mm" in sub:
                print(f"  [{name}] linear_rail_mm: {sub['linear_rail_mm']}")
            if "end_effector" in sub:
                print(f"  [{name}] end_effector: {sub['end_effector'].get('type')}")
        return 0

    # --- Single arm ---
    print("Connecting to robot...")
    try:
        a = arm(name=arm_choice, ip=args.ip)
    except Exception as e:
        print(f"Connection failed: {e}")
        return 1

    label = arm_choice or "default"
    print(f"[{label}] Entering manual mode.")
    a.set_manual_mode()
    try:
        input(f"[{label}] Move the robot to the desired pose, then press Enter... ")
    except KeyboardInterrupt:
        print("\nCancelled.")
        a.set_position_mode()
        return 1

    a.set_position_mode()

    if args.zdown:
        print(f"[{label}] Running z_down()...")
        a.z_down()

    data = _read_arm_pose(a, label)
    if data is None:
        return 1

    if arm_choice:
        data["arm"] = arm_choice

    ee_state = _try_record_end_effector(
        a, arm_choice or "default", visual=args.visual, port=args.port,
    )
    if ee_state is not None:
        data["end_effector"] = ee_state

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved location '{location_name}' to {out_path}")
    print(f"  position_mm: {data['position_mm']}")
    print(f"  orientation_deg: {data['orientation_deg']}")
    if "joint_angles_deg" in data:
        print(f"  joint_angles_deg: {data['joint_angles_deg']}")
    if arm_choice:
        print(f"  arm: {arm_choice}")
    if "linear_rail_mm" in data:
        print(f"  linear_rail_mm: {data['linear_rail_mm']}")
    if "end_effector" in data:
        print(f"  end_effector: {data['end_effector'].get('type')} angles={data['end_effector'].get('angles')}")
    print("Use: arm().go_to('" + location_name + "')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
