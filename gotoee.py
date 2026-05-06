#!/usr/bin/env python3
"""
Move only the end-effector to the position saved in a location file,
or update a location file's end_effector field with the current state.

Usage:
    python gotoee.py hand/repeater-grip              # move EE to saved position
    python gotoee.py hand/repeater-grip --arm left   # explicit arm
    python gotoee.py hand/repeater-grip --save       # save current EE state into location file
    python gotoee.py hand/repeater-grip --visual     # open 3D editor then save
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Move end-effector to a saved location or update a location's EE state",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("location", type=str, help="Location name (from locations/<name>.json)")
    parser.add_argument("--arm", type=str, default=None, help="Which arm (default: from location file)")
    parser.add_argument("--save", action="store_true",
                        help="Save current end-effector state into the location file (instead of moving)")
    parser.add_argument("--visual", action="store_true",
                        help="Open 3D visual editor, then save result into the location file")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port for --visual (default: 8080)")
    parser.add_argument("--ip", type=str, default=None, help="Robot IP override")
    args = parser.parse_args()

    try:
        from aira.robot import arm, load_location
        from aira.utils.paths import get_project_root
    except ImportError as e:
        print(f"Error: {e}")
        return 1

    locations_dir = get_project_root() / "locations"
    loc_path = locations_dir / f"{args.location}.json"

    if not loc_path.exists():
        print(f"Error: location file not found: {loc_path}")
        return 1

    loc_data = load_location(args.location)
    arm_name = args.arm or loc_data.get("arm")
    label = arm_name or "default"

    a = arm(name=arm_name, ip=args.ip)

    if args.save or args.visual:
        ee = a.end_effector()
        if ee is None:
            ee = a.connect_end_effector()
        if ee is None:
            print(f"[{label}] No dexterous end-effector configured for this arm.")
            return 1

        if args.visual:
            print(f"[{label}] Opening visual editor...")
            ee_state = ee.visual_edit(port=args.port)
            if ee_state is None:
                print(f"[{label}] Visual editor cancelled.")
                return 1
        else:
            ee_state = ee.state_dict()

        loc_data["end_effector"] = ee_state
        with open(loc_path, "w") as f:
            json.dump(loc_data, f, indent=2)
        print(f"[{label}] Saved end-effector state to {loc_path}")
        print(f"  {ee_state}")
        return 0

    ee_state = loc_data.get("end_effector")
    if ee_state is None:
        print(f"Error: location '{args.location}' has no 'end_effector' field.")
        return 1

    ee = a.end_effector()
    if ee is None:
        ee = a.connect_end_effector()
    if ee is None:
        print(f"[{label}] No dexterous end-effector configured for this arm.")
        return 1

    print(f"[{label}] Moving end-effector to '{args.location}' state...")
    if ee.load_state_dict(ee_state):
        print(f"[{label}] Done. angles={ee_state.get('angles')}")
    else:
        print(f"[{label}] Failed to set end-effector position.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
