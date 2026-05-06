#!/usr/bin/env python3
"""
Move a robot arm to a saved location from locations/.

Usage:
    python goto.py miniprep_rack                    # default arm
    python goto.py miniprep_rack --arm right
    python goto.py miniprep_home --arm right --zdown
    python goto.py miniprep_rack --speed 500 --acc 800
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Move arm to a saved location",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("location", type=str, help="Location name (from locations/<name>.json)")
    parser.add_argument("--arm", type=str, default=None, help="Which arm to move (default: from location file or default)")
    parser.add_argument("--zdown", action="store_true", help="Run z_down() after arriving at the location")
    parser.add_argument("--speed", type=float, default=None, help="Override movement speed (mm/s)")
    parser.add_argument("--acc", type=float, default=None, help="Override movement acceleration (mm/s²)")
    parser.add_argument("--ip", type=str, default=None, help="Robot IP override")
    parser.add_argument("--offset", type=float, nargs="+", default=None,
                        metavar="V", help="Offset [dx dy dz [droll dpitch dyaw]] in mm/deg")
    args = parser.parse_args()

    try:
        from aira.robot import arm
    except ImportError as e:
        print(f"Error: {e}")
        return 1

    arm_name = args.arm
    a = arm(name=arm_name, ip=args.ip)
    label = arm_name or "default"

    kwargs = {}
    if args.speed is not None:
        kwargs["speed"] = args.speed
    if args.acc is not None:
        kwargs["acc"] = args.acc
    if args.offset is not None:
        kwargs["offset"] = args.offset

    print(f"[{label}] Moving to '{args.location}'...")
    a.go_to(args.location, **kwargs)
    print(f"[{label}] Arrived at '{args.location}'.")

    if args.zdown:
        print(f"[{label}] Running z_down()...")
        a.z_down()
        code, pose = a.get_position()
        if code == 0:
            print(f"[{label}] Orientation: roll={pose[3]:.1f} pitch={pose[4]:.1f} yaw={pose[5]:.1f}")
        print(f"[{label}] z_down complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
