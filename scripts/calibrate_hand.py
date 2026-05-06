#!/usr/bin/env python3
"""
Zero-position calibration for ZWHAND DM17 end-effectors connected via xArm RS485.

Usage (from the robot/ directory):

    PYTHONPATH=. python scripts/calibrate_hand.py --arm right
    PYTHONPATH=. python scripts/calibrate_hand.py --all
    PYTHONPATH=. python scripts/calibrate_hand.py --arm right --steppers-only
"""

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aira.robot import load_robot_mapping, XArmController

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = {"zw-dm17"}

# The vendor directory uses a hyphen ("zw-dm17") which is not a valid
# Python identifier, so we must use importlib for the import.
_zw_dm17_ctrl = importlib.import_module("aira.endeffector.zw-dm17.controller")
ZWDM17XArmController = _zw_dm17_ctrl.ZWDM17XArmController


def _build_hand(ee_cfg: dict) -> "ZWDM17XArmController":
    """Instantiate the right EndEffector subclass from a config dict."""
    ee_type = ee_cfg.get("type", "")
    if ee_type == "zw-dm17":
        return ZWDM17XArmController(
            device_id=ee_cfg.get("device_id", 1),
            baudrate=ee_cfg.get("baudrate", 115200),
        )
    raise ValueError(f"Unsupported end-effector type: {ee_type!r}")


def calibrate_arm(
    arm_name: str,
    arm_cfg: dict,
    steppers_only: bool = False,
) -> bool:
    """Connect to one xArm, calibrate its hand, then disconnect."""
    ee_cfg = arm_cfg.get("end_effector")
    if ee_cfg is None:
        logger.info("[%s] No end_effector configured — skipping", arm_name)
        return True
    ee_type = ee_cfg.get("type", "")
    if ee_type not in SUPPORTED_TYPES:
        logger.info("[%s] End-effector type %r not calibratable — skipping", arm_name, ee_type)
        return True

    ip = arm_cfg["ip"]
    logger.info("[%s] Connecting to xArm at %s …", arm_name, ip)
    ctrl = XArmController(ip)
    if not ctrl.connect():
        logger.error("[%s] Failed to connect to xArm", arm_name)
        return False

    try:
        hand = _build_hand(ee_cfg)
        if not hand.connect(ctrl.arm):
            logger.error("[%s] Hand did not initialise", arm_name)
            return False

        if steppers_only:
            logger.info("[%s] Calibrating stepper joints only …", arm_name)
            ok = hand.calibrate_all_steppers()
            wait_s = 2
        else:
            logger.info("[%s] Calibrating all joints (this takes ~13 s) …", arm_name)
            ok = hand.calibrate()
            wait_s = 14

        if not ok:
            logger.error("[%s] Calibration command failed", arm_name)
            return False

        time.sleep(wait_s)

        time.sleep(1)

        angles = hand.get_angles()
        if angles is False:
            logger.warning("[%s] Could not read back angles after calibration", arm_name)
        else:
            logger.info("[%s] Post-calibration angles: %s", arm_name, angles)

        hand.clear_errors()
        errors = hand.get_errors()
        if errors is not False and any(e != 0 for e in errors):
            active = {i: v for i, v in enumerate(errors) if v != 0}
            logger.warning("[%s] Errors still present after calibration: %s", arm_name, active)
        else:
            logger.info("[%s] No errors after calibration", arm_name)

        # --- Post-calibration grasp test ---
        logger.info("[%s] Grasp test: closing all fingers …", arm_name)
        grasp = [1000] * 17
        grasp[3 - 1] = 0   # joint 3 → index 2
        grasp[16 - 1] = 0  # joint 16 → index 15
        hand.set_all_absolute(grasp)
        time.sleep(3)

        grasp_angles = hand.get_angles()
        if grasp_angles is not False:
            logger.info("[%s] Grasp angles: %s", arm_name, grasp_angles)

        logger.info("[%s] Opening all fingers …", arm_name)
        hand.set_all_absolute([0] * 17)
        time.sleep(3)

        open_angles = hand.get_angles()
        if open_angles is not False:
            logger.info("[%s] Open angles:  %s", arm_name, open_angles)

        saved = hand.save_config()
        if saved:
            logger.info("[%s] Calibration data saved (power-off save)", arm_name)
        else:
            logger.warning("[%s] Failed to save calibration data", arm_name)

        hand.disconnect()
        logger.info("[%s] Calibration + grasp test complete", arm_name)
        return True

    finally:
        ctrl.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate ZWHAND DM17 end-effectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--arm", type=str, help="Name of a single arm to calibrate (e.g. 'right')")
    group.add_argument("--all", action="store_true", help="Calibrate all arms with supported end-effectors")
    parser.add_argument("--steppers-only", action="store_true", help="Only calibrate stepper joints (faster)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    mapping = load_robot_mapping()

    if args.all:
        arm_names = list(mapping.keys())
    else:
        if args.arm not in mapping:
            logger.error("Unknown arm %r. Available: %s", args.arm, list(mapping.keys()))
            return 1
        arm_names = [args.arm]

    failures = []
    for name in arm_names:
        ok = calibrate_arm(name, mapping[name], steppers_only=args.steppers_only)
        if not ok:
            failures.append(name)

    if failures:
        logger.error("Calibration failed for: %s", failures)
        return 1

    logger.info("All done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
