#!/usr/bin/env python3
"""
Scan the xArm tool-end RS485 bus for ZWHAND DM17 devices.

Iterates through Modbus device IDs 1-255, sends a get_init_state() probe
with a short timeout, and reports which IDs responded.

Usage (from the robot/ directory):

    PYTHONPATH=. python scripts/scan_zwdm17.py --ip 192.168.1.202
    PYTHONPATH=. python scripts/scan_zwdm17.py --arm left
    PYTHONPATH=. python scripts/scan_zwdm17.py --arm left --baud 2000000
    PYTHONPATH=. python scripts/scan_zwdm17.py --arm left --start 1 --end 10
"""

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aira.robot import load_robot_mapping, XArmController

logger = logging.getLogger(__name__)

TGPIO_HOST_ID = 9


def _modbus_crc(data: list[int]) -> list[int]:
    """Standard Modbus CRC-16 (polynomial 0xA001). Returns [crc_lo, crc_hi]."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc >>= 1
                crc ^= 0xA001
            else:
                crc >>= 1
    return [crc & 0xFF, (crc >> 8) & 0xFF]


def probe_device(arm, device_id: int) -> bool:
    """
    Send a get_init_state read (function 0x04, address 0x00, 1 register)
    to *device_id* via transparent transmission and return True if it responds.

    Calls ``arm_cmd.tgpio_set_modbus`` directly to avoid the SDK's
    ``_check_modbus_code`` error-latching logic.
    """
    frame = [device_id, 0x04, 0x00, 0x00, 0x00, 0x01]
    frame_with_crc = frame + _modbus_crc(frame)
    try:
        ret = arm.core.tgpio_set_modbus(
            frame_with_crc,
            len(frame_with_crc),
            host_id=TGPIO_HOST_ID,
            is_transparent_transmission=True,
        )
        if ret[0] == 0 and len(ret) >= 7:
            return True
    except Exception:
        pass
    return False


def scan(arm, start: int, end: int, baudrate: int) -> list[int]:
    """Scan device IDs [start, end] and return the list that responded."""
    code = arm.set_tgpio_modbus_baudrate(baudrate)
    if code != 0:
        logger.error("Failed to set tool RS485 baudrate to %d (code=%d)", baudrate, code)
        return []
    logger.info("Tool RS485 baudrate set to %d", baudrate)
    time.sleep(2)

    code = arm.set_tgpio_modbus_timeout(100, is_transparent_transmission=True)
    if code != 0:
        logger.warning("set_tgpio_modbus_timeout returned code=%d", code)

    total = end - start + 1
    found: list[int] = []

    logger.info("Scanning device IDs %d–%d at %d baud …", start, end, baudrate)
    for device_id in range(start, end + 1):
        label = f"[{device_id:3d}/{end}]"
        if probe_device(arm, device_id):
            found.append(device_id)
            logger.info("%s ID %d — FOUND", label, device_id)
        else:
            if total <= 20:
                logger.debug("%s ID %d — no response", label, device_id)

        progress = (device_id - start + 1) / total * 100
        if device_id % 25 == 0 or device_id == end:
            logger.info("Progress: %d/%d (%.0f%%)", device_id - start + 1, total, progress)

    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan for ZWHAND DM17 devices on xArm tool-end RS485",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    conn = parser.add_mutually_exclusive_group(required=True)
    conn.add_argument("--ip", type=str, help="xArm IP address directly")
    conn.add_argument("--arm", type=str, help="Arm name from robot_mapping.json")

    parser.add_argument("--baud", type=int, default=115200,
                        help="RS485 baudrate to scan at (default: 115200)")
    parser.add_argument("--start", type=int, default=1,
                        help="First device ID to probe (default: 1)")
    parser.add_argument("--end", type=int, default=255,
                        help="Last device ID to probe (default: 255)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.arm:
        mapping = load_robot_mapping()
        if args.arm not in mapping:
            logger.error("Unknown arm %r. Available: %s", args.arm, list(mapping.keys()))
            return 1
        ip = mapping[args.arm]["ip"]
    else:
        ip = args.ip

    logger.info("Connecting to xArm at %s …", ip)
    ctrl = XArmController(ip)
    if not ctrl.connect():
        logger.error("Failed to connect to xArm at %s", ip)
        return 1

    try:
        found = scan(ctrl.arm, args.start, args.end, args.baud)
    finally:
        ctrl.disconnect()

    print()
    if found:
        print(f"Found {len(found)} device(s) at baud {args.baud}:")
        for dev_id in found:
            print(f"  Device ID: {dev_id}")
    else:
        print(f"No devices found (IDs {args.start}–{args.end} at {args.baud} baud).")
        print()
        print("Troubleshooting:")
        print("  - Is the hand powered (12V)?")
        print("  - Is RS485 A/B wired to the xArm tool connector?")
        print(f"  - Try a different baud rate: --baud 9600 / 921600 / 2000000")

    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
