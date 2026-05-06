#!/usr/bin/env python3
"""
Comprehensive RS485 diagnostics for the ZWHAND DM17 via xArm tool-end.

Tests multiple baud rates, device IDs, Modbus function codes, and both
normal and transparent-transmission modes to identify why the hand is
not responding.

Usage (from the robot/ directory):

    PYTHONPATH=. python scripts/test_zwdm17.py --ip 192.168.1.202
    PYTHONPATH=. python scripts/test_zwdm17.py --arm left
"""

import argparse
import contextlib
import logging
import os
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aira.robot import load_robot_mapping, XArmController

logger = logging.getLogger(__name__)

TGPIO_HOST_ID = 9

BAUD_RATES = [115200, 9600, 921600, 2000000]
DEVICE_IDS = [1, 2]

INIT_STATE_ADDR = 0x00
SW_VER_ADDR = 0x03


# ------------------------------------------------------------------
# CRC-16/Modbus (polynomial 0xA001) — matches both xArm SDK and
# the vendor DOF17_ZWHAND_API.py despite the manual saying 0x3D65.
# ------------------------------------------------------------------
def modbus_crc(data: list[int]) -> list[int]:
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


@contextlib.contextmanager
def _suppress_sdk_noise():
    from xarm.core.utils import log as _sdk_log

    orig = _sdk_log.logger.level
    _sdk_log.logger.setLevel(logging.CRITICAL)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(1)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)
    try:
        yield
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        _sdk_log.logger.setLevel(orig)


def clear(arm):
    try:
        arm.clean_error()
        arm.clean_warn()
        arm.set_state(0)
    except Exception:
        pass


def hex_str(data) -> str:
    if data is None:
        return "<None>"
    return " ".join(f"{b:02X}" for b in data)


# ------------------------------------------------------------------
# Probe helpers
# ------------------------------------------------------------------
def probe_normal(arm, dev_id: int, func_code: int, addr: int, count: int) -> dict:
    """Send a Modbus read in normal (CRC-handled-by-firmware) mode."""
    frame = [dev_id, func_code, 0x00, addr, 0x00, count]
    t0 = time.monotonic()
    try:
        code, res = arm.getset_tgpio_modbus_data(
            frame, min_res_len=0, host_id=TGPIO_HOST_ID,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "elapsed_ms": _ms(t0)}
    elapsed = _ms(t0)
    ok = code == 0 and res is not None and len(res) >= 3
    result = {
        "ok": ok,
        "code": code,
        "res_hex": hex_str(res),
        "res_len": len(res) if res else 0,
        "elapsed_ms": elapsed,
    }
    if not ok:
        clear(arm)
    return result


def probe_transparent(arm, dev_id: int, func_code: int, addr: int, count: int) -> dict:
    """Send a Modbus read in transparent-transmission mode (we add CRC)."""
    frame = [dev_id, func_code, 0x00, addr, 0x00, count]
    frame_with_crc = frame + modbus_crc(frame)
    expected_res_len = 3 + count * 2 + 2  # header + data + CRC
    t0 = time.monotonic()
    try:
        code, res = arm.getset_tgpio_modbus_data(
            frame_with_crc,
            min_res_len=0,
            host_id=TGPIO_HOST_ID,
            is_transparent_transmission=True,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "elapsed_ms": _ms(t0)}
    elapsed = _ms(t0)
    ok = code == 0 and res is not None and len(res) >= 5
    result = {
        "ok": ok,
        "code": code,
        "res_hex": hex_str(res),
        "res_len": len(res) if res else 0,
        "elapsed_ms": elapsed,
    }
    if not ok:
        clear(arm)
    return result


def _ms(t0: float) -> float:
    return round((time.monotonic() - t0) * 1000, 1)


# ------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------
def run_diagnostics(arm):
    sep = "-" * 70
    logger.info(sep)
    logger.info("PHASE 0: xArm tool-end diagnostics")
    logger.info(sep)

    # Tool GPIO firmware version
    try:
        code, ver = arm.get_tgpio_version()
        logger.info("Tool GPIO firmware version: %s (code=%d)", ver, code)
    except Exception as exc:
        logger.error("Could not read tool GPIO version: %s", exc)

    # Current modbus baudrate
    try:
        code, baud = arm.get_tgpio_modbus_baudrate()
        logger.info("Current tool modbus baudrate: %s (code=%d)", baud, code)
    except Exception as exc:
        logger.error("Could not read tool modbus baudrate: %s", exc)

    # Arm error/warn
    try:
        code, (err, warn) = arm.get_err_warn_code()
        logger.info("Arm error=%d, warn=%d (code=%d)", err, warn, code)
    except Exception as exc:
        logger.error("Could not read arm error/warn: %s", exc)

    # Tool GPIO digital
    try:
        code, vals = arm.get_tgpio_digital()
        logger.info("Tool GPIO digital inputs: %s (code=%d)", vals, code)
    except Exception as exc:
        logger.error("Could not read tool GPIO digital: %s", exc)

    # Tool GPIO analog
    try:
        code, vals = arm.get_tgpio_analog()
        logger.info("Tool GPIO analog inputs: %s (code=%d)", vals, code)
    except Exception as exc:
        logger.error("Could not read tool GPIO analog: %s", exc)

    clear(arm)


def run_baud_tests(arm) -> list[dict]:
    sep = "-" * 70
    results = []

    for baud_idx, baud in enumerate(BAUD_RATES):
        logger.info(sep)
        logger.info("PHASE %d: Testing baud rate %d", baud_idx + 1, baud)
        logger.info(sep)

        with _suppress_sdk_noise():
            code = arm.set_tgpio_modbus_baudrate(baud)
        if code != 0:
            logger.error("  Failed to set baudrate %d (code=%d) — skipping", baud, code)
            clear(arm)
            continue
        logger.info("  Baudrate set to %d, waiting 2s for tool MCU reboot …", baud)
        time.sleep(2)

        # Set timeouts for both normal and transparent modes
        with _suppress_sdk_noise():
            arm.set_tgpio_modbus_timeout(200)
            arm.set_tgpio_modbus_timeout(200, is_transparent_transmission=True)
        clear(arm)

        # Verify baud was actually applied
        with _suppress_sdk_noise():
            code2, actual_baud = arm.get_tgpio_modbus_baudrate()
        logger.info("  Readback baudrate: %s (code=%d)", actual_baud, code2)

        for dev_id in DEVICE_IDS:
            for func_code, fc_name in [(0x04, "ReadInputReg"), (0x03, "ReadHoldingReg")]:
                for addr, addr_name in [
                    (INIT_STATE_ADDR, "init_state"),
                    (SW_VER_ADDR, "sw_version"),
                ]:
                    label = f"baud={baud} id={dev_id} {fc_name}(0x{addr:02X}) [{addr_name}]"

                    # Normal mode
                    with _suppress_sdk_noise():
                        r = probe_normal(arm, dev_id, func_code, addr, 1)
                    tag = "OK" if r["ok"] else "FAIL"
                    results.append({"label": f"NORMAL  {label}", **r})
                    logger.info("  [%s] NORMAL  %s  code=%s res=%s (%s ms)",
                                tag, label, r.get("code"), r.get("res_hex"), r.get("elapsed_ms"))

                    # Transparent mode
                    with _suppress_sdk_noise():
                        r = probe_transparent(arm, dev_id, func_code, addr, 1)
                    tag = "OK" if r["ok"] else "FAIL"
                    results.append({"label": f"TRANSP  {label}", **r})
                    logger.info("  [%s] TRANSP  %s  code=%s res=%s (%s ms)",
                                tag, label, r.get("code"), r.get("res_hex"), r.get("elapsed_ms"))

    return results


def print_summary(results: list[dict]):
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    successes = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]

    if successes:
        print(f"\n  {len(successes)} SUCCESSFUL probe(s):")
        for r in successes:
            print(f"    {r['label']}  res={r.get('res_hex')}")
    else:
        print("\n  NO successful probes.")

    print(f"\n  {len(failures)} failed probe(s)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ZWHAND DM17 RS485 diagnostic test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    conn = parser.add_mutually_exclusive_group(required=True)
    conn.add_argument("--ip", type=str, help="xArm IP address")
    conn.add_argument("--arm", type=str, help="Arm name from robot_mapping.json")
    parser.add_argument("-v", "--verbose", action="store_true")
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
        arm = ctrl.arm
        run_diagnostics(arm)
        results = run_baud_tests(arm)
    finally:
        ctrl.disconnect()

    print_summary(results)
    return 0 if any(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
