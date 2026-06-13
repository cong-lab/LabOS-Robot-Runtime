#!/usr/bin/env python3
"""Slim LabOS MCP runner for the AWE vortexing demo."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import concurrent.futures
import contextlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from labos.mcp import Context, RemoteMCP
from labos.mcp.protocol import ToolHeartbeat

LOGGER = logging.getLogger("run_awe_mcp")
LIVEKIT_CAPTURE_TIMEOUT_S = 1.0

_visualizer: "RunVisualizer | None" = None
_livekit_session: "LiveKitCameraSession | None" = None
_livekit_lock: "asyncio.Lock | None" = None
RACK_MODEL: str | None = None
_protocol_lock = threading.Lock()
_protocol_interrupt = threading.Event()


async def _send_ws_payload(ws: Any, payload: dict[str, Any]) -> None:
    await ws.send(json.dumps(payload))


class LoggingRemoteMCP(RemoteMCP):
    """RemoteMCP subclass that logs heartbeat activity for local operators."""

    async def _heartbeat_loop(self, ws: Any) -> None:
        while True:
            await _send_ws_payload(ws, ToolHeartbeat().model_dump(mode="json"))
            LOGGER.info("Heartbeat sent to LabOS relay")
            await asyncio.sleep(self.heartbeat_interval)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _mask_secret_tail(value: str, keep: int = 4) -> str:
    """Mask a secret while keeping only the last few characters visible."""
    if not value:
        return "<unset>"
    if keep <= 0:
        return "*" * len(value)
    visible = value[-keep:] if len(value) > keep else value
    masked_len = max(0, len(value) - len(visible))
    return ("*" * masked_len) + visible


def _load_dotenv_into_environ(path: Path | None = None) -> dict[str, str]:
    """Parse .env and set missing values in os.environ."""
    dotenv_path = Path(path) if path is not None else (ROOT / ".env")
    values: dict[str, str] = {}
    if not dotenv_path.is_file():
        return values
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        key, separator, raw_value = stripped.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key:
            continue
        value = raw_value.strip()
        if (value.startswith("\"") and value.endswith("\"")) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        values[key] = value
        os.environ.setdefault(key, value)
    return values


def _require_labos_settings() -> None:
    """Exit with a clear message if required LabOS settings are missing."""
    missing: list[str] = []
    if not os.environ.get("LABOS_URL"):
        missing.append("LABOS_URL")
    if not os.environ.get("LABOS_API_KEY"):
        missing.append("LABOS_API_KEY")
    if missing:
        joined = ", ".join(missing)
        LOGGER.error(
            "Missing required LabOS settings: %s. Set them in .env, the environment, or via constructor arguments.",
            joined,
        )
        raise SystemExit(2)


def configure_logging(verbose: bool = False) -> None:
    """Configure console + file logging under logs/."""
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "run_awe_mcp.log"
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    LOGGER.info("Logging to %s", log_file)


@dataclass
class LiveKitCameraSession:
    room: Any
    source: Any
    track_name: str
    task: asyncio.Task[None]


def _get_livekit_lock() -> asyncio.Lock:
    global _livekit_lock
    if _livekit_lock is None:
        _livekit_lock = asyncio.Lock()
    return _livekit_lock


def _latest_robot_vision_frame() -> Any | None:
    """Return the latest color-only robot vision frame (BGR), including overlays."""
    try:
        import aira.robot as robot
    except Exception:
        return None
    with robot._vision_display_lock:
        frame = getattr(robot, "_vision_display_last_color_frame", None)
        if frame is None:
            frame = robot._vision_display_last_frame
        return None if frame is None else frame.copy()


async def _publish_robot_vision_frames(source: Any, rtc: Any, width: int, height: int, fps: float) -> None:
    import cv2
    import numpy as np

    period = 1.0 / max(float(fps), 1.0)
    next_frame_time = asyncio.get_running_loop().time()
    blank = None
    frames_published = 0
    saw_robot_frame = False
    consecutive_capture_timeouts = 0
    report_every = max(1, int(max(float(fps), 1.0) * 30))
    LOGGER.info("LiveKit frame publisher started: %dx%d @ %.1f fps", width, height, fps)
    capture_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="livekit-capture",
    )
    try:
        while True:
            frame = _latest_robot_vision_frame()
            if frame is None:
                if blank is None:
                    blank = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(
                        blank,
                        "Waiting for robot camera",
                        (40, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (200, 200, 200),
                        2,
                    )
                    LOGGER.info("LiveKit publisher has no robot vision frame yet; publishing placeholder")
                frame = blank
            elif not saw_robot_frame:
                saw_robot_frame = True
                LOGGER.info(
                    "LiveKit publisher received first robot vision frame: shape=%s",
                    getattr(frame, "shape", None),
                )
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            rgb_bytes = rgb.tobytes()
            video_frame = rtc.VideoFrame(width, height, rtc.VideoBufferType.RGB24, rgb_bytes)
            try:
                loop = asyncio.get_running_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(capture_executor, source.capture_frame, video_frame),
                    timeout=LIVEKIT_CAPTURE_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                consecutive_capture_timeouts += 1
                LOGGER.warning(
                    "LiveKit source.capture_frame timed out (%d consecutive); skipping frame",
                    consecutive_capture_timeouts,
                )
                if consecutive_capture_timeouts >= 3:
                    raise RuntimeError("LiveKit source.capture_frame timed out repeatedly")
                next_frame_time = asyncio.get_running_loop().time()
                await asyncio.sleep(period)
                continue
            consecutive_capture_timeouts = 0
            frames_published += 1
            if frames_published == 1:
                LOGGER.info("LiveKit publisher captured first video frame")
            elif frames_published % report_every == 0:
                LOGGER.info("LiveKit publisher captured %d video frames", frames_published)

            next_frame_time += period
            delay = next_frame_time - asyncio.get_running_loop().time()
            if delay < 0:
                next_frame_time = asyncio.get_running_loop().time()
                delay = 0
            await asyncio.sleep(delay)
    finally:
        capture_executor.shutdown(wait=False, cancel_futures=True)
        LOGGER.info("LiveKit frame publisher stopped after %d frame(s)", frames_published)


def _log_livekit_task_result(task: asyncio.Task[None]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        LOGGER.exception("LiveKit camera publisher stopped unexpectedly")


async def _leave_camera_room_locked() -> str:
    """Stop publishing while the LiveKit session lock is already held."""
    global _livekit_session
    if _livekit_session is None:
        return "No LiveKit camera room is currently joined."
    session = _livekit_session
    _livekit_session = None
    LOGGER.info(
        "Leaving LiveKit camera room: room=%s track=%s",
        getattr(session.room, "name", "unknown"),
        session.track_name,
    )
    session.task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await session.task
    with contextlib.suppress(Exception):
        await session.source.aclose()
    disconnect = getattr(session.room, "disconnect", None)
    if disconnect is not None:
        result = disconnect()
        if asyncio.iscoroutine(result):
            await result
    LOGGER.info("Left LiveKit camera room")
    return "Left LiveKit camera room."


async def leave_camera_room() -> str:
    """Stop publishing the robot camera feed and disconnect from LiveKit."""
    async with _get_livekit_lock():
        return await _leave_camera_room_locked()


async def join_camera_room(
    livekiturl: str,
    joincode: str,
    *,
    track_name: str = "robot-camera",
    width: int = 1280,
    height: int = 720,
    fps: float = 10.0,
    max_bitrate: int = 2_000_000,
) -> str:
    """Join a LiveKit room and publish the rendered robot camera visualization.

    ``joincode`` is expected to be a LiveKit participant JWT/token generated by
    the remote controller for the target room.
    """
    global _livekit_session

    if not livekiturl.strip():
        raise ValueError("livekiturl is required")
    if not joincode.strip():
        raise ValueError("joincode is required")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    async with _get_livekit_lock():
        LOGGER.info(
            "join_camera_room requested: url=%s track=%s size=%dx%d fps=%.1f bitrate=%d",
            livekiturl,
            track_name,
            width,
            height,
            fps,
            max_bitrate,
        )
        replaced_existing = _livekit_session is not None
        if _livekit_session is not None:
            LOGGER.info("Replacing existing LiveKit camera room session")
            await _leave_camera_room_locked()

        from livekit import rtc

        start_camera_and_visualization()
        room = rtc.Room()

        @room.on("participant_connected")
        def _on_participant_connected(participant: Any) -> None:
            LOGGER.info(
                "LiveKit participant connected: identity=%s sid=%s",
                getattr(participant, "identity", "unknown"),
                getattr(participant, "sid", "unknown"),
            )

        @room.on("participant_disconnected")
        def _on_participant_disconnected(participant: Any) -> None:
            LOGGER.info(
                "LiveKit participant disconnected: identity=%s sid=%s",
                getattr(participant, "identity", "unknown"),
                getattr(participant, "sid", "unknown"),
            )

        @room.on("track_published")
        def _on_track_published(publication: Any, participant: Any) -> None:
            LOGGER.info(
                "LiveKit remote track published: participant=%s track_sid=%s track_name=%s",
                getattr(participant, "identity", "unknown"),
                getattr(publication, "sid", "unknown"),
                getattr(publication, "name", "unknown"),
            )

        @room.on("track_subscribed")
        def _on_track_subscribed(track: Any, publication: Any, participant: Any) -> None:
            LOGGER.info(
                "LiveKit track subscribed locally: participant=%s track_sid=%s track_name=%s",
                getattr(participant, "identity", "unknown"),
                getattr(publication, "sid", "unknown"),
                getattr(publication, "name", "unknown"),
            )

        @room.on("disconnected")
        def _on_disconnected(*args: Any) -> None:
            LOGGER.info("LiveKit room disconnected: args=%s", args)

        source = None
        try:
            LOGGER.info("Connecting to LiveKit camera room at %s", livekiturl)
            await room.connect(livekiturl, joincode)
            LOGGER.info(
                "Connected to LiveKit room: room=%s participant_identity=%s participant_sid=%s",
                getattr(room, "name", "unknown"),
                getattr(room.local_participant, "identity", "") or "unknown",
                getattr(room.local_participant, "sid", "") or "unknown",
            )

            source = rtc.VideoSource(width, height)
            track = rtc.LocalVideoTrack.create_video_track(track_name, source)
            options = rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_CAMERA,
                video_encoding=rtc.VideoEncoding(
                    max_framerate=float(fps),
                    max_bitrate=int(max_bitrate),
                ),
            )
            publication = await room.local_participant.publish_track(track, options)
            task = asyncio.create_task(_publish_robot_vision_frames(source, rtc, width, height, fps))
            task.add_done_callback(_log_livekit_task_result)
            _livekit_session = LiveKitCameraSession(
                room=room,
                source=source,
                track_name=track_name,
                task=task,
            )
            participant_identity = getattr(room.local_participant, "identity", "")
            participant_sid = getattr(room.local_participant, "sid", "")
            publication_sid = getattr(publication, "sid", "")
            LOGGER.info(
                "Published LiveKit camera track %s (%s) as participant identity=%s sid=%s",
                track_name,
                publication_sid or "unknown",
                participant_identity or "unknown",
                participant_sid or "unknown",
            )
            return json.dumps({
                "status": "joined",
                "replaced_existing": replaced_existing,
                "room": room.name,
                "track_name": track_name,
                "track_sid": publication_sid,
                "participant_identity": participant_identity,
                "participant_sid": participant_sid,
            })
        except Exception:
            LOGGER.exception("Failed to join/publish LiveKit camera room")
            if source is not None:
                with contextlib.suppress(Exception):
                    await source.aclose()
            disconnect = getattr(room, "disconnect", None)
            if disconnect is not None:
                with contextlib.suppress(Exception):
                    result = disconnect()
                    if asyncio.iscoroutine(result):
                        await result
            raise


class RunVisualizer:
    """Live RealSense + YOLO preview window.

    Opens the local camera viewer immediately on startup. LiveKit publishing is
    controlled separately by join_camera_room()/leave_camera_room().
    """

    def start(self) -> None:
        from aira.robot import start_vision_display

        start_vision_display()
        LOGGER.info("Vision display window started")

    def stop(self) -> None:
        try:
            from aira.robot import _vision_display_stop

            _vision_display_stop.set()
        except Exception:
            pass
        LOGGER.info("Vision display window stopped")


def start_camera_and_visualization() -> "RunVisualizer":
    """Initialize the RealSense camera singleton and start visualization.

    Called once at process startup so the camera and the (future LiveKit)
    visualization are live before any vortexing tool call arrives.
    """
    global _visualizer

    from aira.vision.singletons import camera

    camera()
    if _visualizer is None:
        _visualizer = RunVisualizer()
        _visualizer.start()
    return _visualizer


def set_rack_model(name: str | None) -> str:
    """Set the optional global rack model hint used by vortexing runs."""
    global RACK_MODEL
    cleaned = (name or "").strip()
    RACK_MODEL = cleaned or None
    return f"Rack model set to '{RACK_MODEL}'." if RACK_MODEL else "Rack model disabled."


class ProtocolInterrupted(RuntimeError):
    """Raised when an operator command interrupts the active protocol."""


def _check_interrupted() -> None:
    if _protocol_interrupt.is_set():
        raise ProtocolInterrupted("AWE protocol interrupted by operator command")


def _check_xarm(code: int | None, action: str) -> None:
    if code not in (0, None):
        from aira.robot import XArmFailure

        raise XArmFailure(action, int(code))


def _stop_robot_motion() -> None:
    """Ask xArm to stop the current motion without disconnecting."""
    try:
        from aira.robot import arm

        raw_arm = getattr(arm(), "arm", None)
        if raw_arm is not None and hasattr(raw_arm, "set_state"):
            raw_arm.set_state(4)
            LOGGER.info("Requested xArm stop state")
    except Exception:
        LOGGER.exception("Failed to request xArm stop state")


def interrupt_current_protocol(reason: str) -> None:
    _protocol_interrupt.set()
    LOGGER.warning("Interrupting AWE protocol: %s", reason)
    _stop_robot_motion()


def turn_vortexer_on() -> str:
    from aira.usb_controller import VortexPowerController

    with VortexPowerController() as vortex:
        response = vortex.vortex_on()
    LOGGER.info("Vortexer turned on")
    return response or "Vortexer on."


def turn_vortexer_off() -> str:
    from aira.usb_controller import VortexPowerController

    with VortexPowerController() as vortex:
        response = vortex.vortex_off()
    LOGGER.info("Vortexer turned off")
    return response or "Vortexer off."


def stop() -> str:
    interrupt_current_protocol("stop requested")
    with contextlib.suppress(Exception):
        turn_vortexer_off()
    return "Stop requested; active protocol interrupted."


def enable_manual_mode(on: bool) -> str:
    interrupt_current_protocol(f"manual mode {'on' if on else 'off'} requested")
    from aira.robot import arm

    robot = arm()
    if on:
        robot.set_manual_mode()
        LOGGER.info("Manual mode enabled")
        return "Manual mode enabled."
    robot.set_position_mode()
    LOGGER.info("Manual mode disabled; position mode enabled")
    return "Manual mode disabled; position mode enabled."


def go_home() -> str:
    interrupt_current_protocol("go_home requested")
    from aira.robot import arm

    robot = arm()
    robot.set_position_mode()
    time.sleep(0.2)
    _check_xarm(robot.go_to("awe_home", speed=325, acc=780), "go_home")
    LOGGER.info("Robot moved home")
    return "Robot moved to awe_home."


def _normalize_color(color: str) -> str:
    """Validate a tube color against AVAILABLE_COLORS, returning the normalized name."""
    from awe_demo import AVAILABLE_COLORS

    normalized = color.strip().lower()
    if normalized not in AVAILABLE_COLORS:
        available = ", ".join(sorted(AVAILABLE_COLORS))
        raise ValueError(f"Unsupported tube color '{color}'. Available colors: {available}.")
    return normalized


def _run_vortexing_stages(color: str, rack_model: str | None) -> None:
    """Run the three vortexing stages for one tube. Assumes the protocol lock
    is held and the interrupt has been cleared by the caller."""
    from awe_demo import pickUpTube, placeDownTube, vortexTube

    stages = [
        ("pick up tube", lambda: pickUpTube(color, rack_model=rack_model)),
        ("vortex tube", vortexTube),
        ("place down tube", lambda: placeDownTube(rack_model=rack_model)),
    ]
    for stage_name, stage_fn in stages:
        try:
            _check_interrupted()
            LOGGER.info("AWE stage started: %s (%s tube)", stage_name, color)
            stage_fn()
            _check_interrupted()
            LOGGER.info("AWE stage complete: %s (%s tube)", stage_name, color)
        except Exception:
            LOGGER.exception("AWE stage failed: %s (%s tube)", stage_name, color)
            raise


def run_vortexing(color: str = "red") -> str:
    """Pick up a tube, vortex it, and place it back into the rack."""
    if not _protocol_lock.acquire(blocking=False):
        raise RuntimeError("AWE protocol is already running")
    _protocol_interrupt.clear()
    try:
        normalized_color = _normalize_color(color)

        # Camera + visualization are normally started at process startup; ensure
        # they are live in case run_vortexing is invoked directly.
        start_camera_and_visualization()

        LOGGER.info("Starting AWE vortexing flow for %s tube (rack_model=%s)", normalized_color, RACK_MODEL)
        _run_vortexing_stages(normalized_color, RACK_MODEL)
        LOGGER.info("AWE vortexing flow complete for %s tube", normalized_color)
        return f"Vortexing complete for '{normalized_color}' tube."
    finally:
        _protocol_lock.release()


def run_vortexing_colors(colors: list[str]) -> str:
    """Vortex a list of tubes in order: for each color, pick up, vortex, and
    place it back. Aborts the whole sequence on the first failure (fail-fast)."""
    if not colors:
        raise ValueError("colors must be a non-empty list of tube colors.")

    # Validate every color up front so we never start moving for an invalid batch.
    normalized_colors = [_normalize_color(color) for color in colors]

    if not _protocol_lock.acquire(blocking=False):
        raise RuntimeError("AWE protocol is already running")
    _protocol_interrupt.clear()
    try:
        start_camera_and_visualization()
        LOGGER.info(
            "Starting AWE vortexing sequence for %d tube(s): %s (rack_model=%s)",
            len(normalized_colors),
            ", ".join(normalized_colors),
            RACK_MODEL,
        )
        for index, color in enumerate(normalized_colors, start=1):
            _check_interrupted()
            LOGGER.info("AWE sequence %d/%d: %s tube", index, len(normalized_colors), color)
            _run_vortexing_stages(color, RACK_MODEL)
        LOGGER.info("AWE vortexing sequence complete for %d tube(s)", len(normalized_colors))
        return f"Vortexing complete for {len(normalized_colors)} tube(s): {', '.join(normalized_colors)}."
    finally:
        _protocol_lock.release()


def create_mcp() -> RemoteMCP:
    mcp = LoggingRemoteMCP(
        name="labos-awe-vortexing",
        metadata={
            "runtime": "LabOS-Robot-Runtime",
            "entrypoint": "run_awe_mcp.py",
            "headless": _env_bool("HEADLESS", False),
        },
    )

    @mcp.on_connect
    async def _connected(_ws: Any) -> None:
        LOGGER.info("Connected to LabOS relay")

    @mcp.on_disconnect
    async def _disconnected(_ws: Any) -> None:
        LOGGER.info("Disconnected from LabOS relay")

    @mcp.tool(
        name="run_vortexing",
        description="Run the full AWE vortexing flow: pick up a tube, vortex it, and place it back.",
    )
    async def _run_vortexing_tool(ctx: Context, color: str = "red") -> str:
        await ctx.info("starting AWE vortexing", color=color, rack_model=RACK_MODEL)
        return await asyncio.to_thread(run_vortexing, color)

    @mcp.tool(
        name="run_vortexing_colors",
        description="Run the full AWE vortexing flow for a list of tube colors in order, aborting on the first failure.",
    )
    async def _run_vortexing_colors_tool(ctx: Context, colors: list[str]) -> str:
        await ctx.info("starting AWE vortexing sequence", colors=colors, rack_model=RACK_MODEL)
        return await asyncio.to_thread(run_vortexing_colors, colors)

    @mcp.tool(
        name="set_rack_model",
        description="Set the global optional rack model name used as a geometry hint for future vortexing runs.",
    )
    async def _set_rack_model_tool(ctx: Context, name: str | None = None) -> str:
        await ctx.info("setting AWE rack model", rack_model=name)
        return set_rack_model(name)

    @mcp.tool(
        name="turn_vortexer_on",
        description="Turn the vortexer power controller on.",
    )
    async def _turn_vortexer_on_tool(ctx: Context) -> str:
        await ctx.info("turning vortexer on")
        return await asyncio.to_thread(turn_vortexer_on)

    @mcp.tool(
        name="turn_vortexer_off",
        description="Turn the vortexer power controller off.",
    )
    async def _turn_vortexer_off_tool(ctx: Context) -> str:
        await ctx.info("turning vortexer off")
        return await asyncio.to_thread(turn_vortexer_off)

    @mcp.tool(
        name="go_home",
        description="Interrupt the active protocol and move the robot to awe_home.",
    )
    async def _go_home_tool(ctx: Context) -> str:
        await ctx.info("interrupting protocol and going home")
        return await asyncio.to_thread(go_home)

    @mcp.tool(
        name="enable_manual_mode",
        description="Interrupt the active protocol and enable or disable xArm manual teaching mode.",
    )
    async def _enable_manual_mode_tool(ctx: Context, on: bool) -> str:
        await ctx.info("interrupting protocol and changing manual mode", on=on)
        return await asyncio.to_thread(enable_manual_mode, on)

    @mcp.tool(
        name="stop",
        description="Interrupt the active protocol, stop xArm motion, and turn the vortexer off.",
    )
    async def _stop_tool(ctx: Context) -> str:
        await ctx.info("stopping active AWE protocol")
        return await asyncio.to_thread(stop)

    @mcp.tool(
        name="join_camera_room",
        description="Join a LiveKit room and publish the robot camera visualization video track.",
    )
    async def _join_camera_room_tool(
        ctx: Context,
        livekiturl: str,
        joincode: str,
        track_name: str = "robot-camera",
        width: int = 1280,
        height: int = 720,
        fps: float = 10,
        max_bitrate: int = 2_000_000,
    ) -> str:
        await ctx.info(
            "joining LiveKit camera room",
            livekiturl=livekiturl,
            track_name=track_name,
            width=width,
            height=height,
            fps=fps,
            max_bitrate=max_bitrate,
        )
        LOGGER.info(
            "MCP join_camera_room tool invoked: url=%s track=%s size=%dx%d fps=%.1f bitrate=%d",
            livekiturl,
            track_name,
            width,
            height,
            fps,
            max_bitrate,
        )
        return await join_camera_room(
            livekiturl,
            joincode,
            track_name=track_name,
            width=width,
            height=height,
            fps=fps,
            max_bitrate=max_bitrate,
        )

    @mcp.tool(
        name="leave_camera_room",
        description="Stop publishing the robot camera visualization and leave the current LiveKit room.",
    )
    async def _leave_camera_room_tool(ctx: Context) -> str:
        await ctx.info("leaving LiveKit camera room")
        LOGGER.info("MCP leave_camera_room tool invoked")
        return await leave_camera_room()

    return mcp


def main() -> int:
    parser = argparse.ArgumentParser(description="LabOS AWE vortexing MCP server.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    _load_dotenv_into_environ()
    configure_logging(args.verbose)
    set_rack_model(os.environ.get("AWE_RACK_MODEL"))
    LOGGER.info(
        "Remote MCP config: url=%s device_id=%s api_key=%s",
        os.environ.get("LABOS_URL", "<unset>"),
        os.environ.get("LABOS_DEVICE_ID", "<auto>"),
        _mask_secret_tail(os.environ.get("LABOS_API_KEY", ""), keep=4),
    )
    _require_labos_settings()

    try:
        start_camera_and_visualization()
    except BaseException:
        LOGGER.warning("Failed to start camera/visualization at startup; continuing", exc_info=True)

    try:
        LOGGER.info("Starting LabOS AWE vortexing MCP server")
        create_mcp().run()
        return 0
    except KeyboardInterrupt:
        return 0
    except BaseException:
        LOGGER.exception("AWE vortexing MCP server exited with error")
        return 1
    finally:
        if _visualizer is not None:
            _visualizer.stop()


if __name__ == "__main__":
    raise SystemExit(main())
