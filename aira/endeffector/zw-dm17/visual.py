"""
Viser-based interactive 3D visual editor for the ZWHAND DM17-V6 hand.

Provides a browser-accessible URDF visualizer with:
  - Per-joint sliders (0-1000 motor steps)
  - Fingertip IK drag gizmos
  - Per-joint rotation rings
  - Tier-linked 4-finger group controls
  - Optional live synchronisation with a physical hand
"""

from __future__ import annotations

import math
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_MODULE_DIR = Path(__file__).resolve().parent
_ROBOT_ROOT = _MODULE_DIR.parent.parent.parent
_DEFAULT_URDF_PATH = (
    _ROBOT_ROOT / "deps" / "ZWHAND-DM17" / "URDF" / "urdf" / "src"
    / "zwhand_17dof_left" / "urdf" / "zwhand_17dof_left.urdf"
)

# ── Hand topology constants ──────────────────────────────────────────

FINGER_LABELS: Dict[str, List[int]] = {
    "Thumb":  [16, 3, 2, 1],
    "Index":  [17, 6, 5, 4],
    "Middle": [9, 8, 7],
    "Ring":   [12, 11, 10],
    "Little": [15, 14, 13],
}

JOINT_NICE_NAMES: Dict[int, str] = {
    16: "Thumb Spread", 3: "Thumb MCP", 2: "Thumb IP", 1: "Thumb Tip",
    17: "Index Spread", 6: "Index MCP", 5: "Index PIP", 4: "Index DIP",
    9: "Middle MCP", 8: "Middle PIP", 7: "Middle DIP",
    12: "Ring MCP", 11: "Ring PIP", 10: "Ring DIP",
    15: "Little MCP", 14: "Little PIP", 13: "Little DIP",
}

GIZMO_FINGERS = [
    ("Thumb",  [0, 1, 2, 3], 3,  (230, 80, 80)),
    ("Index",  [4, 5, 6, 7], 7,  (80, 200, 80)),
    ("Middle", [8, 9, 10],   10, (80, 120, 230)),
    ("Ring",   [11, 12, 13], 13, (220, 200, 60)),
    ("Little", [14, 15, 16], 16, (200, 80, 220)),
]

TIERS = [
    ("MCP", [5, 8, 11, 14], [6, 9, 12, 15], (255, 180, 80)),
    ("PIP", [6, 9, 12, 15], [5, 8, 11, 14], (180, 255, 80)),
    ("DIP", [7, 10, 13, 16], [4, 7, 10, 13], (80, 200, 255)),
]


# ── Quaternion helpers (wxyz convention) ─────────────────────────────

def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )


def _qinv(q):
    return (q[0], -q[1], -q[2], -q[3])


def _qz(angle: float):
    return (math.cos(angle * 0.5), 0.0, 0.0, math.sin(angle * 0.5))


def _align_z_to(axis):
    a = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(a)
    if n < 1e-8:
        return (1.0, 0.0, 0.0, 0.0)
    a = a / n
    dot = a[2]
    if dot > 0.99999:
        return (1.0, 0.0, 0.0, 0.0)
    if dot < -0.99999:
        return (0.0, 1.0, 0.0, 0.0)
    cross = np.array([-a[1], a[0], 0.0])
    w = 1.0 + dot
    q = np.array([w, cross[0], cross[1], cross[2]])
    q /= np.linalg.norm(q)
    return tuple(q)


# ── URDF helper ──────────────────────────────────────────────────────

def _fix_urdf_for_pybullet(urdf_path: Path) -> str:
    pkg_dir = urdf_path.parent.parent
    text = urdf_path.read_text()
    text = text.replace("package://zwhand_17dof_left/", str(pkg_dir) + "/")
    fd, tmp = tempfile.mkstemp(suffix=".urdf")
    os.write(fd, text.encode())
    os.close(fd)
    return tmp


# ── Main visual editor ──────────────────────────────────────────────


def dm17_visual_edit(
    ee_ctrl=None,
    start_angles: Optional[List[int]] = None,
    port: int = 8080,
    urdf_path: Optional[Path] = None,
) -> Optional[List[int]]:
    """
    Open an interactive viser-based 3D editor for the DM17 hand.

    Parameters
    ----------
    ee_ctrl : ZWDM17XArmController, optional
        Connected controller for live read/send. ``None`` for offline editing.
    start_angles : list[int], optional
        Initial 17-element motor positions (0-1000) to load.
    port : int
        Viser HTTP/WebSocket port (default 8080).
    urdf_path : Path, optional
        Override the default URDF file path.

    Returns
    -------
    list[int] or None
        17-element list of motor positions on save, ``None`` on cancel.
    """
    import pybullet as pb
    import viser
    import yourdfpy
    from scipy.optimize import minimize as sp_minimize
    from viser.extras import ViserUrdf

    urdf_path = urdf_path or _DEFAULT_URDF_PATH
    if not urdf_path.exists():
        print(f"Error: URDF not found at {urdf_path}")
        return None

    # ── Viser scene ──────────────────────────────────────────────────
    pkg_dir = str(urdf_path.parent.parent)

    def _pkg_handler(fname):
        return fname.replace("package://zwhand_17dof_left/", pkg_dir + "/")

    urdf = yourdfpy.URDF.load(str(urdf_path), filename_handler=_pkg_handler)

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")

    viser_urdf = ViserUrdf(server, urdf_or_path=urdf)
    limits = viser_urdf.get_actuated_joint_limits()
    joint_names = list(limits.keys())
    dm17_indices = [int(re.search(r"\d+", n).group()) for n in joint_names]

    # ── PyBullet (headless FK / IK) ──────────────────────────────────
    pb_cid = pb.connect(pb.DIRECT)
    fixed_urdf = _fix_urdf_for_pybullet(urdf_path)
    try:
        pb_hand = pb.loadURDF(fixed_urdf, useFixedBase=True, physicsClientId=pb_cid)
    finally:
        os.unlink(fixed_urdf)

    pb_nj = pb.getNumJoints(pb_hand, physicsClientId=pb_cid)
    pb_lower: list[float] = []
    pb_upper: list[float] = []
    pb_to_dm17: list[int] = []
    pb_axis_local: list[tuple] = []
    for i in range(pb_nj):
        info = pb.getJointInfo(pb_hand, i, physicsClientId=pb_cid)
        pb_to_dm17.append(int(re.search(r"\d+", info[1].decode()).group()))
        pb_lower.append(info[8])
        pb_upper.append(info[9])
        pb_axis_local.append(info[13])

    def _m2r(motor, lo, hi):
        return lo + (float(motor) / 1000.0) * (hi - lo)

    def _r2m(rad, lo, hi):
        rng = hi - lo
        return max(0, min(1000, int(round((rad - lo) / rng * 1000)))) if rng > 1e-8 else 0

    _pb_color: dict[int, tuple[int, int, int]] = {}
    for _fn, _jts, _tl, _fc in GIZMO_FINGERS:
        for j in _jts:
            _pb_color[j] = _fc

    # ── Shared flags ─────────────────────────────────────────────────
    _suppress = False
    _dragging_tip: set = set()
    _dragging_jrot: set = set()
    _dragging_tier: set = set()

    motor_sliders: dict[int, viser.GuiInputHandle] = {}
    all_slider_handles: list[tuple[int, viser.GuiInputHandle]] = []
    tip_gizmos: dict = {}
    jrot_gizmos: dict = {}
    jrot_base_orn: dict = {}
    jball_handles: dict = {}
    tier_handles: dict = {}
    tier_line_handles: dict = {}
    tier_line_names: dict = {}

    # ── GUI: Motor sliders ───────────────────────────────────────────
    for finger_name, dm17_ids in FINGER_LABELS.items():
        with server.gui.add_folder(finger_name):
            for dm17_id in dm17_ids:
                nice = JOINT_NICE_NAMES.get(dm17_id, f"Joint {dm17_id}")
                s = server.gui.add_slider(label=nice, min=0, max=1000, step=1, initial_value=0)
                motor_sliders[dm17_id] = s
                all_slider_handles.append((dm17_id, s))

    # ── GUI: Presets ─────────────────────────────────────────────────
    preset_values = {
        "Open (all 0)": [0] * 17,
        "Closed": [1000 if dm17 not in (3, 16) else 0 for dm17 in range(1, 18)],
        "Half": [500 if dm17 != 16 else 0 for dm17 in range(1, 18)],
    }
    with server.gui.add_folder("Presets"):
        for pname, pangles in preset_values.items():
            pbtn = server.gui.add_button(pname)

            def _mk_preset(a):
                return lambda _: _batch({dm17: a[dm17 - 1] for dm17 in range(1, 18)})

            pbtn.on_click(_mk_preset(pangles))

    # ── GUI: Group control (4-finger tiers) ──────────────────────────
    with server.gui.add_folder("4-Finger Group"):
        tier_sliders: dict[str, viser.GuiInputHandle] = {}
        for tname, _, tdm17, _ in TIERS:
            ts = server.gui.add_slider(label=f"{tname} Grip", min=0, max=1000, step=1, initial_value=0)
            tier_sliders[tname] = ts

            def _mk_tier_slider_cb(tn, dm17s):
                def cb(_):
                    if _suppress:
                        return
                    val = int(tier_sliders[tn].value)
                    _batch({d: val for d in dm17s})
                return cb

            ts.on_update(_mk_tier_slider_cb(tname, tdm17))

    # ── GUI: Robot controls ──────────────────────────────────────────
    auto_send_cb = None
    if ee_ctrl is not None:
        with server.gui.add_folder("Robot"):
            auto_send_cb = server.gui.add_checkbox("Auto-send to hand", initial_value=True)
            server.gui.add_button("Read from hand").on_click(lambda _: _read_from_hand())
            server.gui.add_button("Send to hand").on_click(lambda _: _send_to_hand())

    # ── GUI: Display toggles ─────────────────────────────────────────
    with server.gui.add_folder("Display"):
        show_balls_cb = server.gui.add_checkbox("Joint balls", initial_value=True)
        show_jrot_cb = server.gui.add_checkbox("Joint rotation rings", initial_value=True)
        show_tiers_cb = server.gui.add_checkbox("Tier lines & handles", initial_value=True)
        show_tips_cb = server.gui.add_checkbox("Fingertip IK gizmos", initial_value=True)

        def _toggle_vis(_):
            for h in jball_handles.values():
                h.visible = show_balls_cb.value
            for g in jrot_gizmos.values():
                g.visible = show_jrot_cb.value
            for tn in tier_line_handles:
                tier_line_handles[tn].visible = show_tiers_cb.value
                if tn in tier_handles:
                    tier_handles[tn].visible = show_tiers_cb.value
            for g in tip_gizmos.values():
                g.visible = show_tips_cb.value

        show_balls_cb.on_update(_toggle_vis)
        show_jrot_cb.on_update(_toggle_vis)
        show_tiers_cb.on_update(_toggle_vis)
        show_tips_cb.on_update(_toggle_vis)

    save_event = threading.Event()
    cancel_event = threading.Event()
    server.gui.add_button("Save Position", color="green").on_click(lambda _: save_event.set())

    # ── Core helpers ─────────────────────────────────────────────────

    def _get_motor_positions() -> List[int]:
        return [
            max(0, min(1000, int(motor_sliders[d].value))) if d in motor_sliders else 0
            for d in range(1, 18)
        ]

    def _read_from_hand():
        if ee_ctrl is None:
            return
        try:
            angles = ee_ctrl.get_angles()
            if angles is False:
                return
            _batch({d: max(0, min(1000, int(angles[d - 1]))) for d in range(1, 18)})
        except Exception as e:
            print(f"Read error: {e}")

    def _send_to_hand():
        if ee_ctrl is None:
            return
        try:
            ee_ctrl.set_all_absolute([max(0, min(1000, p)) for p in _get_motor_positions()])
        except Exception as e:
            print(f"Send error: {e}")

    def _sync_urdf():
        cfg = np.zeros(len(joint_names))
        for i, name in enumerate(joint_names):
            d = dm17_indices[i]
            lo, hi = limits[name]
            lo = lo if lo is not None else 0.0
            hi = hi if hi is not None else 1.5708
            cfg[i] = lo + (float(motor_sliders[d].value) / 1000.0) * (hi - lo) if d in motor_sliders else lo
        viser_urdf.update_cfg(cfg)

    def _sync_pb():
        for j in range(pb_nj):
            d = pb_to_dm17[j]
            mv = float(motor_sliders[d].value) if d in motor_sliders else 0.0
            pb.resetJointState(pb_hand, j, _m2r(mv, pb_lower[j], pb_upper[j]), physicsClientId=pb_cid)

    def _update_tip_gizmos(skip=None):
        for fn, _jts, tl, _ in GIZMO_FINGERS:
            if fn == skip or fn not in tip_gizmos:
                continue
            tip_gizmos[fn].position = pb.getLinkState(pb_hand, tl, physicsClientId=pb_cid)[0]

    def _update_joint_visuals(skip_jrot=None, skip_tier=None):
        for j in range(pb_nj):
            ls = pb.getLinkState(pb_hand, j, physicsClientId=pb_cid)
            pos = ls[4]
            orn_xyzw = ls[5]

            if j in jball_handles:
                jball_handles[j].position = pos

            if j in jrot_gizmos and j not in _dragging_jrot and j != skip_jrot:
                R = np.array(pb.getMatrixFromQuaternion(orn_xyzw)).reshape(3, 3)
                axis_w = R @ np.array(pb_axis_local[j])
                base = _align_z_to(axis_w)
                jrot_base_orn[j] = base
                d = pb_to_dm17[j]
                mv = float(motor_sliders[d].value) if d in motor_sliders else 0.0
                cur_rad = _m2r(mv, pb_lower[j], pb_upper[j])
                gizmo_q = _qmul(base, _qz(cur_rad))
                jrot_gizmos[j].position = pos
                jrot_gizmos[j].wxyz = gizmo_q

        for tname, pb_links, _, tcolor in TIERS:
            positions = [np.array(pb.getLinkState(pb_hand, l, physicsClientId=pb_cid)[4]) for l in pb_links]
            n = len(positions)
            if n >= 2 and tname in tier_line_names:
                pts = np.array([[positions[i], positions[i + 1]] for i in range(n - 1)])
                colors = np.array([[tcolor, tcolor]] * (n - 1), dtype=np.uint8)
                server.scene.add_line_segments(
                    tier_line_names[tname], points=pts, colors=colors, line_width=3,
                )
            if tname in tier_handles and tname != skip_tier and tname not in _dragging_tier:
                centroid = np.mean(positions, axis=0)
                tier_handles[tname].position = tuple(centroid)

    def _batch(values: dict[int, int], skip_tip=None, skip_jrot=None, skip_tier=None):
        nonlocal _suppress
        _suppress = True
        for d, v in values.items():
            if d in motor_sliders:
                motor_sliders[d].value = max(0, min(1000, int(v)))
        _suppress = False
        _sync_urdf()
        _sync_pb()
        _update_tip_gizmos(skip=skip_tip)
        _update_joint_visuals(skip_jrot=skip_jrot, skip_tier=skip_tier)
        if auto_send_cb is not None and auto_send_cb.value:
            _send_to_hand()

    # ── IK solver ────────────────────────────────────────────────────

    def _solve_finger_ik(target_pos, pb_joints, tip_link):
        x0 = np.array([
            _m2r(
                float(motor_sliders[pb_to_dm17[j]].value) if pb_to_dm17[j] in motor_sliders else 0,
                pb_lower[j], pb_upper[j],
            )
            for j in pb_joints
        ])
        bounds = [(pb_lower[j], pb_upper[j]) for j in pb_joints]
        tgt = np.asarray(target_pos, dtype=np.float64)

        def cost(x):
            for ji, j in enumerate(pb_joints):
                pb.resetJointState(pb_hand, j, float(x[ji]), physicsClientId=pb_cid)
            fk = np.array(pb.getLinkState(pb_hand, tip_link, physicsClientId=pb_cid)[0])
            return float(np.sum((fk - tgt) ** 2))

        return sp_minimize(cost, x0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 40, "ftol": 1e-10}).x

    # ── Slider callback ──────────────────────────────────────────────

    def _on_slider(_):
        if _suppress:
            return
        _sync_urdf()
        _sync_pb()
        if not _dragging_tip:
            _update_tip_gizmos()
        if not _dragging_jrot:
            _update_joint_visuals()
        if auto_send_cb is not None and auto_send_cb.value:
            _send_to_hand()

    for _, sl in all_slider_handles:
        sl.on_update(_on_slider)

    # ── Fingertip IK gizmos ──────────────────────────────────────────
    _sync_pb()

    for fn, pjts, tl, fc in GIZMO_FINGERS:
        pos = pb.getLinkState(pb_hand, tl, physicsClientId=pb_cid)[0]
        g = server.scene.add_transform_controls(
            f"/tip_{fn}", scale=0.015,
            disable_rotations=True, position=pos, depth_test=False,
        )
        server.scene.add_icosphere(f"/tip_{fn}/sphere", radius=0.004, color=fc)
        tip_gizmos[fn] = g

        def _mk_ds(n):
            return lambda _: _dragging_tip.add(n)

        def _mk_de(n):
            def cb(_):
                _dragging_tip.discard(n)
                _sync_pb()
                _update_tip_gizmos()
                _update_joint_visuals()
            return cb

        def _mk_du(n, jts, link):
            def cb(ev):
                if n not in _dragging_tip:
                    return
                angles = _solve_finger_ik(np.array(ev.target.position), jts, link)
                _batch(
                    {pb_to_dm17[j]: _r2m(a, pb_lower[j], pb_upper[j]) for j, a in zip(jts, angles)},
                    skip_tip=n,
                )
            return cb

        g.on_drag_start(_mk_ds(fn))
        g.on_drag_end(_mk_de(fn))
        g.on_update(_mk_du(fn, pjts, tl))

    # ── Joint balls ──────────────────────────────────────────────────
    for j in range(pb_nj):
        pos = pb.getLinkState(pb_hand, j, physicsClientId=pb_cid)[4]
        col = _pb_color.get(j, (180, 180, 180))
        jball_handles[j] = server.scene.add_icosphere(f"/jball_{j}", radius=0.003, color=col, position=pos)

    # ── Joint rotation gizmos ────────────────────────────────────────
    for j in range(pb_nj):
        ls = pb.getLinkState(pb_hand, j, physicsClientId=pb_cid)
        pos = ls[4]
        orn_xyzw = ls[5]
        R = np.array(pb.getMatrixFromQuaternion(orn_xyzw)).reshape(3, 3)
        axis_w = R @ np.array(pb_axis_local[j])
        base = _align_z_to(axis_w)
        jrot_base_orn[j] = base

        d = pb_to_dm17[j]
        mv = float(motor_sliders[d].value) if d in motor_sliders else 0.0
        cur_rad = _m2r(mv, pb_lower[j], pb_upper[j])
        gizmo_q = _qmul(base, _qz(cur_rad))

        col = _pb_color.get(j, (180, 180, 180))
        g = server.scene.add_transform_controls(
            f"/jrot_{j}", scale=0.008,
            disable_axes=True, disable_sliders=True,
            rotation_limits=((-0.001, 0.001), (-0.001, 0.001), (-10.0, 10.0)),
            wxyz=gizmo_q, position=pos, depth_test=False, opacity=0.6,
        )
        server.scene.add_icosphere(f"/jrot_{j}/dot", radius=0.0015, color=col)
        jrot_gizmos[j] = g

        def _mk_jr_ds(idx):
            return lambda _: _dragging_jrot.add(idx)

        def _mk_jr_de(idx):
            def cb(_):
                _dragging_jrot.discard(idx)
                _sync_pb()
                _update_joint_visuals()
            return cb

        def _mk_jr_du(idx):
            def cb(ev):
                if idx not in _dragging_jrot:
                    return
                base_q = jrot_base_orn[idx]
                rel = _qmul(_qinv(base_q), tuple(ev.target.wxyz))
                z_angle = 2.0 * math.atan2(rel[3], rel[0])
                z_angle = max(pb_lower[idx], min(pb_upper[idx], z_angle))
                dm = pb_to_dm17[idx]
                _batch({dm: _r2m(z_angle, pb_lower[idx], pb_upper[idx])}, skip_jrot=idx)
            return cb

        g.on_drag_start(_mk_jr_ds(j))
        g.on_drag_end(_mk_jr_de(j))
        g.on_update(_mk_jr_du(j))

    # ── Tier lines & handles ─────────────────────────────────────────
    tier_open: dict = {}
    tier_flex_dir: dict = {}
    tier_flex_len: dict = {}

    _saved = {pb_to_dm17[j]: motor_sliders[pb_to_dm17[j]].value for j in range(pb_nj)}

    for j in range(pb_nj):
        pb.resetJointState(pb_hand, j, pb_lower[j], physicsClientId=pb_cid)
    for tn, pbl, _, _ in TIERS:
        tier_open[tn] = np.mean([
            np.array(pb.getLinkState(pb_hand, l, physicsClientId=pb_cid)[4]) for l in pbl
        ], axis=0)

    for j in range(pb_nj):
        pb.resetJointState(pb_hand, j, pb_upper[j], physicsClientId=pb_cid)
    for tn, pbl, _, _ in TIERS:
        closed_c = np.mean([
            np.array(pb.getLinkState(pb_hand, l, physicsClientId=pb_cid)[4]) for l in pbl
        ], axis=0)
        vec = closed_c - tier_open[tn]
        length = float(np.linalg.norm(vec))
        tier_flex_dir[tn] = vec / max(length, 1e-8)
        tier_flex_len[tn] = length

    for j in range(pb_nj):
        d = pb_to_dm17[j]
        pb.resetJointState(pb_hand, j, _m2r(float(_saved.get(d, 0)), pb_lower[j], pb_upper[j]),
                           physicsClientId=pb_cid)

    for tn, pbl, tdm17, tcol in TIERS:
        positions = [np.array(pb.getLinkState(pb_hand, l, physicsClientId=pb_cid)[4]) for l in pbl]
        n = len(positions)

        lname = f"/tier_line_{tn}"
        if n >= 2:
            pts = np.array([[positions[i], positions[i + 1]] for i in range(n - 1)])
            colors = np.array([[tcol, tcol]] * (n - 1), dtype=np.uint8)
            tier_line_handles[tn] = server.scene.add_line_segments(
                lname, points=pts, colors=colors, line_width=3,
            )
        tier_line_names[tn] = lname

        centroid = np.mean(positions, axis=0)
        th = server.scene.add_transform_controls(
            f"/tier_{tn}", scale=0.012,
            disable_rotations=True, position=tuple(centroid), depth_test=False,
        )
        server.scene.add_icosphere(f"/tier_{tn}/sphere", radius=0.005, color=tcol)
        tier_handles[tn] = th

        def _mk_td_ds(name):
            return lambda _: _dragging_tier.add(name)

        def _mk_td_de(name):
            def cb(_):
                _dragging_tier.discard(name)
                _sync_pb()
                _update_joint_visuals()
            return cb

        def _mk_td_du(name, dm17s):
            def cb(ev):
                if name not in _dragging_tier:
                    return
                pos_now = np.array(ev.target.position)
                proj = float(np.dot(pos_now - tier_open[name], tier_flex_dir[name]))
                motor = max(0, min(1000, int(round(proj / max(tier_flex_len[name], 1e-8) * 1000))))
                _batch({d: motor for d in dm17s}, skip_tier=name)
            return cb

        th.on_drag_start(_mk_td_ds(tn))
        th.on_drag_end(_mk_td_de(tn))
        th.on_update(_mk_td_du(tn, tdm17))

    # ── Initial state ────────────────────────────────────────────────
    if start_angles is not None and len(start_angles) >= 17:
        _batch({dm17: start_angles[dm17 - 1] for dm17 in range(1, 18)})
    elif ee_ctrl is not None:
        _read_from_hand()
    else:
        _sync_urdf()
        _sync_pb()
        _update_tip_gizmos()
        _update_joint_visuals()

    print(f"\n  Viser UI running at http://localhost:{port}")
    print("  Open the URL in your browser.")
    print("  - Drag colored fingertip spheres for IK finger posing")
    print("  - Rotate joint rings for per-joint control")
    print("  - Drag tier handles to move 4 fingers together")
    print("  - Click 'Save Position' when ready.\n")

    try:
        save_event.wait()
    except KeyboardInterrupt:
        print("\nCancelled.")
        pb.disconnect(pb_cid)
        return None

    result = _get_motor_positions()

    if ee_ctrl is not None:
        _send_to_hand()

    pb.disconnect(pb_cid)
    return result
