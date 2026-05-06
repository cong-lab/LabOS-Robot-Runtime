"""
Abstract base class for robot end-effectors.

Every end-effector (gripper, dexterous hand, vacuum tool, etc.) should
subclass :class:`EndEffector` so that the rest of the stack can record,
replay, and swap end-effectors without knowing implementation details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class EndEffector(ABC):
    """Hardware-agnostic interface for a robot end-effector."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this end-effector type (e.g. ``"zw-dm17"``)."""

    @property
    @abstractmethod
    def num_joints(self) -> int:
        """Number of independently controllable joints / DOFs."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def connect(self) -> bool:
        """Initialise hardware and verify the device is ready."""

    @abstractmethod
    def disconnect(self) -> bool:
        """Release hardware resources gracefully."""

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    @abstractmethod
    def calibrate(self) -> bool:
        """Run zero-position (home) calibration for all joints."""

    @abstractmethod
    def stop(self) -> bool:
        """Emergency-stop all actuators immediately."""

    # ------------------------------------------------------------------
    # State serialisation (PyTorch-style)
    # ------------------------------------------------------------------

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """
        Snapshot the current end-effector state as a plain dict.

        The dict must be JSON-serialisable so it can be saved alongside
        arm poses during recording sessions.  At minimum it should contain
        ``{"type": "<name>", ...}``.
        """

    @abstractmethod
    def load_state_dict(self, state: Dict[str, Any]) -> bool:
        """
        Restore the end-effector to a previously captured state.

        Returns True if the command was accepted by the hardware.
        """

    # ------------------------------------------------------------------
    # Visual editing
    # ------------------------------------------------------------------

    def visual_edit(
        self,
        start_angles: Optional[List[int]] = None,
        port: int = 8080,
    ) -> Optional[Dict[str, Any]]:
        """
        Open an interactive visual editor for setting joint positions.

        Subclasses that support a 3-D visual editor (e.g. viser-based URDF
        viewer) should override this method.

        Parameters
        ----------
        start_angles : list[int], optional
            Initial joint positions to load into the editor.
        port : int
            Network port for the visual server (default 8080).

        Returns
        -------
        dict or None
            A ``state_dict``-compatible dict on save, or ``None`` if the
            user cancelled.
        """
        return None
