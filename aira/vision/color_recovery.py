"""VLM-based color-recovery timing.

Streams wrist-camera frames to an OpenAI-compatible vLLM endpoint
(default http://rtxworkstation:8101/v1, model 'vlm' / nvidia/Cosmos3-Nano) and
asks whether a vortexed liquid's color has returned to a registered baseline.

The flow that uses this (see ``awe_demo.vortexAndTimeRecovery``):
  1. position the tube and capture a baseline frame of its *original* color,
  2. vortex to trigger the color change, stop the relay (t0),
  3. poll wrist-cam frames -> VLM until it reports the color reverted,
  4. report elapsed time from t0.

Config via env: COLOR_VLM_URL, COLOR_VLM_MODEL.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.request
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_VLM_URL = os.environ.get("COLOR_VLM_URL", "http://rtxworkstation:8101/v1")
DEFAULT_VLM_MODEL = os.environ.get("COLOR_VLM_MODEL", "vlm")
DEFAULT_MAX_EDGE = 512

# Two-image prompt: image 1 = original color, image 2 = current. Validated to
# return a clean 'true'/'false' from the Cosmos-Nano VLM.
REVERT_PROMPT = (
    "The FIRST image shows the original color of the liquid sample. "
    "The SECOND image shows the liquid sample now. "
    "Has the liquid returned to its original color? "
    "Answer with only one word: true or false."
)


def encode_frame(frame: np.ndarray, max_edge: int = DEFAULT_MAX_EDGE, quality: int = 85) -> str:
    """BGR ndarray -> base64 JPEG, downscaled so the longest edge <= ``max_edge``."""
    h, w = frame.shape[:2]
    scale = min(1.0, float(max_edge) / float(max(h, w)))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("failed to JPEG-encode frame")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class VLMColorJudge:
    """Thin client for the OpenAI-compatible /chat/completions vision endpoint."""

    def __init__(self, url: str = DEFAULT_VLM_URL, model: str = DEFAULT_VLM_MODEL,
                 timeout: float = 20.0):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _chat(self, content, max_tokens: int = 8, temperature: float = 0.0) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.url + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.load(resp)
        return out["choices"][0]["message"]["content"].strip()

    @staticmethod
    def _data_url(b64: str) -> dict:
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

    def has_reverted(self, baseline_b64: str, current_b64: str,
                     prompt: str = REVERT_PROMPT) -> Tuple[bool, str]:
        """Return (reverted, raw_reply). reverted=True when the VLM says the
        current color matches the registered baseline."""
        content = [
            {"type": "text", "text": prompt},
            self._data_url(baseline_b64),
            self._data_url(current_b64),
        ]
        reply = self._chat(content)
        verdict = reply.strip().lower()
        reverted = verdict.startswith("true") or verdict.startswith("yes")
        return reverted, reply


def wait_for_color_recovery(
    get_frame: Callable[[], Optional[np.ndarray]],
    baseline_frame: np.ndarray,
    *,
    t0: Optional[float] = None,
    judge: Optional[VLMColorJudge] = None,
    poll_interval: float = 0.5,
    debounce: int = 2,
    timeout: float = 120.0,
    on_sample: Optional[Callable[[float, bool, str], None]] = None,
) -> Tuple[bool, float]:
    """Poll frames until the VLM reports the color reverted to baseline for
    ``debounce`` consecutive samples.

    Returns ``(recovered, elapsed_s)`` measured from ``t0`` (defaults to now).
    On timeout returns ``(False, elapsed_s)`` so the caller can still place the
    tube back safely.
    """
    judge = judge or VLMColorJudge()
    baseline_b64 = encode_frame(baseline_frame)
    if t0 is None:
        t0 = time.monotonic()
    consecutive = 0
    while True:
        elapsed = time.monotonic() - t0
        if elapsed > timeout:
            logger.warning("color recovery timed out after %.1fs", elapsed)
            return False, elapsed
        frame = get_frame()
        if frame is None:
            time.sleep(poll_interval)
            continue
        try:
            reverted, reply = judge.has_reverted(baseline_b64, encode_frame(frame))
        except Exception as exc:  # network/endpoint hiccup -> retry, don't abort
            logger.warning("VLM judge call failed: %s", exc)
            time.sleep(poll_interval)
            continue
        if on_sample is not None:
            on_sample(elapsed, reverted, reply)
        logger.info("color recovery poll t=%.1fs reverted=%s reply=%r", elapsed, reverted, reply)
        if reverted:
            consecutive += 1
            if consecutive >= debounce:
                return True, time.monotonic() - t0
        else:
            consecutive = 0
        time.sleep(poll_interval)
