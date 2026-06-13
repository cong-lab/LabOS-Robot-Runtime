#!/usr/bin/env python3
"""Wrist-cam capture + VLM reliability check for the color-recovery experiment.

Opens a live wrist-camera (RealSense) preview window so you can frame the tube,
snap a "before" (original color) and an "after" (vortexed / changed color) photo,
then run the same VLM judge the recovery timer uses against those two real images
to see whether it reliably tells them apart.

Usage:
  # Interactive: live window + capture + judge
  python scripts/capture_color_test.py
  python scripts/capture_color_test.py --outdir output/color_test

  # Non-interactive: judge two images you already have
  python scripts/capture_color_test.py --judge before.jpg after.jpg

Keys in the window:
  b : save current frame as <outdir>/before.jpg  (original color)
  a : save current frame as <outdir>/after.jpg   (changed color)
  SPACE : save a numbered snapshot snap_NN.jpg
  t : run the VLM reliability test on before.jpg + after.jpg
  q / ESC : quit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aira.vision.color_recovery import VLMColorJudge, encode_frame  # noqa: E402

WINDOW = "Wrist cam - color test (b=before  a=after  SPACE=snap  t=test  q=quit)"


def _name_color(judge: VLMColorJudge, b64: str) -> str:
    content = [
        {"type": "text",
         "text": "What is the single dominant color of the liquid/sample in this image? Reply with one or two words only."},
        judge._data_url(b64),
    ]
    try:
        return judge._chat(content, max_tokens=12)
    except Exception as exc:
        return f"<error: {exc}>"


def run_judge(before_path: Path, after_path: Path, repeats: int = 3) -> bool:
    """Feed two real images to the VLM and report whether it separates them reliably."""
    before = cv2.imread(str(before_path))
    after = cv2.imread(str(after_path))
    if before is None or after is None:
        print(f"ERROR: could not read {before_path} and/or {after_path}")
        return False

    judge = VLMColorJudge()
    b_before = encode_frame(before)
    b_after = encode_frame(after)

    print("\n--- color naming ---")
    print(f"  before.jpg dominant color: {_name_color(judge, b_before)!r}")
    print(f"  after.jpg  dominant color: {_name_color(judge, b_after)!r}")

    print("\n--- revert judgments (image1=original, image2=current) ---")

    def judge_n(label, base_b64, cur_b64, expect):
        verdicts = []
        for _ in range(repeats):
            reverted, reply = judge.has_reverted(base_b64, cur_b64)
            verdicts.append(reverted)
        agree = all(v == verdicts[0] for v in verdicts)
        ok = verdicts[0] is expect and agree
        flag = "OK " if ok else "!! "
        print(f"  {flag}{label}: {verdicts}  (expected {expect}, stable={agree})")
        return ok

    r1 = judge_n("before vs before  -> reverted?", b_before, b_before, True)
    r2 = judge_n("before vs after   -> reverted?", b_before, b_after, False)
    r3 = judge_n("after  vs before   -> reverted?", b_after, b_before, False)

    reliable = r1 and r2 and r3
    print("\n=== RESULT:", "RELIABLE — VLM clearly separates the two colors ===" if reliable
          else "UNRELIABLE — VLM did not cleanly/consistently separate the colors ===")
    if not reliable:
        print("  Consider: better framing of the liquid (not the cap), steadier lighting,"
              " a larger color difference, or a tuned prompt.")
    return reliable


def run_interactive(outdir: Path) -> None:
    from aira.vision.singletons import camera

    outdir.mkdir(parents=True, exist_ok=True)
    cam = camera()  # main-thread init; no other camera reader running
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    before_path = outdir / "before.jpg"
    after_path = outdir / "after.jpg"
    snap_n = 0
    status = "Frame the tube. b=before, a=after, then t=test."

    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            cv2.waitKey(30)
            continue
        disp = frame.copy()
        cv2.putText(disp, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        have = []
        if before_path.exists():
            have.append("before")
        if after_path.exists():
            have.append("after")
        cv2.putText(disp, "saved: " + (", ".join(have) if have else "none"),
                    (10, disp.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(WINDOW, disp)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("b"):
            cv2.imwrite(str(before_path), frame)
            status = f"Saved BEFORE -> {before_path}"
            print(status)
        elif key == ord("a"):
            cv2.imwrite(str(after_path), frame)
            status = f"Saved AFTER  -> {after_path}"
            print(status)
        elif key == ord(" "):
            p = outdir / f"snap_{snap_n:02d}.jpg"
            cv2.imwrite(str(p), frame)
            snap_n += 1
            status = f"Saved snapshot -> {p}"
            print(status)
        elif key == ord("t"):
            if before_path.exists() and after_path.exists():
                print("\nRunning VLM reliability test on captured images...")
                run_judge(before_path, after_path)
                status = "Test done (see terminal). Re-capture and press t to retry."
            else:
                status = "Capture BOTH before (b) and after (a) first."
                print(status)

    cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="Wrist-cam capture + VLM color reliability check.")
    parser.add_argument("--outdir", default="output/color_test",
                        help="Where before.jpg/after.jpg/snapshots are saved.")
    parser.add_argument("--judge", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="Skip the window; just run the VLM test on two existing images.")
    args = parser.parse_args()

    if args.judge:
        ok = run_judge(Path(args.judge[0]), Path(args.judge[1]))
        return 0 if ok else 1

    run_interactive(Path(args.outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
