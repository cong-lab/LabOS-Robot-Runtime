#!/usr/bin/env python3
"""Mock Remote MCP runner for LabOS Robot Runtime."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["MOCK_MODE"] = "1"
os.environ.setdefault("HEADLESS", "1")

from run_mcp import main


if __name__ == "__main__":
    raise SystemExit(main())
