#!/usr/bin/env python3
"""Report local model weight status under GROUNDED_MODELS_ROOT."""

from __future__ import annotations

import json
import os
import sys

from grounded.config import load_dotenv_project
from grounded.utils.model_paths import model_status


def main() -> int:
    load_dotenv_project()
    report = model_status()
    report["GROUNDED_MODELS_ROOT"] = os.environ.get(
        "GROUNDED_MODELS_ROOT", report["root"]
    )
    print(json.dumps(report, indent=2))
    ready = [m["weights_ready"] for m in report["models"].values()]
    if not any(ready):
        print(
            "\nNo model weights detected. See docs/OFFLINE_MODEL_TRANSFER.md",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
