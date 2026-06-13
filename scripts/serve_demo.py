#!/usr/bin/env python3
"""M-8.3 — serve FastAPI demo (real 8B inference by default; optional mock via GROUNDED_DEMO_MOCK=1)."""

from __future__ import annotations

import click
import uvicorn


@click.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8080, show_default=True)
def main(host: str, port: int) -> None:
    uvicorn.run("grounded.demo.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
