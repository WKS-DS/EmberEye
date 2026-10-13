"""
run.py — Entry point.

Picks an adapter based on CLI args, connects it to the (currently minimal)
pipeline, and prints what it sees. As the pipeline grows, this file wires
in each new stage.

Usage:
    python -m embereye.run --adapter recorded --input data/recordings/demo_flight/
    python -m embereye.run --adapter sim --duration 120
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from embereye.adapters.base import DroneAdapter
from embereye.adapters.recorded import RecordedFootageAdapter
from embereye.adapters.sim import SimAdapter


def build_adapter(name: str, args: argparse.Namespace) -> DroneAdapter:
    """Instantiate the adapter the user asked for.

    Adding a new adapter later is a matter of adding one branch here plus
    the import at the top. Nothing else changes.
    """
    if name == "recorded":
        if not args.input:
            raise SystemExit("The 'recorded' adapter requires --input")
        return RecordedFootageAdapter(
            recording_dir=args.input,
            playback_fps=args.fps,
            loop=args.loop,
        )
    if name == "sim":
        return SimAdapter(duration_s=args.duration)
    # Future:
    #   if name == "rpi_custom": return RpiCustomAdapter(...)
    #   if name == "dji":        return DjiAdapter(...)
    raise SystemExit(f"Unknown adapter: {name}")


async def main_async(args: argparse.Namespace) -> None:
    """Consume frames from the adapter. This is where the pipeline will plug in."""
    adapter = build_adapter(args.adapter, args)
    print(f"Starting EmberEye with adapter: {adapter.name}")

    async with adapter:
        n = 0
        async for frame in adapter.stream_frames():
            n += 1
            if n % 30 == 0:                # a status line every ~1s of footage
                pos = frame.telemetry.position
                pos_str = f"{pos.lat:.5f}, {pos.lon:.5f}" if pos else "no GPS"
                print(
                    f"  frame {n:>6}  |  {frame.sensor.value:<7}  |  "
                    f"{frame.image.shape}  |  ts={frame.ts.isoformat()}  |  {pos_str}"
                )
    print(f"Done. {n} frames processed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="EmberEye runner")
    parser.add_argument("--adapter", required=True, choices=["recorded", "sim"], help="Which drone adapter to use")
    parser.add_argument("--input",   type=Path, help="Path to recording directory (recorded adapter)")
    parser.add_argument("--fps",     type=float, default=None, help="Playback fps (recorded adapter); default = as fast as possible")
    parser.add_argument("--loop",    action="store_true", help="Loop the recording forever")
    parser.add_argument("--duration", type=float, default=600.0, help="Length of the simulated flight in seconds (sim adapter)")
    args = parser.parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Interrupted.")


if __name__ == "__main__":
    main()
