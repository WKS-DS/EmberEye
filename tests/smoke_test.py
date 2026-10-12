"""Smoke test — generate a fake recording, run adapter against it, count frames."""
import asyncio
import json
from pathlib import Path

import cv2
import numpy as np

from embereye.adapters.recorded import RecordedFootageAdapter


def make_fake_recording(out_dir: Path, n_frames: int = 60, size=(320, 240)):
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = size

    # RGB video: gradient with a moving red dot (stands in for "hotspot")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    rgb = cv2.VideoWriter(str(out_dir / "rgb.mp4"), fourcc, 30, (w, h))
    for i in range(n_frames):
        img = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cx = int((i / n_frames) * w)
        cv2.circle(img, (cx, h // 2), 10, (0, 0, 255), -1)
        rgb.write(img)
    rgb.release()

    # Telemetry: fake GPS drift
    telem = [
        {"ts": f"2026-08-15T14:00:{i//30:02d}.{(i%30)*33_000:06d}+00:00",
         "lat": 47.6588 + i * 1e-6,
         "lon": -117.4260 + i * 1e-6,
         "alt_m": 120.0,
         "heading_deg": 90.0}
        for i in range(n_frames)
    ]
    (out_dir / "telemetry.json").write_text(json.dumps(telem))
    (out_dir / "metadata.json").write_text(json.dumps({"rgb_fov_deg": 84.0}))


async def run():
    recording_dir = Path("/tmp/fake_flight")
    make_fake_recording(recording_dir, n_frames=60)

    adapter = RecordedFootageAdapter(recording_dir=recording_dir)
    n = 0
    positions_seen = 0
    async with adapter:
        async for frame in adapter.stream_frames():
            n += 1
            if frame.telemetry.position:
                positions_seen += 1

    assert n == 60, f"expected 60 frames, got {n}"
    assert positions_seen == 60, f"expected all frames to have GPS, got {positions_seen}"
    print(f"OK — {n} frames, {positions_seen} with GPS")


if __name__ == "__main__":
    asyncio.run(run())
