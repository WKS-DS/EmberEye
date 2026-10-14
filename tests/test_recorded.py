"""Tests for the recorded-footage adapter (embereye/adapters/recorded.py).

Each test builds a tiny fake recording — a short video plus telemetry —
in a temporary folder. `tmp_path` is a pytest "fixture": naming it as a
test argument makes pytest hand the test a fresh, empty folder that is
cleaned up afterwards.
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from embereye.adapters.recorded import RecordedFootageAdapter

FPS = 30


def make_fake_recording(out_dir: Path, n_frames: int = 60, size=(320, 240)):
    """Write rgb.mp4, telemetry.json and metadata.json for a short fake flight."""
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = size

    # RGB video: gradient with a moving red dot (stands in for "hotspot")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    rgb = cv2.VideoWriter(str(out_dir / "rgb.mp4"), fourcc, FPS, (w, h))
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


def remove_telemetry_timestamps(recording_dir: Path) -> None:
    """Delete the "ts" key from every telemetry row."""
    path = recording_dir / "telemetry.json"
    rows = json.loads(path.read_text())
    for row in rows:
        del row["ts"]
    path.write_text(json.dumps(rows))


def collect_frames(recording_dir: Path, max_frames=None, **adapter_kwargs):
    """Stream a recording and return its Frames as a list."""
    async def run():
        frames = []
        async with RecordedFootageAdapter(recording_dir=recording_dir, **adapter_kwargs) as adapter:
            async for frame in adapter.stream_frames():
                frames.append(frame)
                if max_frames is not None and len(frames) >= max_frames:
                    break
        return frames

    return asyncio.run(run())


def test_streams_every_frame_with_gps(tmp_path):
    make_fake_recording(tmp_path, n_frames=60)
    frames = collect_frames(tmp_path)

    assert len(frames) == 60
    assert all(frame.telemetry.position is not None for frame in frames)
    assert frames[0].telemetry.position.lat == pytest.approx(47.6588)
    assert frames[0].fov_deg == 84.0


def test_timestamps_come_from_telemetry_when_present(tmp_path):
    make_fake_recording(tmp_path, n_frames=10)
    frames = collect_frames(tmp_path)

    telemetry = json.loads((tmp_path / "telemetry.json").read_text())
    for frame, row in zip(frames, telemetry):
        assert frame.ts == datetime.fromisoformat(row["ts"])


def test_timestamps_are_evenly_spaced_without_telemetry_ts(tmp_path):
    make_fake_recording(tmp_path, n_frames=10)
    remove_telemetry_timestamps(tmp_path)
    frames = collect_frames(tmp_path)

    # datetime only stores whole microseconds, so allow a few microseconds of rounding.
    gaps = [(b.ts - a.ts).total_seconds() for a, b in zip(frames, frames[1:])]
    assert gaps == pytest.approx([1 / FPS] * 9, abs=1e-5)
    assert (frames[-1].ts - frames[0].ts).total_seconds() == pytest.approx(9 / FPS, abs=1e-5)


def test_start_time_can_come_from_metadata(tmp_path):
    make_fake_recording(tmp_path, n_frames=5)
    remove_telemetry_timestamps(tmp_path)
    (tmp_path / "metadata.json").write_text(json.dumps({"start_ts": "2026-09-01T12:00:00+00:00"}))
    frames = collect_frames(tmp_path)

    assert frames[0].ts == datetime.fromisoformat("2026-09-01T12:00:00+00:00")


@pytest.mark.parametrize("with_telemetry_ts", [True, False])
def test_loop_mode_keeps_ids_and_timestamps_increasing(tmp_path, with_telemetry_ts):
    make_fake_recording(tmp_path, n_frames=10)
    if not with_telemetry_ts:
        remove_telemetry_timestamps(tmp_path)

    frames = collect_frames(tmp_path, max_frames=15, loop=True)     # about 1.5 loops

    ids = [int(frame.frame_id) for frame in frames]
    assert ids == list(range(15))
    assert all(b.ts > a.ts for a, b in zip(frames, frames[1:]))

    # Telemetry is still looked up by position in the recording, so the
    # second loop repeats the first loop's GPS track.
    assert frames[10].telemetry.position.lat == pytest.approx(frames[0].telemetry.position.lat)
