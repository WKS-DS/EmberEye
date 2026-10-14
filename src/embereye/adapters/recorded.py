"""
recorded.py — Adapter for pre-recorded flight footage.

This is the most important adapter for development. It reads a video file
(and optionally a matching thermal video + telemetry JSON) and emits Frames
just like a real drone would, at the recorded frame rate or a rate we
specify.

Why this exists:
- Develop the entire pipeline with no drone, no field trip, no permits.
- Reproduce demos exactly — same recording in, same output out.
- Debug the pipeline on known-hard cases (edge conditions from real flights)
  without waiting for the weather to cooperate.

Companion file layout for a "flight":

    recordings/
      demo_flight_2026_08_15/
        rgb.mp4                 # visible-light video
        thermal.mp4  (optional) # thermal video, radiometric preferred
        telemetry.json          # per-frame GPS, altitude, gimbal, etc.
        metadata.json           # fov_deg, camera params, sensor types, start_ts
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

import cv2
import numpy as np

from embereye.adapters.base import DroneAdapter
from embereye.models.types import Frame, GeoPoint, SensorType, Telemetry


class RecordedFootageAdapter(DroneAdapter):
    """Read a recorded flight from disk and emit Frames."""

    name = "recorded"

    def __init__(
        self,
        recording_dir: str | Path,
        playback_fps: Optional[float] = None,   # None = as fast as pipeline can consume
        loop: bool = False,
    ) -> None:
        self.recording_dir  = Path(recording_dir)
        self.playback_fps   = playback_fps
        self.loop           = loop

        self._rgb_cap:      Optional[cv2.VideoCapture] = None
        self._thermal_cap:  Optional[cv2.VideoCapture] = None
        self._telemetry:    list[dict]                 = []
        self._metadata:     dict                       = {}
        self._start_ts:     Optional[datetime]         = None

    # ---------------------------------------------------------------- setup
    async def connect(self) -> None:
        if not self.recording_dir.exists():
            raise FileNotFoundError(f"Recording directory not found: {self.recording_dir}")

        # Metadata is optional but recommended.
        meta_path = self.recording_dir / "metadata.json"
        if meta_path.exists():
            self._metadata = json.loads(meta_path.read_text())

        # When the recording started. Frames whose telemetry has no "ts"
        # are timestamped from this plus their position in the stream.
        # Chosen once here, so every frame shares the same starting point.
        if "start_ts" in self._metadata:
            self._start_ts = datetime.fromisoformat(self._metadata["start_ts"])
        else:
            self._start_ts = datetime.now(timezone.utc)

        # Telemetry — per-frame drone state, indexed by frame number.
        # Optional; if missing, telemetry fields on Frame will be None.
        telem_path = self.recording_dir / "telemetry.json"
        if telem_path.exists():
            self._telemetry = json.loads(telem_path.read_text())

        # RGB video is required.
        rgb_path = self.recording_dir / "rgb.mp4"
        if not rgb_path.exists():
            raise FileNotFoundError(f"Missing required rgb.mp4 in {self.recording_dir}")
        self._rgb_cap = cv2.VideoCapture(str(rgb_path))
        if not self._rgb_cap.isOpened():
            raise RuntimeError(f"Failed to open {rgb_path}")

        # Thermal video is optional (useful for RGB-only recordings from
        # consumer drones during early development).
        thermal_path = self.recording_dir / "thermal.mp4"
        if thermal_path.exists():
            self._thermal_cap = cv2.VideoCapture(str(thermal_path))

    async def disconnect(self) -> None:
        if self._rgb_cap is not None:
            self._rgb_cap.release()
        if self._thermal_cap is not None:
            self._thermal_cap.release()

    # ---------------------------------------------------------------- stream
    async def stream_frames(self) -> AsyncIterator[Frame]:
        """Yield Frames for as long as there's footage to yield."""
        if self._rgb_cap is None:
            raise RuntimeError("Adapter not connected. Call connect() or use `async with`.")

        # Native frame interval (seconds) — used when playback_fps is None
        # and we're pacing to real time; also useful for timestamping.
        source_fps = self._rgb_cap.get(cv2.CAP_PROP_FPS) or 30.0
        native_dt  = 1.0 / source_fps
        target_dt  = 1.0 / self.playback_fps if self.playback_fps else 0.0

        # Two counters, because looping makes them differ:
        # - frame_idx: position within the recording (0, 1, 2, ... restarting
        #   at 0 on every loop). Telemetry is looked up with this.
        # - emitted:   frames handed out since streaming began, never reset.
        #   Frame ids and computed timestamps use this, so they keep
        #   increasing across loops.
        frame_idx = 0
        emitted = 0
        loops_done = 0
        frames_per_loop = 0
        while True:
            # Note: read() is a blocking call, so it holds up the event loop
            # while it decodes. It could later move to asyncio.to_thread().
            ok, rgb = self._rgb_cap.read()
            if not ok:
                if self.loop and frame_idx > 0:      # frame_idx == 0 means an empty video: nothing to loop
                    self._rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    if self._thermal_cap is not None:
                        self._thermal_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frames_per_loop = frame_idx
                    loops_done += 1
                    frame_idx = 0
                    continue
                return

            # How far later than the first pass this loop is, in seconds.
            # Added to timestamps that come from telemetry, which repeat
            # every loop.
            loop_offset_s = loops_done * frames_per_loop * native_dt

            # OpenCV returns BGR — convert to RGB so downstream code doesn't
            # have to remember what colorspace this is.
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            yield self._build_frame(rgb, SensorType.RGB, frame_idx, emitted, native_dt, loop_offset_s)

            if self._thermal_cap is not None:
                ok_t, thermal = self._thermal_cap.read()
                if ok_t:
                    # Thermal videos from consumer drones are usually already
                    # false-color-mapped RGB. If radiometric, this needs to
                    # be decoded per platform — a job for the platform-specific
                    # adapter, not this one.
                    thermal_gray = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    yield self._build_frame(thermal_gray, SensorType.THERMAL, frame_idx, emitted, native_dt, loop_offset_s)

            frame_idx += 1
            emitted += 1
            if target_dt > 0:
                await asyncio.sleep(target_dt)

    # ---------------------------------------------------------------- helpers
    def _build_frame(
        self,
        image: np.ndarray,
        sensor: SensorType,
        frame_idx: int,
        emitted: int,
        native_dt: float,
        loop_offset_s: float,
    ) -> Frame:
        """Assemble a Frame with telemetry looked up by frame index.

        frame_idx:     position within the recording (selects the telemetry row)
        emitted:       frames emitted so far across all loops (id and timestamp)
        loop_offset_s: seconds to add to telemetry timestamps on repeat loops
        """
        telem_row = self._telemetry[frame_idx] if frame_idx < len(self._telemetry) else {}

        position = None
        if "lat" in telem_row and "lon" in telem_row:
            position = GeoPoint(
                lat=telem_row["lat"],
                lon=telem_row["lon"],
                alt_m=telem_row.get("alt_m"),
            )

        telemetry = Telemetry(
            position=position,
            heading_deg=telem_row.get("heading_deg"),
            gimbal_pitch=telem_row.get("gimbal_pitch"),
            gimbal_yaw=telem_row.get("gimbal_yaw"),
            battery_pct=telem_row.get("battery_pct"),
        )

        # Timestamp: if telemetry has one, use it (shifted along on repeat
        # loops). Otherwise rebuild it from the recording's start time and
        # how many frames have gone by at the video's native frame rate.
        if "ts" in telem_row:
            ts = datetime.fromisoformat(telem_row["ts"]) + timedelta(seconds=loop_offset_s)
        else:
            ts = self._start_ts + timedelta(seconds=emitted * native_dt)

        return Frame(
            ts=ts,
            sensor=sensor,
            image=image,
            telemetry=telemetry,
            fov_deg=self._metadata.get(f"{sensor.value}_fov_deg"),
            frame_id=f"{emitted:08d}",
        )
