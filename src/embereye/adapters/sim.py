"""
sim.py — Adapter for the fire simulator.

Presents the simulated drone (see embereye/sim/) through the same
DroneAdapter interface as a real one, so the pipeline cannot tell the
difference: it just receives Frames with an image and telemetry.

A real drone does not know where the fire is, and neither does a Frame.
The simulator does, though, and tests need that answer key. So alongside
each Frame, the adapter keeps the full SimObservation (fire mask, exact
camera pose, true ground state) in `adapter.latest`:

    async with SimAdapter(duration_s=600) as adapter:
        async for frame in adapter.stream_frames():
            truth = adapter.latest      # the observation this frame came from
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

from embereye.adapters.base import DroneAdapter
from embereye.models.types import Frame, GeoPoint, SensorType, Telemetry
from embereye.sim.camera import CameraModel, CameraPath, OrbitPath
from embereye.sim.fire_sim import FireSim, Scenario, SimObservation, demo_scenario


class SimAdapter(DroneAdapter):
    """Fly a scripted camera over a simulated fire and emit Frames."""

    name = "sim"

    def __init__(
        self,
        scenario: Optional[Scenario] = None,        # None = the built-in demo fire
        path: Optional[CameraPath] = None,          # None = the default orbit
        camera: Optional[CameraModel] = None,
        duration_s: float = 600.0,                  # length of the simulated flight
        frame_interval_s: float = 1.0,              # simulated seconds between frames
        realtime: bool = False,                     # True = wait between frames like a live feed
    ) -> None:
        self.scenario         = scenario if scenario is not None else demo_scenario()
        self.path             = path if path is not None else OrbitPath()
        self.camera           = camera if camera is not None else CameraModel()
        self.duration_s       = duration_s
        self.frame_interval_s = frame_interval_s
        self.realtime         = realtime

        self.sim:    Optional[FireSim]        = None
        self.latest: Optional[SimObservation] = None   # answer key for the most recent Frame

    # ---------------------------------------------------------------- setup
    async def connect(self) -> None:
        # Building the sim works out the whole fire and opens the renderer.
        self.sim = FireSim(self.scenario, self.path, self.camera)

    async def disconnect(self) -> None:
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    # ---------------------------------------------------------------- stream
    async def stream_frames(self) -> AsyncIterator[Frame]:
        """Yield one RGB Frame per simulated time step."""
        if self.sim is None:
            raise RuntimeError("Adapter not connected. Call connect() or use `async with`.")

        for obs in self.sim.run(self.duration_s, self.frame_interval_s):
            self.latest = obs
            yield self._build_frame(obs)

            # Rendering is quick but not instant, so hand control back to
            # the event loop between frames. In realtime mode, also wait
            # out the gap a live camera would have.
            await asyncio.sleep(self.frame_interval_s if self.realtime else 0)

    # ---------------------------------------------------------------- helpers
    def _build_frame(self, obs: SimObservation) -> Frame:
        """Package an observation the way a real drone would report it."""
        pose = obs.pose
        telemetry = Telemetry(
            position=GeoPoint(lat=pose.lat, lon=pose.lon, alt_m=pose.alt_msl_m),
            heading_deg=pose.heading_deg,
            gimbal_pitch=pose.pitch_deg,
            gimbal_yaw=0.0,             # the camera always faces the way the drone is heading
        )
        return Frame(
            ts=obs.ts,
            sensor=SensorType.RGB,
            image=obs.rgb,
            telemetry=telemetry,
            fov_deg=self.camera.fov_h_deg,
            frame_id=f"{obs.index:08d}",
        )
