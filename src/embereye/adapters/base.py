"""
base.py — Drone adapter interface.

Every drone platform (DJI, custom PX4, simulated, recorded footage) is
represented by a class inheriting from DroneAdapter. The pipeline never
imports or knows about specific platforms; it only ever sees this
interface.

Adding support for a new drone means writing one class here. Nothing
downstream changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from embereye.models.types import Frame


class DroneAdapter(ABC):
    """
    Abstract base for anything that can produce Frames.

    The interface is deliberately tiny — just "give me frames as they arrive."
    Adapters handle everything platform-specific internally:
    - Connecting to the drone / SDK / file
    - Decoding video
    - Reading telemetry
    - Synchronizing RGB and thermal streams
    - Rate-limiting to the pipeline's target fps

    Adapters are async iterators. Consuming them looks like:

        async for frame in adapter:
            pipeline.process(frame)
    """

    #: Human-readable identifier, e.g. "recorded", "dji_m3t", "rpi_custom".
    name: str = "abstract"

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the drone / file / simulator.
        Called once before frames are requested. Should raise if the
        source is unreachable so we fail fast rather than mid-flight.
        """
        raise NotImplementedError

    @abstractmethod
    async def stream_frames(self) -> AsyncIterator[Frame]:
        """
        Yield Frames as they arrive from the platform.
        Should run until disconnected, the file ends, or the caller cancels.

        Implementations must:
        - Populate Frame.telemetry as fully as the platform allows
        - Use UTC timestamps
        - Not block the event loop with sync I/O
        """
        raise NotImplementedError
        yield  # pragma: no cover — makes this a valid generator

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean up resources. Called on shutdown."""
        raise NotImplementedError

    # -- default async context manager plumbing so consumers can use `async with`
    async def __aenter__(self) -> "DroneAdapter":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.disconnect()
