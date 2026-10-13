"""
world.py — The MuJoCo scene: a ground made of coloured tiles and one camera.

MuJoCo is a physics simulator, but nothing here moves under physics. We
use it for two things it does well and exactly:

1. Rendering what a camera at a known position would see (the RGB frame).
2. "Segmentation" rendering: instead of colours, every pixel reports WHICH
   object it is looking at. That gives a perfect answer to "which ground
   cell does this pixel show?" with no machine learning involved.

The ground is one box ("tile") per grid cell, so recolouring a tile is how
a cell is shown as unburned, burning, or burned out.

World axes match geo.py: x = east, y = north, z = up, in metres.
"""

from __future__ import annotations

import mujoco
import numpy as np

from embereye.geo import GroundGrid
from embereye.prediction.terrain import Terrain
from embereye.sim.camera import CameraModel, look_rotation

CAMERA_NAME = "drone_cam"

OUTSKIRTS_RGB = (0.33, 0.38, 0.24)   # ground beyond the edge of the grid
TILE_DEPTH_M  = 1.0                  # how far each tile extends below the lowest ground


def build_world_xml(grid: GroundGrid, terrain: Terrain, camera: CameraModel) -> str:
    """Write the scene description (MuJoCo's XML format) as a string."""
    floor_up = float(terrain.elevation.min())
    half = grid.cell_size_m / 2
    east, north = grid.cell_centers()

    # One box per cell. A box is described by its centre and its HALF-sizes.
    # Each box runs from below the lowest ground up to that cell's
    # elevation, so neighbouring tiles of different heights leave no gaps.
    bottom = floor_up - TILE_DEPTH_M
    tiles = []
    for row in range(grid.rows):
        for col in range(grid.cols):
            top = float(terrain.elevation[row, col])
            center_up   = (top + bottom) / 2
            half_height = (top - bottom) / 2
            tiles.append(
                f'    <geom name="tile_{row}_{col}" type="box" '
                f'pos="{east[row, col]:.3f} {north[row, col]:.3f} {center_up:.3f}" '
                f'size="{half:.3f} {half:.3f} {half_height:.3f}"/>'
            )
    tiles_xml = "\n".join(tiles)

    extent = max(grid.width_m, grid.height_m)
    outskirts = " ".join(str(v) for v in OUTSKIRTS_RGB)

    return f"""
<mujoco model="embereye_fire_world">
  <!-- Nothing collides or moves; this scene is only ever looked at. -->
  <option><flag contact="disable"/></option>
  <default><geom contype="0" conaffinity="0"/></default>

  <statistic center="0 0 0" extent="{extent:.1f}"/>
  <visual>
    <global offwidth="{camera.width}" offheight="{camera.height}"/>
    <!-- Camera sees from about 1 m to far beyond the grid. -->
    <map znear="0.002" zfar="100"/>
    <!-- Even daylight: soft fill light everywhere plus a "sun" below. -->
    <headlight ambient="0.45 0.45 0.45" diffuse="0 0 0" specular="0 0 0"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.70 0.90" rgb2="0.85 0.90 0.95" width="64" height="384"/>
  </asset>

  <worldbody>
    <light name="sun" directional="true" dir="0.4 0.3 -1" diffuse="0.6 0.6 0.6" specular="0 0 0" castshadow="false"/>

    <!-- Plain ground beyond the grid, a hair below it so the two never overlap. -->
    <geom name="outskirts" type="plane" pos="0 0 {floor_up - 0.05:.3f}" size="20000 20000 1" rgba="{outskirts} 1"/>

{tiles_xml}

    <!-- A "mocap" body is one we place by hand each frame. The camera rides on it. -->
    <body name="drone" mocap="true" pos="0 0 100">
      <camera name="{CAMERA_NAME}" fovy="{camera.fov_v_deg}"/>
    </body>
  </worldbody>
</mujoco>
"""


class FireWorld:
    """The rendered scene. Recolour tiles, place the camera, take a picture."""

    def __init__(self, grid: GroundGrid, terrain: Terrain, camera: CameraModel) -> None:
        if terrain.shape != grid.shape:
            raise ValueError(f"terrain shape {terrain.shape} does not match grid shape {grid.shape}")

        self.grid    = grid
        self.terrain = terrain
        self.camera  = camera

        self.model = mujoco.MjModel.from_xml_string(build_world_xml(grid, terrain, camera))
        self.data  = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model,
            height=camera.height,
            width=camera.width,
            max_geom=self.model.ngeom + 100,
        )
        self._camera_id = self.model.camera(CAMERA_NAME).id

        # MuJoCo numbers objects ("geoms") in the order they appear in the
        # XML. Tiles were written row by row, so their ids are consecutive
        # and tile number k is grid cell (k // cols, k % cols).
        n_cells = grid.rows * grid.cols
        first = self.model.geom("tile_0_0").id
        last  = self.model.geom(f"tile_{grid.rows - 1}_{grid.cols - 1}").id
        if last - first != n_cells - 1:
            raise RuntimeError("Tile ids are not consecutive; cannot map pixels to grid cells.")
        self._tile_geom_ids = np.arange(first, first + n_cells)

        # Lookup table: MuJoCo geom id -> flat cell index (-1 for non-tiles).
        self._geom_to_cell = np.full(self.model.ngeom, -1, dtype=np.int32)
        self._geom_to_cell[self._tile_geom_ids] = np.arange(n_cells, dtype=np.int32)

    # ------------------------------------------------------------------ tiles
    def set_tile_colors(self, colors: np.ndarray) -> None:
        """Recolour every tile. `colors` is (rows, cols, 3) with values 0..1."""
        self.model.geom_rgba[self._tile_geom_ids, :3] = colors.reshape(-1, 3)

    # ----------------------------------------------------------------- camera
    def move_camera(
        self,
        east_m: float,
        north_m: float,
        up_m: float,
        heading_deg: float,
        pitch_deg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Place the camera, then read back the pose MuJoCo will render from.

        Returns (position, rotation). Reading it back, rather than assuming
        it, guarantees the reported pose is the one the picture was taken from.
        """
        rotation = look_rotation(heading_deg, pitch_deg)

        # MuJoCo stores orientations as quaternions (4 numbers).
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.flatten())

        self.data.mocap_pos[0]  = (east_m, north_m, up_m)
        self.data.mocap_quat[0] = quat
        mujoco.mj_forward(self.model, self.data)    # update everything that depends on the new pose

        position = self.data.cam_xpos[self._camera_id].copy()
        rotation = self.data.cam_xmat[self._camera_id].reshape(3, 3).copy()
        return position, rotation

    # ----------------------------------------------------------------- render
    def render(self) -> tuple[np.ndarray, np.ndarray]:
        """Take the picture from the current camera pose.

        Returns:
            rgb:      (height, width, 3) uint8 colour image.
            cell_ids: (height, width) int32. For each pixel, the flat index
                      of the ground cell it shows (row * cols + col), or -1
                      if the pixel shows sky or ground beyond the grid.
        """
        self.renderer.update_scene(self.data, camera=CAMERA_NAME)
        rgb = self.renderer.render().copy()

        # Same view again, but each pixel reports (object id, object type).
        self.renderer.enable_segmentation_rendering()
        segmentation = self.renderer.render()
        self.renderer.disable_segmentation_rendering()

        object_ids   = segmentation[..., 0]
        object_types = segmentation[..., 1]
        is_geom = object_types == int(mujoco.mjtObj.mjOBJ_GEOM)

        cell_ids = np.full(object_ids.shape, -1, dtype=np.int32)
        cell_ids[is_geom] = self._geom_to_cell[object_ids[is_geom]]
        return rgb, cell_ids

    def close(self) -> None:
        """Release the renderer's graphics resources."""
        self.renderer.close()
