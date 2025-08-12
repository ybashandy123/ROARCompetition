# roar_gym_env.py
import math
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import platform
import asyncio

# Windows event loop policy fix (prevents deadlocks with CARLA/asyncio)
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import carla
import roar_py_carla
import roar_py_interface

from competition_code.competition_runner_plot_3d import RoarCompetitionRule  # reuse your progress/lap logic
# closest-waypoint helpers from your project
from submission import dist_to_waypoint, filter_waypoints


def _ensure_event_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


class RoarCarlaGymEnv(gym.Env):
    """
    Single-agent Gymnasium env for ROAR-Py/CARLA that exposes:
      Observation (vector):
        [ ray_0..ray_{N-1},  speed_kmh, sin(yaw), cos(yaw),
          pos_x_scaled, pos_y_scaled, rel_next_wp_dx, rel_next_wp_dy ]

      Action (continuous Box):
        [ steer(-1..1), throttle(0..1), brake(0..1) ]

    Reward:
      progress_delta * progress_scale
      + 0.003 * speed_kmh
      - collision_penalty (on major collision)
      - 0.001 * offroad_pixels (from occupancy map near ego)
    """
    metadata = {"render_modes": ["rgb_array", "none"]}
    def _ensure_world_once(self):
        if self._carla_client is None:
            self._carla_client = carla.Client(self.host, self.port)
            self._carla_client.set_timeout(10.0)
            self._roar_instance = roar_py_carla.RoarPyCarlaInstance(self._carla_client)
            self.world = self._roar_instance.world
            self.world.set_control_steps(0.05, 0.005)
            self.world.set_asynchronous(True)  # training-friendly

    def _spawn_vehicle_once(self):
        if self.vehicle is not None:
            return  # already have one

        self.waypoints = self.world.maneuverable_waypoints
        spawn_wp = self.waypoints[0]

        self.vehicle = self.world.spawn_vehicle(
            "vehicle.tesla.model3",
            spawn_wp.location + np.array([0, 0, 1.0]),
            spawn_wp.roll_pitch_yaw,
            True,
        )

        # attach sensors exactly once, bound to this vehicle
        self.camera = self.vehicle.attach_camera_sensor(
            roar_py_interface.RoarPyCameraSensorDataRGB,
            np.array([-2.0 * self.vehicle.bounding_box.extent[0], 0.0, 3.0 * self.vehicle.bounding_box.extent[2]]),
            np.array([0, 10 / 180.0 * np.pi, 0]),
            image_width=1024,
            image_height=768,
        )
        self.location_sensor = self.vehicle.attach_location_in_world_sensor()
        self.velocity_sensor = self.vehicle.attach_velocimeter_sensor()
        self.rpy_sensor = self.vehicle.attach_roll_pitch_yaw_sensor()
        self.occupancy_map_sensor = self.vehicle.attach_occupancy_map_sensor(
            50, 50, self._meters_per_cell_x, self._meters_per_cell_y
        )
    
        self.collision_sensor = self.vehicle.attach_collision_sensor(np.zeros(3), np.zeros(3))

    def _reset_vehicle_pose(self):
        # Move the same vehicle back to the spawn waypoint & zero dynamics
        spawn_wp = self.waypoints[0]
        # Set pose
        self._run(self.vehicle.set_3d_location(spawn_wp.location + np.array([0, 0, 1.0])), "set_location")
        self._run(self.vehicle.set_roll_pitch_yaw(spawn_wp.roll_pitch_yaw), "set_rpy")
        # Zero controls & wait a couple frames
        self._run(self.vehicle.apply_action({"steer":0.0,"throttle":0.0,"brake":1.0,"hand_brake":1,"reverse":0,"target_gear":1}), "apply_brake")
        for i in range(5):
            self._run(self.world.step(), f"world.step@reset_stabilize[{i}]")
        # Release handbrake for the episode
        self._run(self.vehicle.apply_action({"steer":0.0,"throttle":0.0,"brake":0.0,"hand_brake":0,"reverse":0,"target_gear":1}), "release_brake")

    def __init__(
        self,
        client_host: str = "127.0.0.1",
        client_port: int = 2000,
        num_rays: int = 32,
        fov_deg: float = 180.0,
        max_ray_m: float = 80.0,
        occ_thresh: int = 128,
        ray_sample_px: int = 120,                # how many pixels we march outward
        max_steps: int = 5000,
        progress_scale: float = 0.5,
        collision_impulse_major: float = 100.0,  # major collision threshold
        render_mode: str = "none",
    ):
        super().__init__()
        self.host = client_host
        self.port = client_port
        self.num_rays = int(num_rays)
        self.fov_deg = float(fov_deg)
        self.max_ray_m = float(max_ray_m)
        self.occ_thresh = int(occ_thresh)
        self.ray_sample_px = int(ray_sample_px)
        self.max_steps = int(max_steps)
        self.progress_scale = float(progress_scale)
        self.collision_impulse_major = float(collision_impulse_major)
        self.render_mode = render_mode

        # Observation space (vector)
        # rays normalized to [0,1] by max range, other scalings are gentle
        ray_dim = self.num_rays
        obs_dim = ray_dim + 1 + 2 + 2 + 2  # rays + speed + (sin,cos yaw) + (x,y) + rel dx,dy
        high = np.ones((obs_dim,), dtype=np.float32) * np.inf
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Action space: steer, throttle, brake
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([+1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Internals
        self._loop = _ensure_event_loop()
        self._await_timeout = 8.0  # seconds (bump if first frame is slow)
        self._carla_client: Optional[carla.Client] = None
        self._roar_instance: Optional[roar_py_carla.RoarPyCarlaInstance] = None
        self.world: Optional[roar_py_carla.RoarPyCarlaWorld] = None

        # Actors/sensors
        self.vehicle: Optional[roar_py_carla.RoarPyCarlaActor] = None
        self.camera = None
        self.location_sensor = None
        self.velocity_sensor = None
        self.rpy_sensor = None
        self.occupancy_map_sensor = None
        self.collision_sensor = None

        # Waypoints + progress tracking (reuse your class)
        self.rule: Optional[RoarCompetitionRule] = None
        self.waypoints = None
        self.current_waypoint_idx = 0

        # Rolling bookkeeping
        self._step_idx = 0
        self._prev_furthest_idx = 0

        # Occupancy grid scale: you attach sensor with (50,50, 2.0,2.0) → ~2 m/cell
        self._meters_per_cell_x = 2.0
        self._meters_per_cell_y = 2.0

    # --------------- Gym API ---------------
    # --- replace your reset() with this version ---

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._step_idx = 0

        # Create client/world once
        self._ensure_world_once()

        # Spawn vehicle/sensors once
        self._spawn_vehicle_once()

        # Progress rule can be rebuilt each episode (it references current vehicle & world)
        self.rule = RoarCompetitionRule(self.waypoints * 3, self.vehicle, self.world)

        # Reposition and stabilize the SAME vehicle (no new spawn)
        self._reset_vehicle_pose()

        # Warmup ticks then receive first frame (tick-before-receive)
        for i in range(20):
            self._run(self.world.step(), f"world.step@warmup[{i}]")
        self._run(self.vehicle.receive_observation(), "vehicle.receive_observation@warmup")

        self.rule.initialize_race()
        self._prev_furthest_idx = self.rule.furthest_waypoints_index

        # Initialize waypoint index near ego
        vehicle_location = self.location_sensor.get_last_gym_observation()
        self.current_waypoint_idx = filter_waypoints(vehicle_location, 0, self.waypoints)

        obs = self._get_obs()
        info = {"furthest_idx": self.rule.furthest_waypoints_index}
        return obs, info


    def step(self, action: np.ndarray):
        self._step_idx += 1
        action = np.asarray(action, dtype=np.float32)
        # Map to ROAR-Py car control dict
        control = {
            "steer": float(np.clip(action[0], -1, 1)),
            "throttle": float(np.clip(action[1], 0, 1)),
            "brake": float(np.clip(action[2], 0, 1)),
            "hand_brake": 0,
            "reverse": 0,
            "target_gear": 1,
        }

        # --- Apply → tick → receive (this order avoids stalls) ---
        self._run(self.vehicle.apply_action(control), "apply_action")
        self._run(self.world.step(), "world.step@step")
        self._run(self.vehicle.receive_observation(), "vehicle.receive_observation@step")

        # Update progression
        self._run(self.rule.tick(), "rule.tick")
        furthest = self.rule.furthest_waypoints_index
        progress_delta = max(0, furthest - self._prev_furthest_idx)
        self._prev_furthest_idx = furthest

        # Collision check
        collision = self.collision_sensor.get_last_observation()
        collision_pen = 0.0
        terminated_on_collision = False
        if collision is not None:
            impulse = float(np.linalg.norm(collision.impulse_normal))
            if impulse > self.collision_impulse_major:
                collision_pen = 5.0  # large negative on crash
                terminated_on_collision = True

        # Observation
        obs = self._get_obs()

        # Reward shaping
        speed_kmh = obs[self.num_rays]  # by construction below
        offroad_pen = self._estimate_offroad_penalty()
        reward = (
            self.progress_scale * float(progress_delta)
            + 0.003 * float(speed_kmh)
            - float(collision_pen)
            - 0.001 * float(offroad_pen)
        )

        # Episode end?
        terminated = bool(self.rule.lap_finished()) or terminated_on_collision
        truncated = self._step_idx >= self.max_steps

        info = {
            "progress_delta": int(progress_delta),
            "furthest_idx": int(furthest),
            "collision_impulse": float(0.0 if collision is None else np.linalg.norm(collision.impulse_normal)),
            "offroad_pen": float(offroad_pen),
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        # Best-effort cleanup
        try:
            if self.vehicle is not None:
                self.vehicle.close()
        except Exception:
            pass
        try:
            if self._roar_instance is not None:
                self._roar_instance.close()
        except Exception:
            pass

    # --------------- Helpers ---------------
    def _run(self, coro, label: str = ""):
        async def _with_to():
            return await asyncio.wait_for(coro, timeout=self._await_timeout)
        try:
            return self._loop.run_until_complete(_with_to())
        except asyncio.TimeoutError:
            raise RuntimeError(f"Async call timed out while waiting: {label or coro}")

    def _get_obs(self) -> np.ndarray:
        # Pull raw readings
        loc = self.location_sensor.get_last_gym_observation()      # [x,y,z]
        vel = self.velocity_sensor.get_last_gym_observation()      # [vx,vy,vz]
        rpy = self.rpy_sensor.get_last_gym_observation()           # [roll,pitch,yaw]
        yaw = float(rpy[2])
        sin_yaw, cos_yaw = math.sin(yaw), math.cos(yaw)
        speed_kmh = float(np.linalg.norm(vel) * 3.6)

        # Next waypoint index by your logic
        self.current_waypoint_idx = filter_waypoints(loc, self.current_waypoint_idx, self.waypoints)
        next_wp = self.waypoints[(self.current_waypoint_idx + 10) % len(self.waypoints)]
        rel_dxdy = (next_wp.location[:2] - np.asarray(loc[:2])).astype(np.float32)
        rel_dxdy /= (np.linalg.norm(rel_dxdy) + 1e-6)

        # Multi-raycast from occupancy map
        rays = self._raycast_distances()

        # Position scaling to keep magnitudes reasonable (~1e3 meters typical)
        pos_scale = 1000.0
        pos_xy = np.asarray(loc[:2], dtype=np.float32) / pos_scale

        obs = np.concatenate([
            rays.astype(np.float32),                     # N
            np.array([speed_kmh], dtype=np.float32),    # 1
            np.array([sin_yaw, cos_yaw], np.float32),   # 2
            pos_xy.astype(np.float32),                  # 2
            rel_dxdy.astype(np.float32),                # 2
        ], axis=0)
        return obs

    def _raycast_distances(self) -> np.ndarray:
        """
        Reads the egocentric occupancy map (PIL/Image) and shoots N rays
        from image center across +/-FOV. Returns normalized distances [0,1]
        where 1 == max_ray_m. If map missing, returns ones.
        """
        occ = self.occupancy_map_sensor.get_last_observation()
        if occ is None:
            return np.ones((self.num_rays,), dtype=np.float32)

        try:
            img = occ if hasattr(occ, "size") else occ.get_image()  # tolerate either direct Image or wrapped
            w, h = img.size
            cx, cy = w // 2, h // 2
            grid = np.array(img.convert("L"))  # grayscale
        except Exception:
            return np.ones((self.num_rays,), dtype=np.float32)

        # angle sweep centered on vehicle forward (assume up = -y in image → forward is upward).
        # If your frame is rotated differently, adjust 'base_angle'
        base_angle = -np.pi / 2  # up
        fov = np.deg2rad(self.fov_deg)
        start = base_angle - fov / 2
        end = base_angle + fov / 2

        # Convert meters to pixels; each cell is ~2m per your attach args
        meters_per_px = (self._meters_per_cell_x + self._meters_per_cell_y) * 0.5
        max_px = int(self.max_ray_m / meters_per_px)
        max_px = min(max_px, self.ray_sample_px)

        rays = np.empty((self.num_rays,), dtype=np.float32)
        for i, a in enumerate(np.linspace(start, end, self.num_rays)):
            dx = math.cos(a)
            dy = math.sin(a)
            hit = max_px
            for s in range(1, max_px + 1):
                x = int(round(cx + dx * s))
                y = int(round(cy + dy * s))
                if x < 0 or x >= w or y < 0 or y >= h:
                    hit = s
                    break
                if grid[y, x] >= self.occ_thresh:
                    hit = s
                    break
            dist_m = hit * meters_per_px
            rays[i] = np.clip(dist_m / self.max_ray_m, 0.0, 1.0)
        return rays

    def _estimate_offroad_penalty(self) -> float:
        """Small penalty proportional to occupied pixels within a close ring around center."""
        occ = self.occupancy_map_sensor.get_last_observation()
        if occ is None:
            return 0.0
        try:
            img = occ if hasattr(occ, "size") else occ.get_image()
            w, h = img.size
            cx, cy = w // 2, h // 2
            grid = np.array(img.convert("L"))
        except Exception:
            return 0.0

        radius_px = 6
        y0, y1 = max(0, cy - radius_px), min(h, cy + radius_px + 1)
        x0, x1 = max(0, cx - radius_px), min(w, cx + radius_px + 1)
        patch = grid[y0:y1, x0:x1]
        return float((patch >= self.occ_thresh).sum())
