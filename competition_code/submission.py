"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from collections import deque
from functools import reduce
import json
import os
from typing import List, Tuple, Dict, Optional
import math
import numpy as np
import roar_py_interface
from LateralController import LatController
from ThrottleController import ThrottleController
import atexit

import infrastructure_debug

# from scipy.interpolate import interp1d

useDebug = True
useDebugPrinting = False
debugData = {}


def dist_to_waypoint(location, waypoint: roar_py_interface.RoarPyWaypoint):
    return np.linalg.norm(location[:2] - waypoint.location[:2])


def filter_waypoints(
    location: np.ndarray,
    current_idx: int,
    waypoints: List[roar_py_interface.RoarPyWaypoint],
) -> int:
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(location, waypoints[i % len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx


def findClosestIndex(location, waypoints: List[roar_py_interface.RoarPyWaypoint]):
    lowestDist = 100
    closestInd = 0
    for i in range(0, len(waypoints)):
        dist = dist_to_waypoint(location, waypoints[i % len(waypoints)])
        if dist < lowestDist:
            lowestDist = dist
            closestInd = i
    return closestInd % len(waypoints)
 
@atexit.register
def saveDebugData():
    if useDebug:
        print("Saving debug data")
        jsonData = json.dumps(debugData, indent=4)
        with open(
            f"{os.path.dirname(__file__)}\\debugData\\debugData.json", "w+"
        ) as outfile:
            outfile.write(jsonData)
        print("Debug Data Saved")

data = np.load(f"{os.path.dirname(__file__)}\\waypoints\\centeredWaypoints.npz")
centerline_xy = roar_py_interface.RoarPyWaypoint.load_waypoint_list(data)

import numpy as np

CAR_LENGTH = 4.719 
CAR_WIDTH  = 2.09 
TRACK_HALF_WIDTH = 5.0 
TRACK_CLOSED = True 
EPS = 1e-9

def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > EPS else v

def _rot90_ccw_batch(T):  # T: (N,2) -> (-y, x) per row
    return np.stack([-T[:, 1], T[:, 0]], axis=1)

def _cross2(a, b):
    # 2D cross product scalar: ax*by - ay*bx
    return a[0]*b[1] - a[1]*b[0]

def _as_xy_points(centerline_like):
    """
    Accepts:
      - list/seq of RoarPyWaypoint (with .location vector-like)
      - Nx2 or Nx3 numpy array (we'll use first 2 cols)
      - list of [x,y,(z)] points
    Returns:
      Nx2 float array
    """
    # If it's already a NumPy array of shape (N, >=2)
    if isinstance(centerline_like, np.ndarray):
        if centerline_like.ndim != 2 or centerline_like.shape[1] < 2:
            raise TypeError("centerline array must be shape (N, >=2)")
        return centerline_like[:, :2].astype(float)

    # Otherwise treat as a sequence
    seq = list(centerline_like)
    if len(seq) == 0:
        raise ValueError("Empty centerline")

    first = seq[0]
    # RoarPyWaypoint path
    if hasattr(first, "location"):
        out = np.empty((len(seq), 2), dtype=float)
        for i, wp in enumerate(seq):
            loc = np.asarray(wp.location, dtype=float).ravel()
            if loc.size < 2:
                raise TypeError("RoarPyWaypoint.location must have at least 2 components (x,y)")
            out[i] = loc[:2]
        return out

    # Generic [x,y,(z)] points
    try:
        arr = np.asarray(seq, dtype=float)
    except Exception as e:
        raise TypeError("Unsupported centerline data type") from e

    if arr.ndim == 1 and arr.size % 2 == 0:
        arr = arr.reshape(-1, 2)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise TypeError("centerline must be convertible to shape (N, >=2)")
    return arr[:, :2]

def _as_xy(vec_like):
    """Coerce 2D from 2D/3D vector-like or object with .x/.y or .location/.position."""
    if hasattr(vec_like, "location"):
        v = np.asarray(vec_like.location, dtype=float).ravel()
        return v[:2]
    if hasattr(vec_like, "position"):
        v = np.asarray(vec_like.position, dtype=float).ravel()
        return v[:2]
    # objects with .x/.y
    if hasattr(vec_like, "x") and hasattr(vec_like, "y"):
        return np.array([float(vec_like.x), float(vec_like.y)], dtype=float)
    # numpy/list
    v = np.asarray(vec_like, dtype=float).ravel()
    if v.size < 2:
        raise TypeError("Vector must have at least 2 components (x,y)")
    return v[:2]

def _extract_yaw_rad(rotation_like):
    """
    Accept float (rad), object with .yaw (deg or rad), or a 3-vector [roll,pitch,yaw].
    Heuristic: if |yaw| > ~1.5π, assume degrees and convert.
    """
    # direct float/int
    if isinstance(rotation_like, (int, float, np.floating)):
        yaw = float(rotation_like)
    elif hasattr(rotation_like, "yaw"):
        yaw = float(rotation_like.yaw)
    else:
        # try indexable [roll, pitch, yaw]
        try:
            yaw = float(rotation_like[2])
        except Exception:
            # last resort: 0
            yaw = 0.0

    if abs(yaw) > np.pi * 1.5:
        yaw = np.deg2rad(yaw)
    return yaw

def build_track_edges(centerline_like):
    """
    Input can be ROAR Py waypoints (list) or an Nx2/Nx3 array.
    Returns left_xy, right_xy as (N,2) arrays.
    """
    C = _as_xy_points(centerline_like)              # (N,2)
    Np = len(C)
    # tangents via central differences (wrap if closed)
    tangents = np.zeros_like(C)
    if TRACK_CLOSED:
        C_prev = np.roll(C, 1, axis=0)
        C_next = np.roll(C, -1, axis=0)
    else:
        C_prev = np.vstack([C[0], C[:-1]])
        C_next = np.vstack([C[1:], C[-1]])
    T = C_next - C_prev
    # normalize row-wise
    norms = np.linalg.norm(T, axis=1, keepdims=True)
    norms = np.where(norms < EPS, 1.0, norms)
    Tn = T / norms

    normals_left = _rot90_ccw_batch(Tn)            # (-Ty, Tx)
    left_xy  = C + TRACK_HALF_WIDTH * normals_left
    right_xy = C - TRACK_HALF_WIDTH * normals_left
    return left_xy, right_xy

def raycast_polyline(origin, direction, polyline):
    """
    Ray: p(t) = origin + t * direction, t >= 0
    Segment: q + s * v, s in [0,1]
    Returns: (t_min, hit_point, seg_index) with np.inf if no hit
    """
    p = _as_xy(origin).astype(float)
    u = _normalize(_as_xy(direction).astype(float))
    P = _as_xy_points(polyline).astype(float)
    M = len(P)

    t_min = np.inf
    hit = None
    hit_idx = -1

    max_i = M if TRACK_CLOSED else M - 1
    for i in range(max_i):
        q  = P[i]
        q2 = P[(i + 1) % M] if TRACK_CLOSED else P[i + 1]
        v = q2 - q
        denom = _cross2(u, v)
        if abs(denom) < EPS:
            continue
        w = q - p
        t = _cross2(w, v) / denom
        s = _cross2(w, u) / denom
        if t >= 0.0 and 0.0 <= s <= 1.0:
            if t < t_min:
                t_min = t
                hit = p + t * u
                hit_idx = i
    return t_min, hit, hit_idx

def distance_to_wall_from_exterior(vehicle_location, vehicle_rotation, vehicle_velocity, centerline_like):
    """
    vehicle_location: 2D/3D vector-like or object with .x/.y or .location holding [x,y,(z)]
    vehicle_rotation: yaw in radians, OR object with .yaw (deg or rad), OR [roll,pitch,yaw]
    vehicle_velocity: 2D/3D vector-like (we use x,y). If near zero, falls back to heading.
    centerline_like: list of RoarPyWaypoint OR Nx2/Nx3 array of center waypoints

    Returns dict:
      distance_exterior (float),
      distance_center_to_hit (float),
      which_wall ('left'|'right'|None),
      hit_point (2,) or None,
      ray_dir (2,)
    """
    left_xy, right_xy = build_track_edges(centerline_like)

    pos_xy = _as_xy(vehicle_location)
    vel_xy = _as_xy(vehicle_velocity)
    yaw = _extract_yaw_rad(vehicle_rotation)

    # Ray direction: prefer velocity if nonzero, else heading
    u = _normalize(vel_xy) if np.linalg.norm(vel_xy) > EPS else np.array([np.cos(yaw), np.sin(yaw)], dtype=float)

    # Vehicle body axes
    f = np.array([np.cos(yaw), np.sin(yaw)], dtype=float)  # forward
    r = np.array([ f[1], -f[0] ], dtype=float)             # right

    # Car support distance along ray direction (center -> exterior skin)
    a = 0.5 * CAR_LENGTH
    b = 0.5 * CAR_WIDTH
    support = a * abs(np.dot(u, f)) + b * abs(np.dot(u, r))

    # Raycast to both walls
    tL, hitL, _ = raycast_polyline(pos_xy, u, left_xy)
    tR, hitR, _ = raycast_polyline(pos_xy, u, right_xy)

    t_hit = min(tL, tR)
    if np.isinf(t_hit):
        return {
            "distance_exterior": np.inf,
            "distance_center_to_hit": np.inf,
            "which_wall": None,
            "hit_point": None,
            "ray_dir": u,
        }

    which = 'left' if tL <= tR else 'right'
    hit_point = hitL if which == 'left' else hitR

    distance_center_to_hit = float(t_hit)
    distance_exterior = max(0.0, distance_center_to_hit - support)

    return {
        "distance_exterior": float(distance_exterior),
        "distance_center_to_hit": distance_center_to_hit,
        "which_wall": which,
        "hit_point": hit_point,
        "ray_dir": u,
    }

class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle: roar_py_interface.RoarPyActor,
        camera_sensor: roar_py_interface.RoarPyCameraSensor = None,
        location_sensor: roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor: roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor: roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor: roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor: roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
        self.lat_controller = LatController()
        self.throttle_controller = ThrottleController()
        self.section_indeces = []
        self.num_ticks = 0
        self.section_start_ticks = 0
        self.current_section = 0
        self.lapNum = 1

    async def initialize(self) -> None:
        # NOTE waypoints are changed through this line
        self.maneuverable_waypoints = (
            roar_py_interface.RoarPyWaypoint.load_waypoint_list(
                np.load(f"{os.path.dirname(__file__)}\\waypoints\\waypointsPrimary.npz")
            )[35:]
        )

        sectionLocations = [
            [-278, 372], # Section 0 start location
            [64, 890], # Section 1 start location
            [511, 1037], # Section 2 start location
            [762, 908], # Section 3 start location
            [198, 307], # Section 4 start location
            [-11, 60], # Section 5 start location
            [-85, -339], # Section 6 start location
            [-210, -1060], # Section 7 start location 
            [-318, -991], # Section 8 start location
            [-352, -119], # Section 9 start location
        ]
        for i in sectionLocations:
            self.section_indeces.append(
                findClosestIndex(i, self.maneuverable_waypoints)
            )

        print(f"True total length: {len(self.maneuverable_waypoints) * 3}")
        print(f"1 lap length: {len(self.maneuverable_waypoints)}")
        print(f"Section indexes: {self.section_indeces}")
        print("\nLap 1\n")

        # Receive location, rotation and velocity data
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 0
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )

    async def step(self) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        self.num_ticks += 1

        # Receive location, rotation and velocity data
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        current_speed_kmh = vehicle_velocity_norm * 3.6

        time_to_hit = 100
        distance_to_hit = {}
        if current_speed_kmh > 100:
            distance_to_hit = distance_to_wall_from_exterior(vehicle_location, vehicle_rotation, vehicle_velocity, centerline_xy)
        if (distance_to_hit.get("distance_exterior") or 100) < 30:
            time_to_hit = distance_to_hit["distance_exterior"] / (vehicle_velocity_norm or 0.001)
            print(f"Distance high enough, time to hit: {time_to_hit}")
        
        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )

        # compute and print section timing
        for i, section_ind in enumerate(self.section_indeces):
            if (
                abs(self.current_waypoint_idx - section_ind) <= 2
                and i != self.current_section
            ):
                print(f"Section {i}: {self.num_ticks - self.section_start_ticks} ticks")
                self.section_start_ticks = self.num_ticks
                self.current_section = i
                if self.current_section == 0 and self.lapNum != 3:
                    self.lapNum += 1
                    print(f"\nLap {self.lapNum}\n")

        nextWaypointIndex = self.get_lookahead_index(current_speed_kmh)
        waypoint_to_follow = self.next_waypoint_smooth(current_speed_kmh)

        # Pure pursuit controller to steer the vehicle
        steer_control = self.lat_controller.run(
            vehicle_location, vehicle_rotation, waypoint_to_follow
        )

        # Custom controller to control the vehicle's speed
        waypoints_for_throttle = (self.maneuverable_waypoints * 2)[
            nextWaypointIndex : nextWaypointIndex + 300
        ]
        throttle, brake, gear = self.throttle_controller.run(
            waypoints_for_throttle,
            vehicle_location,
            current_speed_kmh,
            self.current_section,
            time_to_hit
        )

        steerMultiplier = round((current_speed_kmh + 0.001) / 120, 3)
        
        if self.current_section == 2:
            steerMultiplier *= 1.2
        if self.current_section in [3]:
            # steerMultiplier *= 0.9
            steerMultiplier = np.clip(steerMultiplier * 1.75, 2.3, 3.5)
        if self.current_section == 4:
            steerMultiplier = min(1.45, steerMultiplier * 1.65)
        if self.current_section == 5:
            steerMultiplier *= 1.1
        if self.current_section in [6]:
            # steerMultiplier = min(steerMultiplier * 5, 5.35)
            steerMultiplier = np.clip(steerMultiplier * 5.5, 5.5, 7)
        if self.current_section == 7:
            steerMultiplier *= 2
        if self.current_section == 9:
            steerMultiplier = max(steerMultiplier, 1.6)

        control = {
            "throttle": np.clip(throttle, 0, 1),
            "steer": np.clip(steer_control * steerMultiplier, -1, 1),
            "brake": np.clip(brake, 0, 1),
            "hand_brake": 0,
            "reverse": 0,
            "target_gear": gear,  # Gears do not appear to have an impact on speed
        }

        infrastructure_debug.control_variable = control
        infrastructure_debug.current_section = self.current_section
        
        if useDebug:
            debugData[self.num_ticks] = {}
            debugData[self.num_ticks]["loc"] = [
                round(vehicle_location[0].item(), 3),
                round(vehicle_location[1].item(), 3),
            ]
            debugData[self.num_ticks]["throttle"] = round(float(control["throttle"]), 3)
            debugData[self.num_ticks]["brake"] = round(float(control["brake"]), 3)
            debugData[self.num_ticks]["steer"] = round(float(control["steer"]), 10)
            debugData[self.num_ticks]["speed"] = round(current_speed_kmh, 3)
            debugData[self.num_ticks]["lap"] = self.lapNum
                
        await self.vehicle.apply_action(control)
        return control

    def get_lookahead_value(self, speed):
        """
        Returns the number of waypoints to look ahead based on the speed the car is currently going
        """
        speed_to_lookahead_dict = {
            90: 9,
            110: 11,
            130: 14,
            160: 18,
            180: 22,
            200: 26,
            250: 30,
            300: 35,
        }

        # Interpolation method
        # NOTE does not work as well as the dictionary lookahead method, likely to cause crashes.

        # speedBoundList = [0, 90, 110, 130, 160, 180, 200, 250, 300]
        # lookaheadList = [5, 11, 13, 15, 18, 22, 25, 28, 32]

        # interpolationFunction = interp1d(speedBoundList, lookaheadList)
        # return int(interpolationFunction(speed))

        for speed_upper_bound, num_points in speed_to_lookahead_dict.items():
            if speed < speed_upper_bound:
                return num_points
        return 8

    def get_lookahead_index(self, speed):
        """
        Adds the lookahead waypoint to the current waypoint and normalizes it so that the value does not go out of bounds
        """
        num_waypoints = self.get_lookahead_value(speed)
        # print("speed " + str(speed)
        #       + " cur_ind " + str(self.current_waypoint_idx)
        #       + " num_points " + str(num_waypoints)
        #       + " index " + str((self.current_waypoint_idx + num_waypoints) % len(self.maneuverable_waypoints)) )
        return (self.current_waypoint_idx + num_waypoints) % len(
            self.maneuverable_waypoints
        )

    # def get_lateral_pid_config(self):
    #     """
    #     Returns the PID values for the lateral (steering) PID
    #     """
    #     with open(
    #         f"{os.path.dirname(__file__)}\\configs\\LatPIDConfig.json", "r"
    #     ) as file:
    #         config = json.load(file)
    #     return config

    # The idea and code for averaging points is from smooth_waypoint_following_local_planner.py (Summer 2023)
    def next_waypoint_smooth(self, current_speed: float):
        """
        If the speed is higher than 70, 'smooth out' the path that the car will take
        """
        if current_speed > 70 and current_speed < 300:
            target_waypoint = self.average_point(current_speed)
        else:
            new_waypoint_index = self.get_lookahead_index(current_speed)
            target_waypoint = self.maneuverable_waypoints[new_waypoint_index]

        return target_waypoint

    def average_point(self, current_speed):
        """
        Returns a new averaged waypoint based on the location of a number of other waypoints
        """
        next_waypoint_index = self.get_lookahead_index(current_speed)
        lookahead_value = self.get_lookahead_value(current_speed)
        num_points = lookahead_value * 2

        # Section specific tuning
        if self.current_section == 0:
            num_points = round(lookahead_value * 1.5)
        if self.current_section == 3:
            next_waypoint_index = self.current_waypoint_idx + 22
            num_points = 35
        if self.current_section == 4:
            num_points = lookahead_value + 5
            next_waypoint_index = self.current_waypoint_idx + 24
        if self.current_section == 5:
            # num_points = round(lookahead_value * 1.1)
            num_points = lookahead_value
        if self.current_section == 6:
            num_points = 5
            next_waypoint_index = self.current_waypoint_idx + 28
        if self.current_section == 7:
            # Jolt between sections 6 and 7 likely due to the differences in lookahead values and steering multipliers. 
            num_points = round(lookahead_value * 1.25)
        if self.current_section == 9:
            (self.current_waypoint_idx + 8) % len(self.maneuverable_waypoints)
            num_points = 0

        start_index_for_avg = (next_waypoint_index - (num_points // 2)) % len(
            self.maneuverable_waypoints
        )

        next_waypoint = self.maneuverable_waypoints[next_waypoint_index]
        next_location = next_waypoint.location

        sample_points = [
            (start_index_for_avg + i) % len(self.maneuverable_waypoints)
            for i in range(0, num_points)
        ]
        if num_points > 3:
            location_sum = reduce(
                lambda x, y: x + y,
                (self.maneuverable_waypoints[i].location for i in sample_points),
            )
            num_points = len(sample_points)
            new_location = location_sum / num_points
            shift_distance = np.linalg.norm(next_location - new_location)
            max_shift_distance = 2.0
            if self.current_section == 1:
                max_shift_distance = 0.2
            if shift_distance > max_shift_distance:
                uv = (new_location - next_location) / shift_distance
                new_location = next_location + uv * max_shift_distance

            target_waypoint = roar_py_interface.RoarPyWaypoint(
                location=new_location,
                roll_pitch_yaw=np.ndarray([0, 0, 0]),
                lane_width=0.0,
            )
            # if next_waypoint_index > 1900 and next_waypoint_index < 2300:
            #   print("AVG: next_ind:" + str(next_waypoint_index) + " next_loc: " + str(next_location)
            #       + " new_loc: " + str(new_location) + " shift:" + str(shift_distance)
            #       + " num_points: " + str(num_points) + " start_ind:" + str(start_index_for_avg)
            #       + " curr_speed: " + str(current_speed))

        else:
            target_waypoint = self.maneuverable_waypoints[next_waypoint_index]

        return target_waypoint