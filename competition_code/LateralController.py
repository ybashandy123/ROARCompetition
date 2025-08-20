import numpy as np
import SpeedData
import math
import time


def normalize_rad(rad: float):
    return rad % (2 * np.pi)

STEERING = { # CAS steering for each section
    0: 0.01,
    1: 0.01,
    2: 0.01,
    3: 0.01,
    4: 0.01,
    5: 0.025,
    6: 0.01,
    7: 0.01,
    8: 0.01,
    9: 0.01
}

CAS_INFO = {
    "cas_ticks": 0,
    "cas_steer": 0
}

class LatController:
    def run(self, vehicle_location, vehicle_rotation, next_waypoint, current_section, time_to_hit, dir) -> float:
        """
        Calculates the steering command using the pure pursuit algorithm.

        Args:
            vehicle_location (np.array): Current vehicle location [x, y].
            vehicle_rotation (float): Current vehicle rotation (yaw) in radians.
            next_waypoint (Waypoint): Next waypoint to track.

        Returns:
            steering_command (float): Steering command in radians.
        """

        steering_command = 0

        if time_to_hit > SpeedData.SpeedData.TTH_THRESHOLD[current_section] and CAS_INFO["cas_ticks"] <= 0:
            # Calculate vector pointing from vehicle to next waypoint
            waypoint_vector = np.array(next_waypoint.location) - np.array(vehicle_location)

            # Project waypoint vector onto heading vector to find lookahead point
            distance_to_waypoint = np.linalg.norm(waypoint_vector)
            if distance_to_waypoint == 0:
                return 0  # Prevent division by zero

            waypoint_vector_normalized = waypoint_vector / distance_to_waypoint

            # Calculate steering command
            alpha = normalize_rad(vehicle_rotation[2]) - normalize_rad(
                math.atan2(waypoint_vector_normalized[1], waypoint_vector_normalized[0])
            )

            steering_command = 1.5 * math.atan2(
                2.0 * 4.7 * math.sin(alpha) / distance_to_waypoint, 1.0
            )
        else:
            print("STEERING AWAY")
            if CAS_INFO["cas_ticks"] <= 0:
                CAS_INFO["cas_ticks"] = 4
                steering_command = STEERING[current_section] * -np.sign(vehicle_rotation[2])
                if current_section == 5:
                    steering_command = -steering_command
                steering_command = np.sign(steering_command) * np.clip(np.abs(steering_command), 0, STEERING[current_section])
                CAS_INFO["cas_steer"] = steering_command
            else:
                steering_command = CAS_INFO["cas_steer"]
                CAS_INFO["cas_ticks"] -= 1

        return float(steering_command)
