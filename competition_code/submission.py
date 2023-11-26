"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from collections import deque
import math
from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np

def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def filter_waypoints(location : np.ndarray, currentIdx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    #Takes in the current car location, the index of the waypoints list, and the waypoint list to get the closest waypoint
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(currentIdx, len(waypoints) + currentIdx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 2:
            return i % len(waypoints)
    return currentIdx

class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverableWaypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        cameraSensor : roar_py_interface.RoarPyCameraSensor = None,
        locationSensor : roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocitySensor : roar_py_interface.RoarPyVelocimeterSensor = None,
        rpySensor : roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancyMapSensor : roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverableWaypoints = maneuverableWaypoints
        self.vehicle = vehicle
        self.camera_sensor = cameraSensor
        self.location_sensor = locationSensor
        self.velocity_sensor = velocitySensor
        self.rpy_sensor = rpySensor
        self.occupancy_map_sensor = occupancyMapSensor
        self.collision_sensor = collision_sensor
    
    async def initialize(self) -> None:
        # TODO: You can do some initial computation here if you want to.
        # For example, you can compute the path to the first waypoint.

        # Receive location, rotation and velocity data 
        vehicleLocation = self.location_sensor.get_last_gym_observation()
        vehicleRotation = self.rpy_sensor.get_last_gym_observation()
        vehicleVelocity = self.velocity_sensor.get_last_gym_observation()

        self.currentWaypointIdx = 10
        self.currentWaypointIdx = filter_waypoints(
            vehicleLocation,
            self.currentWaypointIdx,
            self.maneuverableWaypoints, 
        )

        # Creates the section dividers and sets the current zone to 0
        # Zone 0: Start - after turn 4
        # Zone 1: Turn 5 - Before turn 6
        # Zone 2: Turn6 - Turn 7
        # Zone 3: After turn 7 - Before turn 9 (Long 180 degree turn)
        # Zone 4: Turns 9 and 10 (Sharp S-turn after long straightaway)
        self.regions = [[740, 720], [90, 215], [-80, -160], [-345, 0], [-290, 400]] # (-290, 400) is the start of the track
        self.currentRegion = 0
        self.brakeLocations = [[75, 210], [-125, -950], [-340, 210]]
        self.currentBrakeLocation = 0

    
    async def step(
        self
    ) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        # TODO: Implement your solution here.

        # Receive location, rotation and velocity data 
        vehicleLocation = self.location_sensor.get_last_gym_observation()
        vehicleRotation = self.rpy_sensor.get_last_gym_observation()
        vehicleVelocity = self.velocity_sensor.get_last_gym_observation()
        vehicleVelocityNorm = np.linalg.norm(vehicleVelocity)
        
        # Find the waypoint closest to the vehicle
        self.currentWaypointIdx = filter_waypoints(
            vehicleLocation,
            self.currentWaypointIdx,
            self.maneuverableWaypoints, 
        )
        # Generates the waypoint to follow based on the vehicle's speed
        waypointToFollow = self.maneuverableWaypoints[(self.currentWaypointIdx + int(vehicleVelocityNorm / 2.75) + 5) % len(self.maneuverableWaypoints)]

        # Calculate delta vector towards the target waypoint
        vectorToWaypoint = (waypointToFollow.location - vehicleLocation)[:2]
        heading_to_waypoint = np.arctan2(vectorToWaypoint[1],vectorToWaypoint[0])

        # Calculate delta angle towards the target waypoint
        deltaHeading = normalize_rad(heading_to_waypoint - vehicleRotation[2])

        # Proportional controller to steer the vehicle towards the target waypoint
        steerControl = (
            -8.0 / np.sqrt(vehicleVelocityNorm) * deltaHeading / np.pi
        ) if vehicleVelocityNorm > 1e-2 else -np.sign(deltaHeading)
        steerControl = np.clip(steerControl, -1.0, 1.0)

        # Calculates the distance to the final two turns of the track 

        nextRegionDistance = math.sqrt((self.regions[self.currentRegion % len(self.regions)][0] - waypointToFollow.location[0]) ** 2 + (self.regions[self.currentRegion % len(self.regions)][1] - waypointToFollow.location[1]) ** 2)
        nextBrakeDistance = math.sqrt((self.brakeLocations[self.currentBrakeLocation % len(self.brakeLocations)][0] - waypointToFollow.location[0]) ** 2 + (self.brakeLocations[self.currentBrakeLocation % len(self.brakeLocations)][1] - waypointToFollow.location[1]) ** 2)

        # Calculates the appropriate throttle response based on the speed and angle to the next waypoint and the 'sector' the car is in
        # TODO: Add better zone management and tune for zones 1 and 2
        
        if nextRegionDistance < 15:
            self.currentRegion += 1

        normalizedRegion = self.currentRegion % len(self.regions)
        
        if nextBrakeDistance < 10: 
            throttle, brake, reverse, handBrake = 1, 1, 1, 1
            self.currentBrakeLocation += 1
        elif normalizedRegion < 2:
            # Handles zones 0 and 1
            if (abs(deltaHeading) > 0.017 and vehicleVelocityNorm > 37.5):
                throttle, brake, reverse, handBrake = 1, 1, 1, 1
            else:
                throttle = 0.75 + (0.7 / deltaHeading - vehicleVelocityNorm) / 20
                brake, reverse, handBrake = 0, 0, 0
        elif normalizedRegion == 2:
            # Handles zone 2
            if (abs(deltaHeading) > 0.001 and vehicleVelocityNorm > 37.5):
                throttle, brake, reverse, handBrake = 1, 1, 1, 1
            else:
                throttle = 0.75 + (0.7 / deltaHeading - vehicleVelocityNorm) / 20
                brake, reverse, handBrake = 0, 0, 0
        elif normalizedRegion == 3:
            # Handles zone 3
            if (abs(deltaHeading) > 0.007 and vehicleVelocityNorm > 50):
                throttle, brake, reverse, handBrake = 1, 1, 1, 1
            else:
                throttle = 0.75 + (1 / deltaHeading - vehicleVelocityNorm) / 20
                brake, reverse, handBrake = 0, 0, 0
        else:
            # Handles zone 4
            if (abs(deltaHeading) > 0.0001 and vehicleVelocityNorm > 20):
                throttle, brake, reverse, handBrake = 1, 1, 1, 1
            else:
                throttle = 1
                brake, reverse, handBrake = 0, 0, 0

        gear = max(1, (int)(vehicleVelocityNorm / 20))

        control = {
            "throttle": throttle,
            "steer": steerControl,
            "brake": brake,
            "hand_brake": handBrake,
            "reverse": reverse,
            "target_gear": gear
        }

        print(f"Current Speed: {vehicleVelocityNorm}\nBrake Value: {brake}")
        print(f"Current region: {normalizedRegion}\nLap Number: {self.currentRegion // len(self.regions) + 1}\nDelta Heading: {deltaHeading}")

        await self.vehicle.apply_action(control)
        return control