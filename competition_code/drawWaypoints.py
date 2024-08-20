import matplotlib.pyplot as plt
import numpy as np
import roar_py_interface
from typing import List
from matplotlib.backend_bases import MouseButton

baseSection = []
replacementSection = []
findTarget = False
waypoints = roar_py_interface.RoarPyWaypoint.load_waypoint_list(
    np.load("competition_code\\waypoints\\waypointsPrimary.npz")
)

plt.figure(figsize=(12, 12))
plt.axis((-1100, 1100, -1100, 1100))
plt.tight_layout()

def drawWaypoints(waypointsBase, additionalWaypoints=waypoints[:1]):
    for i in waypointsBase:
        plt.plot(i.location[0], i.location[1], "ro")

    for i in additionalWaypoints:
        plt.plot(i.location[0], i.location[1], "g^")
        
drawWaypoints(waypoints)
plt.show()