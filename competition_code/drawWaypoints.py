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

for waypoint in track[:] if track is not None else []:
    rep_line = waypoint.line_representation
    rep_line = np.asarray(rep_line)
    waypoint_heading = tr3d.euler.euler2mat(*waypoint.roll_pitch_yaw) @ np.array(
        [1, 0, 0]
    )
    plt.plot(rep_line[:, 0], rep_line[:, 1], "k", linewidth=2)
    plt.arrow(
        waypoint.location[0],
        waypoint.location[1],
        waypoint_heading[0] * 1,
        waypoint_heading[1] * 1,
        width=0.5,
        color="r",
    )
    progressBar.next()

    for i in additionalWaypoints:
        plt.plot(i.location[0], i.location[1], "g^")
        
drawWaypoints(waypoints)
plt.show()