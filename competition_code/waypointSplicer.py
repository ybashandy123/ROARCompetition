import matplotlib.pyplot as plt
import numpy as np
import roar_py_interface
from typing import List
from matplotlib.backend_bases import MouseButton

baseSection = []
replacementSection = []
findTarget = False
baseWaypoints = roar_py_interface.RoarPyWaypoint.load_waypoint_list(
    np.load("competition_code\\waypoints\\waypoints10.npz")
)

replacementWaypoints = roar_py_interface.RoarPyWaypoint.load_waypoint_list(
    np.load("competition_code\\waypoints\\waypoints5.npz")
)

newWaypoints = []


def drawWaypoints(waypoints, additionalWaypoints=baseWaypoints[:1]):
    plt.figure(figsize=(24, 13.5))

    for i in waypoints:
        plt.plot(i.location[0], i.location[1], "ro")

    for i in additionalWaypoints:
        plt.plot(i.location[0], i.location[1], "g^")

    plt.axis((-1150, 1150, -1150, 1150))
    plt.connect("button_press_event", on_click)
    plt.show()


def distanceToWaypoint(currentLoc, waypoint: roar_py_interface.RoarPyWaypoint):
    return np.linalg.norm(currentLoc[:2] - waypoint.location[:2])


def findCurrentIndex(currentLoc, waypoints: List[roar_py_interface.RoarPyWaypoint]):
    for i in range(0, len(waypoints)):
        if distanceToWaypoint(currentLoc, waypoints[i % len(waypoints)]) < 3:
            return i % len(waypoints)
    return 0


def on_click(event):
    if event.button is MouseButton.LEFT:
        if len(baseSection) < 2:
            baseSection.append(
                findCurrentIndex([event.xdata, event.ydata], replacementWaypoints)
            )
            print(
                f"Plotting source waypoint {len(baseSection)} at ({event.xdata}, {event.ydata})"
            )
            plt.plot(event.xdata, event.ydata, "bs")
        else:
            plt.close()
            for i in range(len(baseSection)):
                print(f"Finding closest waypoint {i + 1}")
                currentLoc = baseWaypoints[baseSection[i]]
                closestWaypoint = replacementWaypoints[0]
                lowestDistance = distanceToWaypoint(
                    currentLoc.location, replacementWaypoints[0]
                )
                for j in replacementWaypoints:
                    currentDist = distanceToWaypoint(currentLoc.location, j)
                    if currentDist < lowestDistance:
                        closestWaypoint = j
                        lowestDistance = currentDist
                replacementSection.append(
                    findCurrentIndex(closestWaypoint.location, replacementWaypoints)
                )
                print(
                    f"Target waypoint {i + 1} found at {closestWaypoint.location}, {lowestDistance} units away"
                )

            print(
                f"Source Waypoints: {len(baseSection)}\nTarget Waypoints: {len(replacementSection)}"
            )


drawWaypoints(baseWaypoints)

for i in range(baseSection[0]):
    newWaypoints.append(baseWaypoints[i])

for i in range(replacementSection[0], replacementSection[1], 1):
    newWaypoints.append(replacementWaypoints[i])

for i in range(replacementSection[1], len(baseWaypoints), 1):
    newWaypoints.append(baseWaypoints[i])

print(f"Waypoints successfully spliced, saving as modifiedWaypoints.npz")
np.savez_compressed(
    "modifiedWaypoints.npz",
    **roar_py_interface.RoarPyWaypoint.save_waypoint_list(newWaypoints),
)
