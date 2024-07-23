import matplotlib.pyplot as plt
import numpy as np
import roar_py_interface
from typing import List
from matplotlib.backend_bases import MouseButton

baseSection = []
replacementSection = []
findTarget = False
baseWaypoints = roar_py_interface.RoarPyWaypoint.load_waypoint_list(
    np.load("competition_code\\waypoints\\waypointsPrimary.npz")
)

replacementWaypoints = roar_py_interface.RoarPyWaypoint.load_waypoint_list(
    np.load("competition_code\\waypoints\\oldWaypointsPrimary.npz")
)

newWaypoints = []

plt.show()
plt.ion()    
plt.figure(figsize=(24, 13.5))
plt.axis((-1150, 1150, -1150, 1150))

def drawWaypoints(waypoints, additionalWaypoints=baseWaypoints[:1]):
    for i in waypoints:
        plt.plot(i.location[0], i.location[1], "ro")

    for i in additionalWaypoints:
        plt.plot(i.location[0], i.location[1], "g^")

    plt.connect("button_press_event", on_click)

def distanceToWaypoint(currentLoc, waypoint: roar_py_interface.RoarPyWaypoint):
    return np.linalg.norm(currentLoc[:2] - waypoint.location[:2])


def findClosestIndex(currentLoc, waypoints: List[roar_py_interface.RoarPyWaypoint]):
    lowestDist = 100
    closestInd = 0
    for i in range(0, len(waypoints)):
        dist = distanceToWaypoint(currentLoc, waypoints[i % len(waypoints)])
        if dist < lowestDist:
            lowestDist = dist
            closestInd = i
    return closestInd % len(waypoints)
    # return 0


def on_click(event):
    if event.button is MouseButton.LEFT:
        if len(baseSection) < 2:
            loc = [event.xdata, event.ydata]
            baseSection.append(
                findClosestIndex(loc, baseWaypoints))
            replacementSection.append(findClosestIndex(loc, replacementWaypoints))
            print(
                f"Plotting section waypoint {len(baseSection)} at ({loc[0]:.2f}, {loc[1]:.2f})"
            )
            plt.plot(event.xdata, event.ydata, "bs")
            plt.draw()
        else:
            plt.close()

print("\n--- Generating Waypoint Map ---\n")

drawWaypoints(baseWaypoints)

while len(replacementSection) < 2:
    plt.pause(0.01)
print(f"Base waypoint indexes: {baseSection}")
print(f"New waypoint indexes: {replacementSection}")
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
