import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import roar_py_interface
from typing import List
from matplotlib.backend_bases import MouseButton
import json
import os

data = json.load(open(f"{os.path.dirname(__file__)}\\debugData\\debugData.json"))

def distanceToWaypoint(currentWaypoint, firstWaypoint):
    return np.linalg.norm(currentWaypoint[:2] - firstWaypoint[:2])

print("--- Generating Plot ---\n")

plt.figure(figsize=(12, 12))
plt.axis((-1100, 1100, -1100, 1100))
plt.margins(x=0, y=0)

for i in data:
    if int(i) == 0 or int(i) >= 750:
        color = (0, 0, 0)
        brakeVal = data[i]["brake"]
        throttleVal = data[i]["throttle"]
        
        if brakeVal > 0:
            color = (brakeVal, 0, 0)
        else:
            color = (0, throttleVal, 0)
        
        if int(i) % 100 == 0:
            print(f"Plotting point {i}")
            
        plt.plot(data[i]["loc"][0], data[i]["loc"][1], "o", color=color)

print("\n--- Plot complete ---\n")
plt.tight_layout()
plt.show()
