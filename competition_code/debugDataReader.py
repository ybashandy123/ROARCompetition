import matplotlib.pyplot as plt
import numpy as np
import roar_py_interface
from typing import List
from matplotlib.backend_bases import MouseButton
import json
import os

data = json.load(open(f"{os.path.dirname(__file__)}\\debugData\\debugData.json"))

def distanceToWaypoint(currentWaypoint, firstWaypoint):
    return np.linalg.norm(currentWaypoint[:2] - firstWaypoint[:2])

plt.figure(figsize=(24, 13.5))
plt.axis((-1150, 1150, -1150, 1150))

for i in data:
    if int(i) == 0 or int(i) > 1000:
        color = (0, 0, 0)
        brakeVal = data[i]["brake"]
        throttleVal = data[i]["throttle"]
        
        if brakeVal > 0:
            color = (brakeVal, 0, 0)
        else:
            color = (0, throttleVal, 0)
            
        print(f"Plotting point {i} with color {color} at ({data[i]['loc'][0]}, {data[i]['loc'][1]})")
        
        plt.plot(data[i]["loc"][0], data[i]["loc"][1], "o", color=color)

plt.show()
