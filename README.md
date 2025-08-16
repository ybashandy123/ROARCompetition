# ROAR_Competition
## Requirements

Requires `progress` module and ROAR_PY

## Project description

This submission is based on Derek Chen's previous first place solution.

For our solution we:
Decreased unnecessary braking in most sections of the track using a neural network to choose braking amounts for each section.

Are computing instantaneous radii for each waypoint on the track to increase target speed accuracy and decrease compute times for each tick (WIP)

Will add a CAS (Collision avoidance system) based on open source Carla CAS systems using LiDar (although this is not implemented yet due to issues with LiDar sensors)
CAS should not be activated unless a collision is imminent, and is just used to lower the probability of crashes despite actually raising race times in practice.

