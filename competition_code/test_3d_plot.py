import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

def plot_vertical_planes(x, y, z, d, plane_width=0.2, orient='x', alpha=0.6):
    """
    Draw a vertical plane (a thin rectangular wall) at each (x[i], y[i]),
    rising from z=0 up to z[i]. Color is green if d[i] is True, else red.

    orient: 'x'  -> plane is parallel to YZ (constant X, vary Y across width)
            'y'  -> plane is parallel to XZ (constant Y, vary X across width)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    d = np.asarray(d, dtype=bool)
    assert x.shape == y.shape == z.shape == d.shape, "x, y, z, d must be same length"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for xi, yi, zi, di in zip(x, y, z, d):
        zi = float(zi)
        if orient == 'x':
            # constant X = xi; vary Y across the width
            X = np.array([[xi, xi],
                          [xi, xi]])
            Y = np.array([[yi - plane_width/2, yi + plane_width/2],
                          [yi - plane_width/2, yi + plane_width/2]])
        elif orient == 'y':
            # constant Y = yi; vary X across the width
            X = np.array([[xi - plane_width/2, xi + plane_width/2],
                          [xi - plane_width/2, xi + plane_width/2]])
            Y = np.array([[yi, yi],
                          [yi, yi]])
        else:
            raise ValueError("orient must be 'x' or 'y'")

        # vertical from z=0 to z=zi
        Z = np.array([[0.0, 0.0],
                      [zi,  zi]])

        color = 'green' if di else 'red'
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha, edgecolor='none')

    ax.set_xlabel('Position X (m)')
    ax.set_ylabel('Position Y (m)')
    ax.set_zlabel('Velocity (kph)')
    plt.tight_layout()
    plt.show()

x = [0, 1, 2]
y = [0, 0.5, 1.0]
z = [1.2, 2.0, 0.8]    # heights
d = [True, False, True]
plot_vertical_planes(x, y, z, d, plane_width=0.3, orient='x', alpha=1)
