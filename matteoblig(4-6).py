import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()
line, = ax.plot([], [], '-')
#line2, = ax.plot([], [], 'r-')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
label = ax.text(0.5, 1.1, "", transform=ax.transAxes, ha="center",
        bbox={'facecolor': 'blue', 'alpha': 1.0, 'pad': 5}, )


def animate(i):
    label.set_text("Time {:.2f} s".format(i*k))
    line.set_data(x, U[:, i])
    # line2.set_data(x, np.sin(x)*np.exp(-i*k))
    return line, label,

def triangular(n: int, l: float, d: float, u: float) -> np.ndarray:
    e = np.ones((1, n))[0]
    e1 = np.ones((1, n-1))[0]
    A = d * np.diag(e, 0) + l * np.diag(e1, -1)   + u * np.diag(e1, 1)
    return A

if __name__ == '__main__':
    h = 0.01   # dx
    k = 0.001   # dt
    a = k / (h*h)

    t_end = 1.0  # Simulation end time
    n_time_steps = int(t_end / k)

    x = np.arange(0, 1+h, h)
    N = len(x)
    U = np.zeros((N, n_time_steps+1))

    # Matrices needed for explict, implicit Euler and CN
    A = triangular(N - 2, a, 1 - 2 * a, a)     # Explicit Euler
    A1 = triangular(N - 2, -a, 1 + 2 * a, -a)  # Implicit Euler
    iA1 = np.linalg.inv(A1)

    # Compute Crank-Nicholson matrix C
    I = np.eye(N-2)
    B = triangular(N - 2, -1, 2, -1)

    iB = np.linalg.inv(2 * I + a * B)
    C = iB @ (2 * I - a * B)

       # Select method
    method = 2   # 0 - Explicit Euler, 1 - Implicit Euler, 2 - CN

    # Initial profile
    U[:,0] = np.sin(x)

    # Set boundary conditions
    U[0, :], U[-1, :] = 0, 0

    j = 0
    while j < n_time_steps:
        if method == 0:
            U[1:-1, j+1] = A @ U[1:-1, j]
        elif method == 1:
            U[1:-1, j + 1] = iA1 @ U[1:-1, j]
        elif method == 2:
            U[1:-1, j+1] = C @ U[1:-1, j]
        else:
            print("Unknown method {}".format(method))
            exit(1)
        j = j + 1

    ani = animation.FuncAnimation(fig, animate, frames=n_time_steps, interval=10,  repeat=False, blit=True)
    plt.show()

