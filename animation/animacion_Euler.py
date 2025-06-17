import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Datos del sistema Lorenz usando Euler ---
def lorenz_euler(x0, y0, z0, sigma, rho, beta, h, t_max):
    n = int(t_max / h)
    ts = np.linspace(0, t_max, n+1)
    xs, ys, zs = [x0], [y0], [z0]
    x, y, z = x0, y0, z0

    for _ in range(n):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        x += h * dx
        y += h * dy
        z += h * dz

        xs.append(x)
        ys.append(y)
        zs.append(z)

    return ts, np.array(xs), np.array(ys), np.array(zs)

# Parámetros
sigma, rho, beta = 10, 28, 8/3
x0, y0, z0 = 1, 1, 1
h, t_max = 0.01, 30

ts, xs, ys, zs = lorenz_euler(x0, y0, z0, sigma, rho, beta, h, t_max)

# --- Animación 3D ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-20, 20)
ax.set_ylim(-30, 30)
ax.set_zlim(0, 50)
ax.set_title("Método de Euler - Sistema de Lorenz")

line, = ax.plot([], [], [], lw=2)
point, = ax.plot([], [], [], 'ro')

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])
    return line, point

def update(i):
    line.set_data(xs[:i], ys[:i])
    line.set_3d_properties(zs[:i])
    point.set_data(xs[i], ys[i])
    point.set_3d_properties(zs[i])
    return line, point

ani = FuncAnimation(fig, update, frames=len(xs), init_func=init, blit=True, interval=30)
ani.save('lorenz_euler.mp4', writer='ffmpeg')
print("🎥 Animación guardada como lorenz_euler.mp4")
