import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Datos del sistema Lorenz usando Runge-Kutta 4 ---
def lorenz_system(x, y, z, sigma, rho, beta):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz

def runge_kutta_4(x0, y0, z0, t0, tf, h, sigma, rho, beta):
    n = int((tf - t0) / h)
    t = np.linspace(t0, tf, n+1)
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    z = np.zeros(n+1)
    x[0], y[0], z[0] = x0, y0, z0

    for i in range(n):
        dx1, dy1, dz1 = lorenz_system(x[i], y[i], z[i], sigma, rho, beta)
        dx2, dy2, dz2 = lorenz_system(x[i]+h*dx1/2, y[i]+h*dy1/2, z[i]+h*dz1/2, sigma, rho, beta)
        dx3, dy3, dz3 = lorenz_system(x[i]+h*dx2/2, y[i]+h*dy2/2, z[i]+h*dz2/2, sigma, rho, beta)
        dx4, dy4, dz4 = lorenz_system(x[i]+h*dx3, y[i]+h*dy3, z[i]+h*dz3, sigma, rho, beta)

        x[i+1] = x[i] + (h/6)*(dx1 + 2*dx2 + 2*dx3 + dx4)
        y[i+1] = y[i] + (h/6)*(dy1 + 2*dy2 + 2*dy3 + dy4)
        z[i+1] = z[i] + (h/6)*(dz1 + 2*dz2 + 2*dz3 + dz4)

    return t, x, y, z

# Parámetros
sigma, rho, beta = 10, 28, 8/3
x0, y0, z0 = 1, 1, 1
h, t_max = 0.01, 30

t, x, y, z = runge_kutta_4(x0, y0, z0, 0, t_max, h, sigma, rho, beta)

# --- Animación 3D ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-20, 20)
ax.set_ylim(-30, 30)
ax.set_zlim(0, 50)
ax.set_title("Runge-Kutta 4 - Sistema de Lorenz")

line, = ax.plot([], [], [], lw=2)
point, = ax.plot([], [], [], 'bo')

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])
    return line, point

def update(i):
    line.set_data(x[:i], y[:i])
    line.set_3d_properties(z[:i])
    point.set_data(x[i], y[i])
    point.set_3d_properties(z[i])
    return line, point

ani = FuncAnimation(fig, update, frames=len(x), init_func=init, blit=True, interval=30)
ani.save('lorenz_kutta.mp4', writer='ffmpeg')
print("🎥 Animación guardada como lorenz_kutta.mp4")
