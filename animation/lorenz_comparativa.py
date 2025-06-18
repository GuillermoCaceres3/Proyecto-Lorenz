import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg' 

#EULER
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

#KUTTA
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
        dx2, dy2, dz2 = lorenz_system(x[i] + h*dx1/2, y[i] + h*dy1/2, z[i] + h*dz1/2, sigma, rho, beta)
        dx3, dy3, dz3 = lorenz_system(x[i] + h*dx2/2, y[i] + h*dy2/2, z[i] + h*dz2/2, sigma, rho, beta)
        dx4, dy4, dz4 = lorenz_system(x[i] + h*dx3, y[i] + h*dy3, z[i] + h*dz3, sigma, rho, beta)
        x[i+1] = x[i] + (h/6)*(dx1 + 2*dx2 + 2*dx3 + dx4)
        y[i+1] = y[i] + (h/6)*(dy1 + 2*dy2 + 2*dy3 + dy4)
        z[i+1] = z[i] + (h/6)*(dz1 + 2*dz2 + 2*dz3 + dz4)
    return t, x, y, z

#ANIMACION DE COMPARACION
def crear_animacion_comparativa(ts, datos_euler, datos_rk4, filename, duration_sec=720, fps=10):
    frames_total = duration_sec * fps
    xs1, ys1, zs1 = datos_euler
    xs2, ys2, zs2 = datos_rk4

    indices = np.linspace(0, len(ts)-1, frames_total, dtype=int)
    xs1, ys1, zs1 = xs1[indices], ys1[indices], zs1[indices]
    xs2, ys2, zs2 = xs2[indices], ys2[indices], zs2[indices]

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.set_title("Método de Euler")
    ax2.set_title("Método de Runge-Kutta 4")

    for ax, xs, ys, zs in zip((ax1, ax2), (xs1, xs2), (ys1, ys2), (zs1, zs2)):
        ax.set_xlim((min(xs), max(xs)))
        ax.set_ylim((min(ys), max(ys)))
        ax.set_zlim((min(zs), max(zs)))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    line1, = ax1.plot([], [], [], lw=1.2, color='tab:blue')
    point1, = ax1.plot([], [], [], 'ro')
    line2, = ax2.plot([], [], [], lw=1.2, color='tab:green')
    point2, = ax2.plot([], [], [], 'ro')

    def init():
        for line, point in [(line1, point1), (line2, point2)]:
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
        return line1, point1, line2, point2

    def update(num):
        line1.set_data(xs1[:num], ys1[:num])
        line1.set_3d_properties(zs1[:num])
        point1.set_data(xs1[num-1:num], ys1[num-1:num])
        point1.set_3d_properties(zs1[num-1:num])

        line2.set_data(xs2[:num], ys2[:num])
        line2.set_3d_properties(zs2[:num])
        point2.set_data(xs2[num-1:num], ys2[num-1:num])
        point2.set_3d_properties(zs2[num-1:num])
        return line1, point1, line2, point2

    ani = animation.FuncAnimation(fig, update, frames=frames_total, init_func=init, interval=1000/fps, blit=True)
    writer = animation.FFMpegWriter(fps=fps)
    ani.save(filename, writer=writer)
    plt.close()

#EJECUTA
def generar_animacion_comparativa():
    x0, y0, z0 = 2.0, 0.009, 0.7
    sigma = 9
    rho = 18
    beta = 10 / 3
    h = 0.01
    T = 60  # sistema simulado por 60 s

    ts1, xs1, ys1, zs1 = lorenz_euler(x0, y0, z0, sigma, rho, beta, h, T)
    ts2, xs2, ys2, zs2 = runge_kutta_4(x0, y0, z0, 0, T, h, sigma, rho, beta)

    crear_animacion_comparativa(ts1, (xs1, ys1, zs1), (xs2, ys2, zs2), "lorenzVS.mp4", duration_sec=60, fps=30)
    print("Animación comparativa guardada como 'lorenzVS.mp4'")

#Ejecuta
if __name__ == "__main__":
    generar_animacion_comparativa()
