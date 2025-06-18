import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

#configurar ffmpeg para video
plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'  

#METODO EULER
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

#METODO DE KUTTA 4
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

#ANIMACION
def crear_animacion(ts, xs, ys, zs, filename, x0, y0, z0, sigma, rho, beta, h):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim((min(xs), max(xs)))
    ax.set_ylim((min(ys), max(ys)))
    ax.set_zlim((min(zs), max(zs)))
    ax.set_title("Sistema de Lorenz", fontsize=14)

    # texto parámetros
    param_text = (
        f"Condiciones iniciales:\n"
        f"x₀ = {x0}, y₀ = {y0}, z₀ = {z0}\n"
        f"σ = {sigma}, ρ = {rho}, β = {beta:.3f}\n"
        f"h = {h}"
    )
    text_box = ax.text2D(0.05, 0.95, param_text, transform=ax.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.7))

    line, = ax.plot([], [], [], lw=1.5, color='darkviolet')
    point, = ax.plot([], [], [], 'ro')

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point, text_box

    def update(num):
        line.set_data(xs[:num], ys[:num])
        line.set_3d_properties(zs[:num])
        point.set_data(xs[num-1:num], ys[num-1:num])
        point.set_3d_properties(zs[num-1:num])
        return line, point, text_box

    ani = animation.FuncAnimation(fig, update, frames=len(ts), init_func=init, interval=20, blit=True)
    writer = animation.FFMpegWriter(fps=30)
    ani.save(filename, writer=writer)
    plt.close()

#FUNCIONES
def generar_animacion_euler():
    x0, y0, z0 = 1, 1, 1
    sigma, rho, beta = 10, 28, 8/3
    h = 0.01
    t_max = 30
    ts, xs, ys, zs = lorenz_euler(x0, y0, z0, sigma, rho, beta, h, t_max)
    crear_animacion(ts, xs, ys, zs, "lorenzz_euler.mp4", x0, y0, z0, sigma, rho, beta, h)
    print("Animación método de Euler guardada como lorenzz_euler.mp4")

def generar_animacion_rk4():
    x0, y0, z0 = 1, 1, 1
    sigma, rho, beta = 10, 28, 8/3
    h = 0.01
    t0, tf = 0, 30
    ts, xs, ys, zs = runge_kutta_4(x0, y0, z0, t0, tf, h, sigma, rho, beta)
    crear_animacion(ts, xs, ys, zs, "lorenzz_rk4.mp4", x0, y0, z0, sigma, rho, beta, h)
    print("Animación método de Runge-Kutta guardada como lorenzz_rk4.mp4")

#EJECUCIÓN
#generar_animacion_euler()
#generar_animacion_rk4()
