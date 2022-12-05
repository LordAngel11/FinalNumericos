import numpy as np
import matplotlib.pyplot as plt      #librerias utilizadas
import ipywidgets as widgets
import sys
from scipy.integrate import odeint
from matplotlib.patches import Circle
import matplotlib.animation as animation
import cv2
#from matplotlib.animation import FuncAnimation, writers


def acel( pos, mass, G, softening ):

    # Posiciones [x,y] para todas las partículas
    x = pos[:,0:1]
    y = pos[:,1:2]

    # Matriz que guarda las diferencias entre la posiciones de cada particula: r_j - r_i
    dx = x.T - x
    dy = y.T - y

    # Matriz que calcula la norma entre cada partícula
    r = (dx**2 + dy**2 + softening**2)
    r[r>0] = r[r>0]**(-1.5)

    # Obtenemos el vector de aceleración en x e y
    ax = G * (dx * r) @ mass
    ay = G * (dy * r) @ mass

    # Unimos la aceleración en un único vector
    a = np.hstack((ax,ay))

    return a


# Condiciones iniciales:

t0         = 0       # Tiempo inicial
tf      = 5.0       # Tiempo final
dt        = 0.1    # Saltos de tiempo
softening = 0.1      # Suavizado
G         = 1.0      # Constante de gravitación universal
randvalues = False  # Activar condiciones iniciales aleatorias


if randvalues:
    N    = 4                              # Numero de cuerpos
    mass = 50.0*np.ones((N,1))/N           # Masa de las partículas
    pos  = np.random.uniform(-1,1,(N,2))   # Posiciones aleatorias
    vel  = np.random.uniform(-2,2,(N,2))   # Velocidades aleatorias
else:
    N = 4
    mass = np.array([[5.0],[5.0],[5.0],[5.0]])                   # Masa de las partículas
    pos  = np.array([[1.0, 0.0],[0.0, 1.0],[-1.0, 0.0],[0.0, -1.0]])   # Posiciones
    vel  = np.array([[0.0, 2.0],[-2.0, 0.0],[0.0, -2.0], [2.0, 0.0]])          # Velocidades


Nt = int(np.ceil(tf/dt)) # No. de intervalos

pos_save = np.zeros((N,2,Nt+1))       # Vector donde almacenamos las posiciones en cada instante
vel_save = np.zeros((N,2,Nt+1))       # Vector donde almacenamos las velocidades en cada instante
posx_mean = np.zeros(Nt+1)            # Media de las posiciones en x
posy_mean = np.zeros(Nt+1)            # Media de las posiciones en y
pos_save[:,:,0] = pos
vel_save[:,:,0] = vel                 # Se guardan en el tiempo t=0 las condiciones iniciales
posx_mean[0] = pos_save[:,0,0].mean()
posy_mean[0] = pos_save[:,1,0].mean()

# Método de Euler
for i in range(Nt):
    acc = acel( pos, mass, G, softening )    # Obtenemos la aceleracion actual
    vel += acc * dt/2                        # Actualizamos velocidades
    pos += vel * dt/2                        # Actualizamos posiciones
    pos_save[:,:,i] = pos                    # Guardamos posiciones
    vel_save[:,:,i] = vel                    # Guardamos velocidades
    posx_mean[i] = pos_save[:,0,i].mean()    # Guardamos la media de las posiciones en x
    posy_mean[i] = pos_save[:,1,i].mean()    # Guardamos la media de las posiciones en y




# Función de animación:
fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
ax = fig.add_subplot(111)

def graf(j):
    ax.clear()
    if j > 50:
        sys.exit()
    ax.set_xlim(0*posx_mean[j] - 2.5,0*posx_mean[j] + 2.5)
    ax.set_ylim(0*posy_mean[j] - 2.5,0*posy_mean[j] + 2.5)

    if j < 5:
        xx = pos_save[:,0,max(j-100,0):j+1]
        yy = pos_save[:,1,max(j-100,0):j+1]
    else:
        xx = pos_save[:,0,j-5:j+1]
        yy = pos_save[:,1,j-5:j+1]

    #plt.scatter(xx,yy,s=1,color=[.9,0.5,0.5])

    ax.plot(xx,yy,'r.')
    ax.plot(pos_save[:,0,j], pos_save[:,1,j], 'ro')


#Animacion
anim = animation.FuncAnimation(fig, graf,
                          interval = 1)



writervideo = animation.FFMpegWriter(fps=60)
anim.save('MovimientoDeParticulas.mp4', writer=writervideo)


plt.close()
