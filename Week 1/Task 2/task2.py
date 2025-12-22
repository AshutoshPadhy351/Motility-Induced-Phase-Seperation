import numpy as np
import matplotlib.pyplot as plt

N = 100
L = 15.0 
v0 = 1.0
Dr = 0.1
mu = 1.0
k = 100.0
dt = 0.01
steps = 20000 
a = 1.0 

x = np.random.uniform(0, L, N)
y = np.random.uniform(0, L, N)
theta = np.random.uniform(0, 2*np.pi, N)

x_raw = x.copy()
y_raw = y.copy()
x0 = x.copy()
y0 = y.copy()

times = []
msd = []

for s in range(steps):
    fx = np.zeros(N)
    fy = np.zeros(N)
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            
            if dx > L/2: dx -= L
            if dx < -L/2: dx += L
            if dy > L/2: dy -= L
            if dy < -L/2: dy += L
            
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < a:
                f_mag = k * (a - dist)
                fx[i] -= f_mag * (dx/dist)
                fy[i] -= f_mag * (dy/dist)
                fx[j] += f_mag * (dx/dist)
                fy[j] += f_mag * (dy/dist)

    theta += np.random.normal(0, np.sqrt(2 * Dr * dt), N)
    
    vx = v0 * np.cos(theta) + mu * fx
    vy = v0 * np.sin(theta) + mu * fy
    
    x += vx * dt
    y += vy * dt
    x_raw += vx * dt
    y_raw += vy * dt
    
    x %= L
    y %= L
    
    if s % 50 == 0:
        d2 = (x_raw - x0)**2 + (y_raw - y0)**2
        msd.append(np.mean(d2))
        times.append(s * dt)

plt.figure()
plt.loglog(times, msd, label='sim')
t_vals = np.array(times)
plt.loglog(t_vals, (v0**2)*(t_vals**2), '--', label='t2')
plt.xlabel('t')
plt.ylabel('msd')
plt.legend()
plt.savefig('msd_plot.png')

plt.figure(figsize=(5,5))
plt.scatter(x, y, s=10)
plt.xlim(0, L)
plt.ylim(0, L)
plt.title('Final Positions')
plt.savefig('snapshot.png')
plt.show()
