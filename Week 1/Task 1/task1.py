import numpy as np
import matplotlib.pyplot as plt
import json
import time

# Simulation parameters
seed = 42
np.random.seed(seed)

N = 500             # number of particles
a = 1.0             # particle diameter
phi = 0.3           # area fraction
v0 = 1.0            # self-propulsion speed
Dr = 0.1            # rotational diffusion coefficient
dt = 0.1            # time step
tmax = 1000.0       # total simulation time
sample_every = 10   # record MSD every this many steps

area_per_particle = np.pi * (a/2.0)**2
total_area = N * area_per_particle / phi
L = np.sqrt(total_area)   # box side length
steps = int(np.ceil(tmax / dt))

print("N =", N, "phi =", phi, "L =", round(L, 3))
print("v0 =", v0, "Dr =", Dr, "dt =", dt, "steps =", steps)

params = dict(N=N, a=a, phi=phi, v0=v0, Dr=Dr, dt=dt, tmax=tmax,
              sample_every=sample_every, L=L, seed=seed)
with open("params.json", "w") as f:
    json.dump(params, f, indent=2)
print("Saved simulation parameters to params.json")

# positions in [0, L)
pos = np.random.rand(N, 2) * L
theta = np.random.rand(N) * 2 * np.pi

# unwrapped positions for MSD
pos_unwrapped = pos.copy()
pos0 = pos_unwrapped.copy()

# Pre-allocate arrays for time and MSD
n_records = steps // sample_every + 1
times = np.zeros(n_records)
msd = np.zeros(n_records)

times[0] = 0.0
msd[0] = 0.0

sqrt2Dr_dt = np.sqrt(2.0 * Dr * dt)
record_idx = 1
t0 = time.time()

for istep in range(1, steps + 1):
    # rotational diffusion
    dtheta = np.random.normal(0.0, 1.0, size=N) * sqrt2Dr_dt
    theta += dtheta

    # displacement
    dx = v0 * dt * np.cos(theta)
    dy = v0 * dt * np.sin(theta)

    pos[:, 0] += dx
    pos[:, 1] += dy

    pos_unwrapped[:, 0] += dx
    pos_unwrapped[:, 1] += dy

    # periodic boundaries
    pos %= L

    if (istep % sample_every) == 0:
        t = istep * dt
        dr2 = (pos_unwrapped - pos0)**2
        dr2_sum = dr2.sum(axis=1)
        msd_val = dr2_sum.mean()
        times[record_idx] = t
        msd[record_idx] = msd_val
        record_idx += 1

        if record_idx % (n_records // 10 + 1) == 0:
            elapsed = time.time() - t0
            pct = 100.0 * istep / steps
            print(f"step {istep}/{steps} ({pct:.1f}%), t={t:.1f}, msd={msd_val:.3f}, elapsed={elapsed:.1f}s")

# reference scalings
ballistic = (v0**2) * times**2
diffusive = (2.0 * v0**2 / Dr) * times

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# linear
ax = axs[0]
ax.plot(times, msd, label="MSD (sim)")
ax.plot(times, ballistic, '--', label="~ t^2")
ax.plot(times, diffusive, '--', label="~ t")
ax.set_xlabel("time")
ax.set_ylabel("MSD")
ax.set_title("MSD vs time (linear)")
ax.legend()
ax.grid(True)

# log-log
ax = axs[1]
mask = times > 0
ax.loglog(times[mask], msd[mask], label="MSD (sim)")
ax.loglog(times[mask], ballistic[mask], '--', label="~ t^2")
ax.loglog(times[mask], diffusive[mask], '--', label="~ t")
ax.set_xlabel("time")
ax.set_ylabel("MSD")
ax.set_title("MSD vs time (log-log)")
ax.legend()
ax.grid(True, which="both", ls=":")

plt.tight_layout()
plt.savefig("msd_task1.png", dpi=200)
print("Saved plot to msd_task1.png")
plt.show()

# slope estimate
def log_slope(x, y, i1, i2):
    return (np.log(y[i2]) - np.log(y[i1])) / (np.log(x[i2]) - np.log(x[i1]))

slope_short = log_slope(times[mask], msd[mask], 0, min(10, len(times[mask]) - 1))
slope_long = log_slope(times[mask], msd[mask],
                       max(1, len(times[mask]) // 3),
                       len(times[mask]) - 1)

print(f"Estimated log-log slope (short times) ~ {slope_short:.2f}")
print(f"Estimated log-log slope (long times)  ~ {slope_long:.2f}")

print("Done.")