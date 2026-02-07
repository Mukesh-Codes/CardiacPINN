import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace

# Grid and simulation setup
nx, ny = 128, 128
dx = 1.0
dt = 0.01   # smaller dt for stability
steps = 500

# Aliev–Panfilov parameters (stable & visible)
a = 0.1
k = 8.0
eps = 0.01
mu1 = 0.2
mu2 = 0.3
D = 0.2

# Initialize
V = np.zeros((nx, ny))
W = np.zeros((nx, ny))

# Stimulus: asymmetrical to create wavefront
V[40:60, 40:80] = 1.0

frames = []

for t in range(steps):
    lapV = laplace(V, mode='reflect')

    dVdt = D * lapV - k * V * (V - a) * (V - 1) - V * W
    dWdt = eps + mu1 * W / (V + mu2)

    # Update with clipping to avoid explosion
    V += dt * dVdt
    W += dt * dWdt
    V = np.clip(V, 0.0, 1.5)
    W = np.clip(W, 0.0, 1.0)

    if t % 10 == 0:
        frames.append(V.copy())

# Visualize a few frames
plt.figure(figsize=(8, 4))
for i, f in enumerate(frames[:8]):
    plt.subplot(2, 4, i+1)
    plt.imshow(f, cmap='plasma', vmin=0, vmax=1)
    plt.axis('off')
    plt.title(f"t={i*10}")
plt.tight_layout()
plt.show()
