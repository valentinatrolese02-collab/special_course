"""
IMD-style Brownian motion on the surface of a cube (non-smooth manifold test case).
"""

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

N = 6000
h = 0.02
n_traj_steps = 4000
L = 1.0  # cube half-extent


def sample_cube(n, L, rng):
    faces = rng.integers(0, 6, size=n)
    u = rng.uniform(-L, L, size=n)
    v = rng.uniform(-L, L, size=n)
    pts = np.zeros((n, 3))
    for f in range(6):
        mask = faces == f
        axis = f // 2
        sign = 1.0 if f % 2 == 0 else -1.0
        other_axes = [a for a in range(3) if a != axis]
        pts[mask, other_axes[0]] = u[mask]
        pts[mask, other_axes[1]] = v[mask]
        pts[mask, axis] = sign * L
    return pts


def retract_to_cube(p, L):
    # Radial projection from the origin onto the cube surface (exact for
    # points near the surface; same role as the sphere's norm-normalization)
    scale = L / np.max(np.abs(p))
    return p * scale


X = sample_cube(N, L, rng)

tree = cKDTree(X)
k_bw = max(10, int(np.ceil(4 * np.log(N))))
dists_k, _ = tree.query(X, k=k_bw + 1)
epsilon = np.median(dists_k[:, k_bw])
from collections import defaultdict

neighbor_lists = defaultdict(list)
for i, j in tree.query_pairs(r=np.sqrt(epsilon)):
    neighbor_lists[i].append(j)
    neighbor_lists[j].append(i)

L_vecs = np.zeros((N, 3))
Gammas = np.zeros((N, 3, 3))
for i in range(N):
    nbrs = neighbor_lists[i]
    if len(nbrs) < 3:
        nbrs = tree.query(X[i], k=6)[1][1:]
    diffs = X[nbrs] - X[i]
    L_vecs[i] = diffs.mean(axis=0)
    cov = diffs.T @ diffs / len(nbrs)
    _, evecs = np.linalg.eigh(cov)
    V = evecs[:, -2:]
    Gammas[i] = V @ V.T

traj = np.zeros((n_traj_steps + 1, 3))
traj[0] = X[rng.integers(N)]
for l in range(n_traj_steps):
    _, i = tree.query(traj[l])
    xi = rng.normal(size=3)
    step = h * L_vecs[i] + np.sqrt(h) * (Gammas[i] @ xi)
    new_p = traj[l] + step
    traj[l + 1] = retract_to_cube(new_p, L)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*X.T, color='0.6', s=2, alpha=0.5, linewidths=0)
ax.scatter(*traj.T, color='red', s=3, alpha=0.9, linewidths=0)
ax.set_box_aspect([1, 1, 1])
ax.set_axis_off()
ax.view_init(elev=20, azim=35)
plt.tight_layout()
plt.savefig('outputs/imd_cube_brownian.png', dpi=200)
print("saved")