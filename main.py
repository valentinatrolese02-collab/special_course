"""
Reproduce the paper's "point cloud + Brownian motion overlay" figure style,
on a sphere, using plain numpy/scipy/matplotlib.

Construction:
  - point cloud = manifold samples (no equation of the sphere is used except
    for the final retraction, which stands in for a DRGD-style correction)
  - W          = explicit sparse adjacency matrix (Eq. 11),
                 built once from an eps-ball proximity graph (here computed as sparse array)
  - D          = diagonal degree matrix for each node of the graph
  - L(X)       = random-walk graph Laplacian applied to coordinates (Eq. 11)
  - Gamma_raw  = carre-du-champ (Eq. 7) applied to ambient coordinate
                 functions
  - Gammas     = Gamma_raw^(1/2), the actual matrix square root used by
                 Algorithm 1's noise term
  - Euler-Maruyama integration (Eq. 15)
"""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import KDTree
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import time
import os



# sampling
def sample_sphere(n, radius, rng):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * radius

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


## Graph construction
def compute_bandwidth(X, tree):
    """Appendix F.1: eps = median distance to the k-th nearest neighbor,
    k = max(10, ceil(4 log N))."""
    n = len(X)
    k_bw = max(10, int(np.ceil(4 * np.log(n))))
    dists_k, _ = tree.query(X, k=k_bw + 1)
    return np.median(dists_k[:, k_bw])


def build_adjacency(tree, epsilon):
    """Explicit sparse W_ij = 1{||x_i - x_j||^2 <= epsilon} (Eq. 11)."""
    D = tree.sparse_distance_matrix(tree, max_distance=np.sqrt(epsilon), p=2.0, output_type='coo_matrix')
    W = sp.coo_array(D).tocsr()
    W = (W > 0).astype(float)
    if W.nnz == 0:
        raise ValueError("epsilon too small: graph has no edges")
    return W


def row_normalize(W):
    """Return D^-1 W where D is the diagonal degree matrix."""
    m = W.sum(axis=1)
    m = np.where(m == 0, 1.0, m)
    return W.multiply(1.0 / m[:, None]).tocsr()


def laplacian_literal(X, W, c, epsilon):
    """Laplacian exactly as Eq. 11, applied to coordinate functions."""
    row_norm_W = row_normalize(W)
    return (c / epsilon**2) * (X - row_norm_W @ X)


def CDC(X, W, c, epsilon, d=2):
    """Gamma(f,g) = L(fg) - f*Lg - g*Lf; where f, g = x_i, x_j are coordinates function and L is laplacian_literal
    This version is fully vectorized, so it computes 1 sparse matvec for all coordinate/product terms
    combined (instead of one per i,j pair).
    """
    n, dim = X.shape

    def apply_L(u):
        return laplacian_literal(u, W, c, epsilon)

    L_X = apply_L(X)

    idx_pairs = [(i, j) for i in range(dim) for j in range(dim)]
    products = np.stack([X[:, i] * X[:, j] for i, j in idx_pairs], axis=1)
    L_products = apply_L(products)

    Gamma_raw = np.zeros((n, dim, dim))
    for k, (i, j) in enumerate(idx_pairs):
        Gamma_raw[:, i, j] = L_products[:, k] - X[:, i] * L_X[:, j] - X[:, j] * L_X[:, i]

    return Gamma_raw


def matrix_sqrt_psd(Gamma_raw):
    """Matrix square root of a stack of symmetric matrices, via scipy's
    sqrtm (batches over the leading dimension).

    Theoretic Gamma(f,g) = L(fg) - f*Lg - g*Lf with L beltrami operator == projector matrix,
    so it should have no negative eigenvalues.

    However our Gamma is the approximated one, and its eigenvalues are only guaranteed
    non-negative in the N->inf, eps->0 limit (from Corollary 4.2 of Bamberger et al.,
    "Riemannian Metric Matching" paper, whose finite-bandwidth Gamma_eps estimator
    is structurally analogous to this. The IMD paper's just tackles the continuum 
    (not finite-N) version of this projector property).
    
    This function simulates the real keeps only the real part of eigenvalues, 
    discarding whatever imaginary component results.
    This way on the simulation we remain on the manifold's domain in R^3.
    """

    Gamma_half = sqrtm(Gamma_raw)

    n_imaginary = np.sum(np.any(np.abs(Gamma_half.imag) > 1e-9, axis=(1, 2)))
    if n_imaginary > 0:
        print(f"matrix_sqrt_psd: {n_imaginary}/{len(Gamma_raw)} points produced "
              f"a complex result (estimator not PSD there) -- discarding imaginary part")

    return Gamma_half.real

def simulate_imd(X, tree, L_vecs, Gammas, n_steps, h, rng):
    """
    It simulates the IMD process with drift, CDC noise, and NO retraction step.

    """
    n_dim = X.shape[1]
    traj = np.zeros((n_steps + 1, n_dim))
    traj[0] = X[rng.integers(len(X))] # randomly picks one of the sampled data points to start from

    for l in range(n_steps):
        _, i = tree.query(traj[l]) # returns distances to the nearest neighbors, index of each neighbor
        xi = rng.normal(size=n_dim)
        traj[l+1] = traj[l] + h * L_vecs[i] + np.sqrt(h) * (Gammas[i] @ xi) #Euler Maruyama discretization

    return traj

 
 
def plot_result_sphere(X, traj, radius, out_path):
    """Same idea as plot_result, but rendered closer to the paper's
    Figure 6 style: a smooth shaded sphere surface instead of a raw
    point-cloud scatter, and the trajectory drawn as a connected line
    instead of scattered dots."""
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
 
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = radius * np.outer(np.cos(u), np.sin(v))
    ys = radius * np.outer(np.sin(u), np.sin(v))
    zs = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='0.85', alpha=0.5, linewidth=0, shade=True, zorder=0)
 
    ax.scatter(*X.T, color='0.5', s=1, alpha=0.15, linewidths=0, zorder=1)
    ax.plot(*traj.T, color='tab:blue', linewidth=0.9, alpha=0.9, zorder=2)
 
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.view_init(elev=20, azim=35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)

def plot_result_cube(X, traj, L, out_path):
    """Same idea as plot_result_sphere, but for the cube: draws the 6 flat
    shaded faces instead of a sphere surface, trajectory as a connected line."""
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    r = np.linspace(-L, L, 2)
    g1, g2 = np.meshgrid(r, r)

    for xval in [-L, L]:
        ax.plot_surface(np.full_like(g1, xval), g1, g2, color='0.85', alpha=0.35, linewidth=0, shade=True, zorder=0)
    for yval in [-L, L]:
        ax.plot_surface(g1, np.full_like(g1, yval), g2, color='0.85', alpha=0.35, linewidth=0, shade=True, zorder=0)
    for zval in [-L, L]:
        ax.plot_surface(g1, g2, np.full_like(g1, zval), color='0.85', alpha=0.35, linewidth=0, shade=True, zorder=0)

    ax.scatter(*X.T, color='0.5', s=1, alpha=0.15, linewidths=0, zorder=1)
    ax.plot(*traj.T, color='tab:blue', linewidth=0.9, alpha=0.9, zorder=2)

    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.view_init(elev=20, azim=35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
 
 
def sphere(N, R, h, n_traj_steps, c, rng):
    X = sample_sphere(N, R, rng)
    tree = KDTree(X)
    epsilon = compute_bandwidth(X, tree)
    W = build_adjacency(tree, epsilon)
 
    L_vecs = laplacian_literal(X, W, c, epsilon)
 
    Gamma_raw = CDC(X, W, c, epsilon)
    Gammas = matrix_sqrt_psd(Gamma_raw)
 
    traj = simulate_imd(X, tree, L_vecs, Gammas, n_traj_steps, h, rng)
    print(f"SPHERE: final |traj| deviation from R=1: {abs(np.linalg.norm(traj[-1]) - R):.2e}")
 
    plot_result_sphere(X, traj, R, f"outputs/path_simulation/imd_sphere{seed}.png")
    print("sphere saved")



def cube(N, L, h, n_traj_steps, c, rng):

    X = sample_cube(N, L, rng)
    tree = KDTree(X)
    epsilon = compute_bandwidth(X, tree)
    W = build_adjacency(tree, epsilon)

    L_vecs = laplacian_literal(X, W, c, epsilon)

    Gamma_raw = CDC(X, W, c, epsilon)
    Gammas = matrix_sqrt_psd(Gamma_raw)

    traj = simulate_imd(X, tree, L_vecs, Gammas, n_traj_steps, h, rng)
    print(f"CUBE: final cube-surface deviation (|max|coord| - L|): {abs(np.max(np.abs(traj[-1])) - L):.2e}")

    plot_result_cube(X, traj, L, f"outputs/path_simulation/imd_cube{seed}.png")
    print("cube saved")
 
 
if __name__ == "__main__":

    seed = 44
    rng = np.random.default_rng(seed)
    N = 6000
    h = 0.001
    n_traj_steps = 2000
    L = 1.0
    c = -1.0  # sign-flipped: matches centroid-seeking drift and correct CDC eigenvalue sign

    os.makedirs("outputs/path_simulation", exist_ok=True)

    cube(N, L, h, n_traj_steps, c, rng)
    sphere(N, L, h, n_traj_steps, c, rng) # R=L