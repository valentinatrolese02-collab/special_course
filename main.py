"""
Reproduce the paper's "point cloud + Brownian motion overlay" figure style,
on a sphere, using plain numpy/scipy/matplotlib.

Construction:
  - point cloud = manifold samples (no equation of the sphere is used except
    for the final retraction, which stands in for a DRGD-style correction)
  - W          = explicit sparse adjacency matrix (Eq. 11's hard cutoff),
                 built once from an eps-ball proximity graph (here computed as sparse array)
  - D          = diagonal degree matrix for each node of the graph
  - L(X)       = random-walk graph Laplacian applied to coordinates (Eq. 11)
  - Gamma_raw  = carre-du-champ (Eq. 7) applied to ambient coordinate
                 functions -- the literal, uncalibrated finite-sample
                 estimator, NOT reprojected into a clean {0,1}-eigenvalue
                 projector. Its eigenvalues are only guaranteed to be
                 non-negative in the N->inf, eps->0 limit (Corollary 4.2).
  - Gammas     = Gamma_raw^(1/2), the actual matrix square root used by
                 Algorithm 1's noise term -- NOT Gamma_raw itself.
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

    If Gamma_raw isn't positive semi-definite at some point -- a real
    possibility at finite N, since the theory only guarantees PSD in the
    N->inf, eps->0 limit (Corollary 4.2) -- sqrtm returns a complex value
    there (not a rotation: Gamma_raw is symmetric, so its eigenvalues are
    always real; the complex part comes purely from taking sqrt() of a
    negative real eigenvalue). We detect that, report it, and discard the
    imaginary part -- exactly equivalent to clipping that eigenvalue to 0
    before the square root, i.e. injecting zero noise along whichever
    direction the estimator's PSD guarantee failed at."""
    Gamma_half = sqrtm(Gamma_raw)

    n_imaginary = np.sum(np.any(np.abs(Gamma_half.imag) > 1e-9, axis=(1, 2)))
    if n_imaginary > 0:
        print(f"matrix_sqrt_psd: {n_imaginary}/{len(Gamma_raw)} points produced "
              f"a complex result (estimator not PSD there) -- discarding imaginary part")

    return Gamma_half.real


def simulate_trajectory(X, tree, L_vecs, Gammas, n_steps, h, radius, rng):
    n_dim = X.shape[1]
    traj = np.zeros((n_steps + 1, n_dim))
    traj[0] = X[rng.integers(len(X))]
    for l in range(n_steps):
        _, i = tree.query(traj[l])
        xi = rng.normal(size=n_dim)
        step = h * L_vecs[i] + np.sqrt(h) * (Gammas[i] @ xi)
        new_p = traj[l] + step
        traj[l + 1] = new_p / np.linalg.norm(new_p) * radius
    return traj


def plot_result(X, traj, out_path):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*X.T, color='0.6', s=2, alpha=0.5, linewidths=0)
    ax.scatter(*traj.T, color='red', s=3, alpha=0.9, linewidths=0)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.view_init(elev=20, azim=35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    rng = np.random.default_rng(0)
    N = 6000
    h = 0.1
    n_traj_steps = 40000
    R = 1.0
    c = -1.0  # sign-flipped: matches centroid-seeking drift and correct CDC eigenvalue sign

    t0 = time.time()
    X = sample_sphere(N, R, rng)
    tree = KDTree(X)
    epsilon = compute_bandwidth(X, tree)
    W = build_adjacency(tree, epsilon)
    print(f"graph built: {time.time()-t0:.1f}s, nnz={W.nnz}")

    t0 = time.time()
    L_vecs = laplacian_literal(X, W, c, epsilon)
    print(f"L computed: {time.time()-t0:.1f}s")

    t0 = time.time()
    Gamma_raw = CDC(X, W, c, epsilon)
    Gammas = matrix_sqrt_psd(Gamma_raw)  # THE FIX: actual Gamma^(1/2), not Gamma itself
    print(f"Gammas built: {time.time()-t0:.1f}s")

    t0 = time.time()
    traj = simulate_trajectory(X, tree, L_vecs, Gammas, n_traj_steps, h, R, rng)
    print(f"trajectory simulated: {time.time()-t0:.1f}s")
    print(f"final |traj| deviation from R=1: {abs(np.linalg.norm(traj[-1]) - R):.2e}")

    os.makedirs("outputs", exist_ok=True)
    plot_result(X, traj, "outputs/imd_sphere_brownian_epsball.png")
    print("saved")


if __name__ == "__main__":
    main()