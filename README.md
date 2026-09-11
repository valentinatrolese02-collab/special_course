Experiments from the paper Diffusion Processes on Implicit Manifolds.
We start by implementing spherical Brownian Motion.
We rely on ALgorithm 1: Implicit manifold-valued diffusions WITHOUT DRGD. In the original paper the DRGD step is obtained by training a noise-conditional score network sσ(x) described in appendix F3.


### Precomputation

For each point $x_i$ in the point cloud $X_N = \{x_i\}_{i=1}^N$:

1. Find its neighbors $\mathcal{N}(i)$ within an $\epsilon$-ball, via a proximity graph built with `scipy.spatial.cKDTree.query_pairs`.
2. Estimate the drift (a proxy for the random-walk graph Laplacian applied to coordinates, Eq. 11):

$$
L(x_i) := \frac{1}{|\mathcal{N}(i)|}\sum_{j \in \mathcal{N}(i)} (x_j - x_i)
$$

3. Estimate the tangent projector (standing in for the carré-du-champ operator, Proposition 1) via local PCA:

$$
\Sigma_i := \frac{1}{|\mathcal{N}(i)|}\sum_{j \in \mathcal{N}(i)} (x_j - x_i)(x_j - x_i)^\top, \qquad
\Gamma(x_i) := V_i V_i^\top
$$

where $V_i$ holds the top-$d$ eigenvectors of $\Sigma_i$ ($d$ = intrinsic manifold dimension). Since $\Gamma$ is an orthogonal projector, $\Gamma^{1/2} = \Gamma$ exactly.

### Simulation

**Input:** point cloud $X_N$, step size $h > 0$, iterations $L$, retraction $\mathcal{R}(\cdot)$

$$
\bar X_0 \leftarrow \text{random point in } X_N
$$

for $\ell = 0, \dots, L-1$:

$$
i \leftarrow \text{nearest-neighbor index of } \bar X_\ell \text{ in } X_N
$$

$$
\xi_{\ell+1} \sim \mathcal{N}(0, I_3)
$$

$$
U_{\ell+1} = \bar X_\ell + h\, L(x_i) + \sqrt{h}\, \Gamma(x_i)\, \xi_{\ell+1} \qquad \text{(E-M step)}
$$

$$
\bar X_{\ell+1} = \mathcal{R}(U_{\ell+1}) \qquad \text{(analytic retraction — not DRGD)}
$$

**Output:** trajectory $\{\bar X_\ell\}_{\ell=0}^L$

- $L(x_i)$ is a simplified neighbor-centroid proxy rather than the paper's precisely scaled random-walk graph Laplacian ($c/\epsilon^2$ normalization from Eq. 11); it reproduces the qualitative centripetal pull but hasn't been checked against their exact constant or sign convention.