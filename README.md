Experiments from the paper Diffusion Processes on Implicit Manifolds.
We start by implementing spherical Brownian Motion.
We rely on ALgorithm 1: Implicit manifold-valued diffusions WITHOUT DRGD. In the original paper the DRGD step is obtained by training a noise-conditional score network sσ(x) described in appendix F3.


### Operators L, $\Gamma$

The paper originally defines the RWGL on a graph over the point cloud $X_N$ (equation (11)) as:

$$
(\mathsf{L}_N u)(x_i) := \frac{c}{\epsilon^2} \sum_{i \sim j} \frac{W_{ij}}{m_i} \left( u(x_i) - u(x_j) \right)
$$

and the carré-du-champ (CDC) operator (equation (7)), defined for every generator L:

$$
\Gamma(f, g) := L(fg) - fLg - gLf
$$

My first attempt consist in implementing these operators following precisely their definition.

To compute the graph, I utilize the class cKDTree from scipy.spatial (identical to KDTree) which implements the nearest neighbor searching. 


**Input:** point cloud $X_N$, step size $h > 0$, iterations $L$

Initialize on $X_N$

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
X_{\ell+1} = \bar X_\ell + h (L(X_\ell) + b(X_\ell)) + \sqrt{h}\, \Gamma(x_\ell)^{1/2} \, \xi_{\ell+1} \qquad \text{(E-M step, eq (15))}
$$

**Output:** trajectory $\{\bar X_\ell\}_{\ell=0}^L$