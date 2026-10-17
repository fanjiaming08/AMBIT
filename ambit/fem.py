"""Two-dimensional linear elasticity on a structured grid of bilinear quadrilaterals.

Element and node numbering follow the 88-line code of Andreassen et al. (2011):
elements and nodes are numbered column-wise, with the y index running downward.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:  # optional, used when available
    from sksparse.cholmod import cholesky as _cholmod_cholesky
except ImportError:  # pragma: no cover
    _cholmod_cholesky = None


def q4_stiffness(nu=0.3):
    """Stiffness matrix of a unit square, unit-modulus, unit-thickness plane-stress element."""
    k = np.array([
        1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8,
        -1 / 4 + nu / 12, -1 / 8 - nu / 8, nu / 6, 1 / 8 - 3 * nu / 8,
    ])
    idx = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 0, 7, 6, 5, 4, 3, 2],
        [2, 7, 0, 5, 6, 3, 4, 1],
        [3, 6, 5, 0, 7, 2, 1, 4],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [5, 4, 3, 2, 1, 0, 7, 6],
        [6, 3, 4, 1, 2, 7, 0, 5],
        [7, 2, 1, 4, 3, 6, 5, 0],
    ])
    return k[idx] / (1.0 - nu ** 2)


class Mesh2D:
    """Structured nelx-by-nely grid of unit square elements."""

    def __init__(self, nelx, nely):
        self.nelx = int(nelx)
        self.nely = int(nely)
        self.n_elem = self.nelx * self.nely
        self.n_node = (self.nelx + 1) * (self.nely + 1)
        self.n_dof = 2 * self.n_node

        elx, ely = np.meshgrid(np.arange(self.nelx), np.arange(self.nely), indexing="ij")
        elx = elx.ravel()
        ely = ely.ravel()
        n1 = (self.nely + 1) * elx + ely
        n2 = (self.nely + 1) * (elx + 1) + ely
        self.edof = np.stack(
            [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3,
             2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1], axis=1
        ).astype(np.int64)
        self.centroids = np.stack([elx + 0.5, ely + 0.5], axis=1).astype(float)

    def node(self, ix, iy):
        """Node index of grid point (ix, iy), with iy counted from the top edge."""
        return (self.nely + 1) * np.asarray(ix) + np.asarray(iy)

    def to_grid(self, field):
        """Reshape an element field to an array indexed as [ix, iy]."""
        return np.asarray(field).reshape(self.nelx, self.nely)


class FESolver:
    """Assembles and solves K(E) U = F for element moduli E."""

    def __init__(self, mesh, fixed_dofs, nu=0.3):
        self.mesh = mesh
        self.KE = q4_stiffness(nu)
        self.fixed = np.unique(np.asarray(fixed_dofs, dtype=np.int64))
        self.free = np.setdiff1d(np.arange(mesh.n_dof), self.fixed)
        edof = mesh.edof
        self.iK = np.repeat(edof, 8, axis=1).ravel()
        self.jK = np.tile(edof, (1, 8)).ravel()
        self._KE_flat = self.KE.ravel()

    def assemble(self, E):
        sK = (np.asarray(E)[:, None] * self._KE_flat[None, :]).ravel()
        n = self.mesh.n_dof
        return sp.coo_matrix((sK, (self.iK, self.jK)), shape=(n, n)).tocsc()

    def solve(self, E, F):
        """Return displacement vector U and the inverse stiffness diagonal (zero on fixed dofs)."""
        K = self.assemble(E)
        Kff = K[self.free][:, self.free]
        Ff = np.asarray(F)[self.free]
        if _cholmod_cholesky is not None:
            Uf = _cholmod_cholesky(Kff)(Ff)
        else:
            Uf = spla.spsolve(Kff, Ff, permc_spec="MMD_AT_PLUS_A")
        U = np.zeros(self.mesh.n_dof)
        U[self.free] = Uf
        inv_diag = np.zeros(self.mesh.n_dof)
        inv_diag[self.free] = 1.0 / K.diagonal()[self.free]
        return U, inv_diag
