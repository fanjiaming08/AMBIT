"""Benchmark problems: MBB beam, cantilever, L-bracket, Michell truss, and generic instances."""

from dataclasses import dataclass, field

import numpy as np

from .fem import Mesh2D


@dataclass
class Problem:
    name: str
    nelx: int
    nely: int
    volfrac: float
    F: np.ndarray
    fixed: np.ndarray
    passive: np.ndarray = None  # boolean mask of elements fixed to void
    meta: dict = field(default_factory=dict)

    @property
    def mesh(self):
        return Mesh2D(self.nelx, self.nely)

    @property
    def design(self):
        n = self.nelx * self.nely
        if self.passive is None:
            return np.ones(n, dtype=bool)
        return ~self.passive


def _scaled(n, scale):
    return max(4, int(round(n * scale)))


def mbb(nelx=300, nely=100, volfrac=0.5):
    """Half MBB beam: symmetry on the left edge, roller at the bottom right, load at the top left."""
    m = Mesh2D(nelx, nely)
    F = np.zeros(m.n_dof)
    F[2 * m.node(0, 0) + 1] = -1.0
    left = m.node(0, np.arange(nely + 1))
    fixed = np.concatenate([2 * left, [2 * m.node(nelx, nely) + 1]])
    return Problem("mbb", nelx, nely, volfrac, F, fixed)


def cantilever(nelx=200, nely=100, volfrac=0.4):
    """Cantilever clamped on the left edge with a vertical point load at mid-height of the free end."""
    m = Mesh2D(nelx, nely)
    F = np.zeros(m.n_dof)
    F[2 * m.node(nelx, nely // 2) + 1] = -1.0
    left = m.node(0, np.arange(nely + 1))
    fixed = np.concatenate([2 * left, 2 * left + 1])
    return Problem("cantilever", nelx, nely, volfrac, F, fixed)


def l_bracket(nelx=150, nely=150, volfrac=0.3, arm=0.4):
    """L-shaped domain clamped at the top of the vertical arm, loaded vertically at the free arm."""
    m = Mesh2D(nelx, nely)
    cx = m.centroids[:, 0] / nelx
    cy = m.centroids[:, 1] / nely
    passive = (cx > arm) & (cy < 1.0 - arm)
    F = np.zeros(m.n_dof)
    iy_load = int(round(nely * (1.0 - arm / 2)))
    F[2 * m.node(nelx, iy_load) + 1] = -1.0
    top = m.node(np.arange(int(round(arm * nelx)) + 1), 0)
    fixed = np.concatenate([2 * top, 2 * top + 1])
    return Problem("l_bracket", nelx, nely, volfrac, F, fixed, passive=passive)


def michell(nelx=240, nely=120, volfrac=0.35):
    """Michell-type truss: central load on the bottom edge, pin and roller at the bottom corners."""
    m = Mesh2D(nelx, nely)
    F = np.zeros(m.n_dof)
    F[2 * m.node(nelx // 2, nely) + 1] = -1.0
    left = m.node(0, nely)
    right = m.node(nelx, nely)
    fixed = np.array([2 * left, 2 * left + 1, 2 * right + 1])
    return Problem("michell", nelx, nely, volfrac, F, fixed)


BENCHMARKS = {
    "mbb": (mbb, (300, 100)),
    "cantilever": (cantilever, (200, 100)),
    "l_bracket": (l_bracket, (150, 150)),
    "michell": (michell, (240, 120)),
}


def benchmark(name, scale=1.0):
    """Benchmark of the given name at the default resolution multiplied by scale."""
    fn, (nx, ny) = BENCHMARKS[name]
    return fn(_scaled(nx, scale), _scaled(ny, scale))


# Boundary-condition families for generic square instances. Each entry returns the
# list of (node, direction) pairs to fix, direction 0 for x and 1 for y.
def _bc_families(n):
    edge = np.arange(n + 1)
    return {
        "left_clamped": [(("x", 0, iy), d) for iy in edge for d in (0, 1)],
        "right_clamped": [(("x", n, iy), d) for iy in edge for d in (0, 1)],
        "top_clamped": [(("x", ix, 0), d) for ix in edge for d in (0, 1)],
        "bottom_clamped": [(("x", ix, n), d) for ix in edge for d in (0, 1)],
        "bottom_corners": [(("x", 0, n), 0), (("x", 0, n), 1), (("x", n, n), 1)],
        "left_right_pins": [(("x", 0, n // 2), 0), (("x", 0, n // 2), 1),
                            (("x", n, n // 2), 0), (("x", n, n // 2), 1)],
        "left_half_clamped": [(("x", 0, iy), d) for iy in edge[: n // 2 + 1] for d in (0, 1)],
        "corner_clamps": [(("x", 0, 0), 0), (("x", 0, 0), 1), (("x", 0, n), 0), (("x", 0, n), 1)],
    }


def generic_instance(n, volfrac, bc_family, load_node, load_angle, name="instance"):
    """Square n-by-n instance with a named support family and one point load.

    load_node is a grid point (ix, iy) and load_angle is in radians.
    """
    m = Mesh2D(n, n)
    fam = _bc_families(n)[bc_family]
    fixed = np.array([2 * m.node(ix, iy) + d for ((_, ix, iy), d) in fam], dtype=np.int64)
    F = np.zeros(m.n_dof)
    nd = m.node(*load_node)
    F[2 * nd] = np.cos(load_angle)
    F[2 * nd + 1] = np.sin(load_angle)
    meta = {"bc_family": bc_family, "load_node": tuple(int(v) for v in load_node),
            "load_angle": float(load_angle)}
    return Problem(name, n, n, float(volfrac), F, fixed, meta=meta)


def random_instances(count, n=64, seed=0, families=None, vf_range=(0.30, 0.50), vf_step=0.02):
    """Draw square instances with volume fractions on a grid and random boundary loads.

    The load is applied at a random boundary node that is not fixed.
    """
    rng = np.random.default_rng(seed)
    all_fam = sorted(_bc_families(n).keys())
    families = all_fam if families is None else list(families)
    vfs = np.round(np.arange(vf_range[0], vf_range[1] + 1e-9, vf_step), 4)
    boundary = [(ix, 0) for ix in range(n + 1)] + [(ix, n) for ix in range(n + 1)] + \
               [(0, iy) for iy in range(1, n)] + [(n, iy) for iy in range(1, n)]
    out = []
    for k in range(count):
        fam = families[rng.integers(len(families))]
        fixed_nodes = {(ix, iy) for ((_, ix, iy), _) in _bc_families(n)[fam]}
        cand = [b for b in boundary if b not in fixed_nodes]
        load = cand[rng.integers(len(cand))]
        angle = rng.uniform(0.0, 2.0 * np.pi)
        vf = float(vfs[rng.integers(len(vfs))])
        out.append(generic_instance(n, vf, fam, load, angle, name=f"inst{k:05d}"))
    return out


def save_instance(problem, path):
    np.savez_compressed(path, name=problem.name, nelx=problem.nelx, nely=problem.nely,
                        volfrac=problem.volfrac, F=problem.F, fixed=problem.fixed,
                        passive=(problem.passive if problem.passive is not None
                                 else np.zeros(problem.nelx * problem.nely, dtype=bool)))


def load_instance(path):
    d = np.load(path, allow_pickle=False)
    passive = d["passive"].astype(bool)
    return Problem(str(d["name"]), int(d["nelx"]), int(d["nely"]), float(d["volfrac"]),
                   d["F"], d["fixed"], passive=passive if passive.any() else None)
