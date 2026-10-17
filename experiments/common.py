"""Method registry, ablation configurations and input/output helpers shared by the experiment scripts."""

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ambit import AMBIT, MMA, OC, AdaptiveMoment, SiMPL, TopOptModel  # noqa: E402


# Methods whose results depend on the random seed (Module II subset sampling).
SEEDED = {"AMBIT", "AMBIT-R"}

METHODS = {
    "OC": lambda seed=0: OC(),
    "MMA": lambda seed=0: MMA(),
    "SiMPL": lambda seed=0: SiMPL(),
    "Adam": lambda seed=0: AdaptiveMoment("adam"),
    "AMSGrad": lambda seed=0: AdaptiveMoment("amsgrad"),
    "AdaBelief": lambda seed=0: AdaptiveMoment("adabelief"),
    "Lion": lambda seed=0: AdaptiveMoment("lion"),
    "AMBIT": lambda seed=0: AMBIT(seed=seed),
    "AMBIT-R": lambda seed=0: AMBIT(seed=seed, fallback=True),
}

DEFAULT_METHODS = ["OC", "MMA", "SiMPL", "Adam", "AMSGrad", "AdaBelief", "Lion", "AMBIT"]

# Module ablation of Figure 7. Module IV off means global bias correction, moments
# updated at every iteration, fixed beta2 and the capped schedule eta0/sqrt(t).
_M4_OFF = dict(active_set=False, step_schedule="cap", adaptive_beta2=False)

ABLATIONS = {
    "Full AMBIT": dict(),
    "Full AMBIT, kappa0=0": dict(kappa0=0.0),
    "w/o median rescaling": dict(median_rescale=False),
    "w/o Module I": dict(mirror=False),
    "w/o Module II": dict(metric="belief"),
    "w/o Module III": dict(filter_moments=False),
    "w/o Module IV": dict(_M4_OFF),
    "Module I only": dict(metric="identity", momentum=False, step_schedule="constant",
                          active_set=False, adaptive_beta2=False),
    "Module I only, kappa0=0": dict(metric="identity", momentum=False, step_schedule="constant",
                                    active_set=False, adaptive_beta2=False, kappa0=0.0),
    "Modules I + II": dict(filter_moments=False, **_M4_OFF),
}


def make_ablation(label, seed=0):
    if label == "SiMPL":
        return SiMPL()
    return AMBIT(seed=seed, name=label, **ABLATIONS[label])


def make_model(problem, rmin_elements=None, volume="physical"):
    """Model with r_min = 0.024 * domain height unless an element count is given."""
    rmin = rmin_elements if rmin_elements is not None else 0.024 * problem.nely
    return TopOptModel(problem, rmin=rmin, volume=volume)


def seeds_for(name, seeds):
    return list(seeds) if name in SEEDED or name.startswith(("Full", "w/o", "Modules")) else [seeds[0]]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def write_csv(path, rows):
    if not rows:
        return
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, obj):
    ensure_dir(os.path.dirname(os.path.abspath(path)))

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)


def save_design(path, problem, model, result):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    rho = np.zeros(problem.nelx * problem.nely)
    rho[model.idx] = result.rho_bar
    np.savez_compressed(path, rho_bar=rho, nelx=problem.nelx, nely=problem.nely,
                        compliance=result.compliance)


def mean_over_seeds(results, key):
    vals = [getattr(r, key) for r in results]
    return float(np.mean(vals))
