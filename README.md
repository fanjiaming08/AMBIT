# AMBIT

Reference implementation of AMBIT, a design update for density-based topology optimization that
places a curvature-informed adaptive diagonal metric inside a Fermi–Dirac Bregman mirror step.
The code accompanies the article *An Improved Gradient Method With Curvature-Informed Adaptive
Moment Estimation for Topology Optimization of Complex Mechanical Structures* (J. Fan).

The implementation covers two-dimensional minimum-compliance problems with a density filter,
smoothed Heaviside projection and SIMP interpolation, and contains:

- **AMBIT**, with its four modules as separate, switchable components:
  - Module I, the constraint-intrinsic polarized mirror step (Eqs. 15–22), with a single monotone bisection on the resource constraint;
  - Module II, the fusion of a belief-centered second moment with the analytical diagonal compliance curvature and its Jacobi coupling term (Eqs. 24–31);
  - Module III, filter-consistent smoothing and a monotone envelope of the metric (Eqs. 32–33);
  - Module IV, active-set-aware bias correction and feedback schedules (Eqs. 35–38);
  - the secant reliability index and identity-metric fallback of Eq. (41), as the variant AMBIT-R.
- **Baselines**: optimality criteria (OC), the method of moving asymptotes (MMA), SiMPL, Adam,
  AMSGrad, AdaBelief and Lion. All iterative methods share the filter, the projection and the
  continuation of the penalization and projection parameters.
- **Diagnostics** for the mismatches of Section 3.2: the moment–constraint inconsistency index,
  the preconditioner correlation length, the cross-element dispersion of the normalized step,
  the fraction of negative composite curvature, the equivalent OC damping of Eq. (23), and the
  per-stage coverage statistics of Theorem 2 and Corollary 1.
- **Experiment scripts** for the classical benchmarks, fixed-budget statistical comparison,
  module ablation, hyperparameter sensitivity and mesh refinement, with figure generation.

## Scope

The following parts of the study are not included in this repository: the three-dimensional
scalability study with a matrix-free multigrid-preconditioned solver, the SimJEB jet engine
brackets (Dataset B), the learning-based baselines (TOuNN, TopoDiff, NITO, IF-TONIR) and the
warm-start experiment that depends on them, and a reader for the TopoDiff benchmark files.
The fixed-budget experiments run on generated 64 × 64 instances by default and accept any set
of instances in the `.npz` format described below.

## Installation

Python 3.9 or later.

```bash
git clone https://github.com/<account>/AMBIT.git
cd AMBIT
pip install -e .
```

The only dependencies are NumPy, SciPy and Matplotlib. If `scikit-sparse` is installed, CHOLMOD
is used for the linear solves; otherwise SciPy's sparse direct solver is used.

## Quick start

```python
from ambit import AMBIT, MMA, OC, TopOptModel, mbb

problem = mbb(150, 50, volfrac=0.5)
model = TopOptModel(problem, rmin=2.4)

for method in (OC(), MMA(), AMBIT(seed=1)):
    result = method.run(model, max_iter=500)
    print(f"{result.name:6s} compliance={result.compliance:.3f} "
          f"iterations={result.iterations} non-discreteness={result.mnd:.2f}%")
```

`result.rho_bar` holds the projected densities on the design elements, and `result.hist` the
per-iteration history (compliance, `p`, `beta`, step size, multiplier, fusion weight and so on).
Pass `record_fields=True` to an optimizer to keep per-iteration fields such as the metric.

## Reproducing the experiments

All scripts are run from the repository root and write CSV or JSON files under `results/`.
Every script accepts `--help`.

| Script | Content |
| --- | --- |
| `experiments/run_diagnostics.py` | Diagnostic measurements on the MBB beam, Table 5 panels A–F |
| `experiments/run_benchmarks.py` | Converged compliance on the MBB beam, cantilever, L-bracket and Michell truss, Table 6 and Figure 2 |
| `experiments/run_dataset.py` | Fixed-budget comparison on a set of instances, Table 7 |
| `experiments/analyze_dataset.py` | Summary table, Friedman and Conover tests, bootstrap intervals, Cliff's delta, family-level inference, Tables 7–8 |
| `experiments/run_ablation.py` | Module ablation, Figure 7 |
| `experiments/run_sensitivity.py` | One-at-a-time hyperparameter variation, Table 9 |
| `experiments/run_mesh_study.py` | Refinement study with the filter radius fixed in physical units, Figure 8 |
| `experiments/make_figures.py` | Figures from the result files |

Typical full runs:

```bash
python experiments/run_diagnostics.py
python experiments/run_benchmarks.py
python experiments/run_dataset.py --count 1800 --seed 1 --budget 60 --out results/dataset
python experiments/analyze_dataset.py results/dataset/runs.csv
python experiments/run_ablation.py --reference-dir results/dataset/references
python experiments/run_sensitivity.py --reference-dir results/dataset/references
python experiments/run_mesh_study.py
python experiments/make_figures.py
```

Full-resolution runs take from minutes (one benchmark) to many hours (large instance sets).
For a quick check, reduce the problem size, for example
`python experiments/run_benchmarks.py --scale 0.5 --methods OC MMA AMBIT --seeds 1`.
Note that the filter radius is fixed in physical units (2.4 % of the domain height), so it
covers fewer elements on reduced meshes; `--rmin` sets it in elements instead.

Results of methods that contain Module II depend on the random subset used for the rank
correlation of Eq. (31). These methods are run for the seeds given by `--seeds` (default 1 to 5)
and reported as means over seeds. All other methods are deterministic.

## Implementation notes

**Resource constraint.** `TopOptModel(volume="physical")` (default) imposes the volume constraint
on the projected field ρ̄; `volume="raw"` imposes it on the raw design field ρ as in Eq. (3).
Both keep the multiplier function of Eq. (22) monotone. With a threshold η_H = 0.5 and the raw-field
constraint, intermediate filtered densities slightly above the threshold are projected to solid
material, so the realized volume fraction of ρ̄ can differ markedly from the prescribed one;
the volume fraction error reported by the scripts is always computed on ρ̄.

**Continuation.** The penalization advances by Δp when the trailing relative compliance change is
below θ_c and the stage is older than T_hold iterations, Eq. (38). For the projection sharpness,
`Continuation(beta_rule="settled")` (default) doubles β under the same settling test, while the
non-discreteness exceeds M_tol or once p has reached p_max; `beta_rule="every_iteration"` doubles β
at every iteration at which the non-discreteness exceeds M_tol. Termination requires p = p_max,
β = β_max, a relative compliance change below θ_stop and non-discreteness below M_stop.

**Step size.** The step size η₀ refers to the logit coordinate and to the sensitivity
normalization g ← g / mean|g₁|. The product η·λ·v of Eq. (23) is logged at every iteration and
reported per stage by `run_diagnostics.py`.

**Metric bias correction.** `AMBIT(metric_bias=True)` (default) divides the enveloped metric by the
active-set product of Eq. (36); `metric_bias=False` uses the enveloped metric directly.

**Final metrics.** Every run, including runs stopped by an analysis budget, is evaluated at
p = p_max and β = β_max. A run is counted as failed when there is no path of elements with
ρ̄ > 0.5, connected through shared edges, between every loaded node and at least one supported node.

**Reference solutions.** For the fixed-budget experiments, each instance has a reference
compliance computed once, independently of the compared methods, as the lower connected result of
long MMA and OC runs from the uniform design. References are cached under `<out>/references/`.

**Filters.** The density filter and the moment filter of Module III are row-normalized cone
filters assembled as sparse matrices.

## Instance format

`ambit.problems.save_instance` and `load_instance` read and write instances as `.npz` files with
the fields `name`, `nelx`, `nely`, `volfrac`, `F` (load vector of length 2(nelx+1)(nely+1)),
`fixed` (constrained degrees of freedom) and `passive` (boolean mask of void elements).
Nodes and elements are numbered column-wise with the y index running downward, and degree of
freedom 2k (2k+1) is the x (y) displacement of node k, as in the 88-line code of Andreassen et al.

## Repository layout

```
ambit/
  fem.py            bilinear elements, assembly and solution
  problems.py       benchmarks, generated instances, instance input/output
  filters.py        cone filter and smoothed Heaviside projection
  model.py          compliance, sensitivities and curvature terms on the design field
  continuation.py   feedback continuation and non-discreteness measure
  optimizers/       OC, MMA, SiMPL, adaptive moment baselines, AMBIT
  diagnostics.py    correlation length and stage statistics
  metrics.py        compliance error and connectivity test
  stats.py          Friedman, Conover, Holm, bootstrap and Cliff's delta
experiments/        experiment and figure scripts
tests/              unit tests
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests check the element stiffness, the sensitivities and the volume gradient against finite
differences, the curvature terms against a finite-difference second derivative, the feasibility
and fixed points of the mirror step, the continuation rules, and that every method runs and
returns a feasible design.

## License

MIT, see `LICENSE`.
