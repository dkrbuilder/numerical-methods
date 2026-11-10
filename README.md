# Numerical Methods

## Projects

| Project | Methods | Technologies |
|---|---|---|
| [Wave equation](01-wave-equation/) | Implicit finite-difference scheme, mixed boundary conditions, Thomas algorithm, convergence analysis | Python, NumPy, Matplotlib |
| [Nonlinear heat equation](02-nonlinear-heat-equation/) | Crank–Nicolson scheme, fixed-point iterations, method of lines, RK4 | Python, NumPy, Matplotlib |
| [QR decomposition with Givens rotations](03-givens-qr/) | Dense and banded matrices, compact band storage, back substitution, complexity benchmarks | Python, NumPy |
| [Least squares with QR and SVD](04-householder-qr-svd/) | Householder reflections, column pivoting, SVD, pseudoinverse, rank-deficient systems | Python, NumPy |

Each project contains the original source code and the corresponding university report. The reports are written in Russian.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run a project from its directory, for example:

```bash
python 03-givens-qr/code/main.py
```

## Skills demonstrated

- numerical solution of partial differential equations;
- matrix decompositions and least-squares problems;
- dense, banded and rank-deficient matrix algorithms;
- error, residual and convergence analysis;
- computational-complexity experiments;
- scientific computing and visualization in Python.

