# Finite-Difference Method for the Wave Equation

Numerical solution of an initial-boundary value problem for the one-dimensional wave equation with mixed boundary conditions.

## Implemented

- implicit three-level finite-difference scheme with weight `sigma = 0.5`;
- Thomas algorithm for tridiagonal linear systems;
- first- and second-order approximations of the Neumann boundary condition;
- verification using five analytical solutions;
- error tables and grid-convergence experiments;
- comparison plots for analytical and numerical solutions.

## Run

```bash
python code/main.py
```

The script prints error tables and opens the generated plots.

## Files

- `code/main.py` — original implementation;
- `report/report.pdf` — report with the mathematical formulation and results;
- `report/latex-source.zip` — LaTeX source of the report.

