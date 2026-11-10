# Nonlinear Heat Equation

Numerical solution of a nonlinear heat equation using a finite-difference method and the method of lines.

## Implemented

- implicit Crank–Nicolson scheme with `sigma = 0.5`;
- fixed-point iterations for the nonlinear discrete problem;
- method of lines for spatial discretization;
- fourth-order Runge–Kutta method for time integration;
- comparison of the two numerical approaches;
- error and convergence experiments.

## Run

```bash
python code/main.py
```

The program uses an interactive menu for selecting an experiment.

## Files

- `code/main.py` — original implementation;
- `report/report.pdf` — report with derivations and experimental results.

