# Least Squares with QR and SVD

Solution of overdetermined least-squares problems for full-rank and rank-deficient matrices.

## Implemented

- QR decomposition using Householder reflections;
- QR decomposition with column pivoting;
- application of `Q^T` without explicitly forming `Q`;
- singular value decomposition;
- construction and verification of the Moore–Penrose pseudoinverse;
- comparison of QR and minimum-norm SVD solutions;
- residual, rank and computational-complexity analysis.

## Run

```bash
python code/main.py
```

The program includes benchmark experiments for matrices up to `N = 1000`, so a complete run may take some time.

## Files

- `code/main.py` — original implementation;
- `report/report.pdf` — report with derivations and numerical results.

