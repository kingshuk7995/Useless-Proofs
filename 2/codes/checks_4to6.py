"""CHECK 4-6: Theorem 6 window, larger sample, Theorem 5 cap."""
import numpy as np, itertools
from common import *

print("="*78)
print("CHECK 4 - Theorem 6: is there ANY (dataset, c) where (C2) actually holds?")
print("  family: x_+ = (R, 1+delta) y=+1 ; x_- = (R, 1-delta) y=-1, NO bias in H1.")
print("  closed form predicts a window iff R < 1, namely sqrt(u)(R^2+1) > (u+1)R,")
print("  u = A^2 + R^2, A = c*sqrt((K-1)d).")
print("="*78)
delta = 0.05
for R in [0.5, 1.0, 2.0]:
    X = np.array([[R, 1+delta], [R, 1-delta]]); y = np.array([1.,-1.])
    g1, R1 = margin(X, y, bias=False), rad_linear(X)
    gb = margin(X, y, bias=True)
    base = R1/g1
    lo, hi = (np.sqrt((R*R+1)**2 - 4*R*R*1.0), None)
    # predicted window endpoints from  t^2 - ((R^2+1)/R) t + 1 = 0, t=sqrt(u)
    bq = (R*R+1)/R; disc = bq*bq - 4
    if disc > 0:
        t1, t2 = (bq-np.sqrt(disc))/2, (bq+np.sqrt(disc))/2
        pred = f"A in (0, {np.sqrt(max(t2**2-R*R,0)):.4f})" if t2**2 > R*R else "empty"
    else:
        pred = "empty"
    print(f"\n R={R}: gamma1(no bias)={g1:.6f}  gamma1(with bias)={gb:.6f}  "
          f"R_S={R1:.6f}  baseline ratio={base:.6f}")
    print(f"   closed-form prediction (delta->0): {pred}")
    best = (None, 1e9)
    for A in [0.0,0.1,0.25,0.5,0.75,1.0,1.146,1.5,1.9,2.0,2.5,4.0,8.0]:
        Xt = np.hstack([X, np.full((2,1), A)])
        g2, R2 = margin(Xt,y,bias=False), rad_linear(Xt)
        r = R2/g2
        if r < best[1]: best = (A, r)
        tag = "  <-- (C2) HOLDS" if r < base-1e-9 else ""
        print(f"     A={A:<6} gamma2={g2:.6f} (x{g2/g1:.4f})  R~={R2:.6f} (x{R2/R1:.4f})"
              f"  ratio={r:.6f}{tag}")
    print(f"   best A={best[0]}, ratio={best[1]:.6f} vs baseline {base:.6f} "
          f"-> improvement {100*(1-best[1]/base):.2f}%")

print()
print("="*78)
print("CHECK 5 - same phenomenon with a larger, jittered sample (not an n=2 artifact)")
print("="*78)
rng = np.random.default_rng(7)
R, delta, nh = 0.5, 0.30, 6
Xp = np.column_stack([R+0.05*rng.normal(size=nh), 1+delta+0.05*rng.normal(size=nh)])
Xm = np.column_stack([R+0.05*rng.normal(size=nh), 1-delta+0.05*rng.normal(size=nh)])
X = np.vstack([Xp,Xm]); y = np.concatenate([np.ones(nh), -np.ones(nh)])
g1, R1 = margin(X,y,bias=False), rad_linear(X); base = R1/g1
print(f"  n={2*nh}  gamma1(no bias)={g1:.6f}  gamma1(bias)={margin(X,y,bias=True):.6f}  "
      f"R_S={R1:.6f}  baseline={base:.6f}")
for A in [0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,3.0,6.0]:
    Xt = np.hstack([X, np.full((2*nh,1), A)])
    g2, R2 = margin(Xt,y,bias=False), rad_linear(Xt); r = R2/g2
    print(f"    A={A:<5} gamma2={g2:.6f} (x{g2/g1:.4f})  R~={R2:.6f} (x{R2/R1:.4f})  "
          f"ratio={r:.6f}{'  <-- (C2) HOLDS' if r < base-1e-9 else ''}")

print()
print("="*78)
print("CHECK 6 - Theorem 5: gamma_2 is capped independently of c  (gamma_2 <= (B/2)*D)")
print("="*78)
D = min(np.linalg.norm(a-b) for a in Xp for b in Xm)
print(f"  D = min_{{y_i != y_j}} ||x_i - x_j|| = {D:.6f},  cap (B/2)D = {D/2:.6f}")
for A in [0.0,1.0,10.0,100.0,1000.0]:
    Xt = np.hstack([X, np.full((2*nh,1), A)])
    print(f"    A={A:<7} gamma2={margin(Xt,y,bias=False):.6f}   "
          f"R~={rad_linear(Xt):.4f}   ratio={rad_linear(Xt)/margin(Xt,y,bias=False):.4f}")
