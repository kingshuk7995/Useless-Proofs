"""Shared helpers: exact empirical Rademacher complexity of a linear ball,
max-margin (hard-margin SVM) solvers, and constant padding.
Used by checks_1to3.py, checks_4to6.py, check_7.py, checks_8to10.py."""
import numpy as np, itertools
from scipy.optimize import minimize
np.set_printoptions(precision=6, suppress=True)

# ---------- exact empirical Rademacher complexity of the linear ball ----------
def rad_linear(Z, B=1.0):
    """R_S(H) = (B/n) E_sigma || sum_i sigma_i z_i ||_2 , exact by enumeration."""
    n = Z.shape[0]
    tot = 0.0
    for s in itertools.product([-1.0, 1.0], repeat=n):
        tot += np.linalg.norm(np.array(s) @ Z)
    return B * tot / (2**n) / n

# ---------- max functional margin over the norm ball (hard-margin SVM) --------
def margin(Z, y, B=1.0, bias=False):
    """B * max_{||u||=1} min_i y_i <u,z_i>  (+ free bias if bias=True).
       = B / ||w*||, w* = argmin ||w||  s.t. y_i(<w,z_i>+b) >= 1."""
    n, p = Z.shape
    def obj(v):  return 0.5*np.dot(v[:p], v[:p])
    def jac(v):  return np.concatenate([v[:p], [0.0]])
    cons = {'type':'ineq',
            'fun': lambda v: y*(Z@v[:p] + (v[p] if bias else 0.0)) - 1.0,
            'jac': lambda v: np.hstack([ (y[:,None]*Z), (y[:,None] if bias else np.zeros((n,1))) ])}
    best = None
    for _ in range(25):
        v0 = np.concatenate([np.random.randn(p), [0.0]])
        r = minimize(obj, v0, jac=jac, constraints=[cons], method='SLSQP',
                     options={'maxiter':800,'ftol':1e-12})
        if r.success and np.min(y*(Z@r.x[:p] + (r.x[p] if bias else 0.0))) > 1-1e-7:
            if best is None or obj(r.x) < obj(best): best = r.x
    if best is None: return np.nan
    return B/np.linalg.norm(best[:p])

def pad(X, c, m):
    return np.hstack([X, c*np.ones((X.shape[0], m))])

