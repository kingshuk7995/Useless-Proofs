"""CHECK 7: padding vs. an explicit bias term."""
import numpy as np, itertools
from scipy.optimize import minimize
from common import *

def rad_bias(Z, B=1.0, beta=0.0):
    """R for {x-><w,x>+b : ||w||<=B, |b|<=beta} = (B/n)E||sum s_i z_i|| + (beta/n)E|sum s_i|"""
    n = Z.shape[0]; t1=t2=0.0
    for s in itertools.product([-1.,1.], repeat=n):
        s=np.array(s); t1 += np.linalg.norm(s@Z); t2 += abs(s.sum())
    return (B*t1/(2**n) + beta*t2/(2**n))/n

def margin_boxbias(Z,y,B=1.0,beta=1.0):
    """B-scaled: max_{||w||<=B,|b|<=beta} min_i y_i(<w,z_i>+b), by 1-D search over b."""
    best=-1e9
    for b in np.linspace(-3*beta,3*beta,1201):
        if abs(b)>beta+1e-12: continue
        # max_{||w||<=B} min_i y_i(<w,z_i>+b): shift labels, solve scaled SVM
        # value = B*max_{||u||=1} min_i (y_i<u,z_i> + y_i b/B)  -> do direct concave max
        v = _mm(Z,y,b,B)
        best=max(best,v)
    return best
def _mm(Z,y,b,B):
    n,p=Z.shape
    f=lambda u: -min(y*(Z@u+b))
    best=-1e9
    for _ in range(6):
        u0=np.random.randn(p); u0*=B/np.linalg.norm(u0)
        r=minimize(f,u0,constraints=[{'type':'ineq','fun':lambda u: B**2-u@u}],method='SLSQP',
                   options={'maxiter':400,'ftol':1e-12})
        if r.success: best=max(best,-r.fun)
    return best

rng=np.random.default_rng(7)
R,delta,nh=0.5,0.30,6
Xp=np.column_stack([R+0.05*rng.normal(size=nh),1+delta+0.05*rng.normal(size=nh)])
Xm=np.column_stack([R+0.05*rng.normal(size=nh),1-delta+0.05*rng.normal(size=nh)])
X=np.vstack([Xp,Xm]); y=np.concatenate([np.ones(nh),-np.ones(nh)])

print("="*78)
print("CHECK 7 - inside the window, is PADDING better than just ADDING A BIAS?")
print("  same dataset as Check 5. Compare, at matched 'bias reach':")
print("    padding  A      -> ellipsoid class {||w||^2 + b^2/A^2 <= B^2}")
print("    explicit beta   -> cylinder  class {||w||<=B, |b|<=beta}")
print("="*78)
base = rad_linear(X)/margin(X,y,bias=False)
print(f"  baseline (no bias, no padding): ratio = {base:.6f}\n")
print("   A/beta   padding ratio      explicit-bias ratio")
for A in [0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,1.0,1.1]:
    Xt=np.hstack([X,np.full((2*nh,1),A)])
    rp=rad_linear(Xt)/margin(Xt,y,bias=False)
    gb=margin_boxbias(X,y,1.0,A); rb=rad_bias(X,1.0,A)/gb
    print(f"   {A:<7} {rp:.6f} {'(helps)' if rp<base else '(hurts)':8s}  "
          f"{rb:.6f} {'(helps)' if rb<base else '(hurts)':8s}  "
          f"-> {'BIAS wins' if rb<rp else 'PADDING wins'}")
