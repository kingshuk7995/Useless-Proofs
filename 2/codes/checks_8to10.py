"""CHECK 8-10: sympy closed forms and exact window endpoints."""
import sympy as sp, numpy as np, itertools
from scipy.optimize import minimize
from common import *

print("="*78); print("CHECK 8 - symbolic derivation of the exact two-point closed forms"); print("="*78)
w1,w2,al,A,R,dl,lam,u = sp.symbols('w1 w2 alpha A R delta lam u', real=True, positive=False)

# gamma_2(A): maximise w2*delta s.t. w1*R + w2 + alpha*A = 0 and ||(w1,w2,alpha)||=1.
# inner step: min w1^2+alpha^2 s.t. w1*R+alpha*A = -w2  (least-norm solution)
sol = sp.solve([sp.Eq(w1*R+al*A, -w2), sp.Eq(w1*A - al*R, 0)], [w1,al], dict=True)[0]
cost = sp.simplify(sol[w1]**2 + sol[al]**2)
print("  min ||(w1,alpha)||^2 given induced offset w2 :", cost, " (expected w2^2/(R^2+A^2))")
w2v = sp.solve(sp.Eq(w2**2 + cost, 1), w2)
w2pos = [s for s in w2v if sp.simplify(s.subs({R:sp.Rational(1,2),A:1})) > 0][0]
gamma2 = sp.simplify(dl*w2pos)
print("  gamma_2(A) =", sp.simplify(gamma2), "   [expected delta*sqrt((A^2+R^2)/(A^2+R^2+1))]")
print("  check A=0  :", sp.simplify(gamma2.subs(A,0)), "  [expected delta*R/sqrt(R^2+1)]")
print("  check A->oo:", sp.limit(gamma2, A, sp.oo), "  [expected delta = the with-bias margin]")

# Rademacher, exact, n=2
Rtil = sp.Rational(1,2)*(sp.sqrt(R**2+1+A**2) + dl)
print("  R~(A)      =", Rtil)

ratio = sp.simplify(Rtil/gamma2)
phi   = sp.simplify(2*dl*ratio)          # = (u+1)/sqrt(u) + delta*sqrt(1+1/u), u=A^2+R^2
phi_u = sp.simplify(phi.subs(A, sp.sqrt(u-R**2)))
print("  2*delta*ratio as a function of u = A^2+R^2 :", sp.simplify(sp.radsimp(phi_u)))
print("     -> phi(u) = (u+1)/sqrt(u) + delta*sqrt(1+1/u)")
print("     first term minimised at u=1; second term STRICTLY DECREASING in u.")
print("     => window {u>R^2 : phi(u)<phi(R^2)} is nonempty iff R<1 (delta->0),")
print("        and for R=1 (tangency) a positive delta still opens a tiny window")
print("        via the decreasing second term - exactly the 0.01% seen in CHECK 4.")

print(); print("="*78); print("CHECK 9 - closed forms vs. the independent numerical solver"); print("="*78)
f_g2 = sp.lambdify((A,R,dl), gamma2); f_rt = sp.lambdify((A,R,dl), Rtil)
print("   R     delta   A      gamma2(closed)  gamma2(numeric)   R~(closed)   R~(numeric)")
for Rv,dv in [(0.5,0.05),(1.0,0.05),(2.0,0.3)]:
    for Av in [0.0,0.5,1.0,2.0]:
        X=np.array([[Rv,1+dv],[Rv,1-dv]]); y=np.array([1.,-1.])
        Xt=np.hstack([X,np.full((2,1),Av)])
        print(f"  {Rv:<5} {dv:<6} {Av:<6} {float(f_g2(Av,Rv,dv)):.8f}      "
              f"{margin(Xt,y,bias=False):.8f}      {float(f_rt(Av,Rv,dv)):.8f}   {rad_linear(Xt):.8f}")

print(); print("="*78); print("CHECK 10 - exact window endpoints from phi(u) vs numerics"); print("="*78)
for Rv,dv in [(0.5,0.05),(0.5,0.0),(0.8,0.0),(1.0,0.0),(1.0,0.05),(1.5,0.0)]:
    uu=sp.symbols('uu',positive=True)
    ph=(uu+1)/sp.sqrt(uu)+dv*sp.sqrt(1+1/uu)
    roots=[sp.nsolve(ph-ph.subs(uu,Rv**2), uu, g) for g in (Rv**2*1.0001, 4.0)]
    cand=[float(r) for r in roots if float(r)>Rv**2+1e-9]
    if cand:
        Amax=np.sqrt(max(cand)-Rv**2)
        print(f"  R={Rv}, delta={dv}: (C2) holds for 0 < A < {Amax:.4f}")
    else:
        print(f"  R={Rv}, delta={dv}: window EMPTY - padding never helps")
