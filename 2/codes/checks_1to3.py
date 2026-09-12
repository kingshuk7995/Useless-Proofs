"""CHECK 1-3: Lemma 1 absorption on an MLP; Theorem 4 reversal; Theorem 2 margin invariance."""
from common import *

print("="*78); print("CHECK 1 - Lemma 1 (padding absorption) on a real 3-layer tanh MLP with bias")
print("="*78)
rng = np.random.default_rng(0)
d, m, H, c = 4, 6, 5, 0.5
X = rng.normal(size=(7,d)); Xt = pad(X, c, m)
W, V, b = rng.normal(size=(H,d)), rng.normal(size=(H,m)), rng.normal(size=H)
W2, b2 = rng.normal(size=(3,H)), rng.normal(size=3); W3, b3 = rng.normal(size=(1,3)), rng.normal(size=1)
net = lambda A,bb,Z: (np.tanh(np.tanh(Z@A.T + bb) @ W2.T + b2) @ W3.T + b3).ravel()
out_padded   = net(np.hstack([W,V]), b, Xt)          # padded net on padded data
out_original = net(W, b + c*V@np.ones(m), X)         # original net, bias shifted
print("  max |f_padded(x~) - f_original_with_shifted_bias(x)| =",
      np.max(np.abs(out_padded-out_original)))
print("  => value vectors coincide, so R and gamma coincide exactly (Theorem 1).")

print(); print("="*78); print("CHECK 2 - Theorem 4: antipodal dataset, bias-free linear ball -> (C2) REVERSES")
print("="*78)
for (n_half, dd) in [(1,1),(3,2),(5,3)]:
    Xh = rng.normal(size=(n_half,dd)); Xh = np.abs(Xh) + 0.5      # all in positive orthant
    X = np.vstack([Xh, -Xh]); y = np.concatenate([np.ones(n_half), -np.ones(n_half)])
    for K in [2,3]:
        mm = (K-1)*dd
        for c in [0.25, 0.5, 1.0]:
            Xt = pad(X, c, mm)
            g1, g2 = margin(X,y,bias=False), margin(Xt,y,bias=False)
            R1, R2 = rad_linear(X), rad_linear(Xt)
            print(f"  n={2*n_half} d={dd} K={K} c={c}:  g1={g1:.6f} g2={g2:.6f} "
                  f"(g2-g1={g2-g1:+.2e})  R1={R1:.5f} R2={R2:.5f}  "
                  f"ratio1={R1/g1:.5f} ratio2={R2/g2:.5f}  -> "
                  f"{'WORSE (conjecture false)' if R2/g2 > R1/g1+1e-9 else 'better'}")

print(); print("="*78); print("CHECK 3 - Theorem 2: with a free bias, geometric margin is EXACTLY preserved")
print("="*78)
X = rng.normal(size=(8,3)); X[:4] += 2.0; y = np.array([1.,1,1,1,-1,-1,-1,-1])
for c in [0.5, 2.0]:
    for K in [2,4]:
        Xt = pad(X, c, (K-1)*3)
        print(f"  c={c} K={K}:  gamma1(bias)={margin(X,y,bias=True):.8f}   "
              f"gamma2(bias)={margin(Xt,y,bias=True):.8f}")
