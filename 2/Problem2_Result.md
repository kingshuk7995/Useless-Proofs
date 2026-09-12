# Problem 2 — Resolution: the conjecture is **false as stated**, with an exact characterisation of the fragment that survives

**Verdict.** Both claims of Problem 2 are disproved.

* **(C1)** $\gamma_2>\gamma_1$ — **false**. Under the standard setting (first layer has a free bias) the margin is *exactly unchanged*, under both readings of "margin" given in the problem statement. Without a bias the margin can rise, but never above the value a single free bias scalar would give.
* **(C2)** $\mathfrak R_{\tilde S}(\mathcal H_2)/\gamma_2<\mathfrak R_S(\mathcal H_1)/\gamma_1$ — **false**. In the standard setting the two sides are *the same number*. In the only setting where padding changes anything, there are datasets on which the inequality **strictly reverses** for every $c\neq0$, and for *every* dataset it reverses once $c$ is large enough.

What does survive is a genuine trade-off with an exact criterion (Theorem 6): the bound improves only when the baseline class has **no bias**, **needs** one, and $c$ lies in a **bounded window**. This is not a guarantee, and it is non-monotone in $c$ — consistent with the paper's own Tables 1 and 2, where accuracy peaks at $K=4$ and at fill value $0.5$ and *declines* on either side.

Every statement below is verified numerically in [codes/](codes/); closed forms are re-derived symbolically with `sympy`.

---

## 0. Notation

$\iota_c:\mathbb R^d\to\mathbb R^{Kd}$, $\iota_c(x)=(x,c\mathbf 1_m)$, $m:=(K-1)d$.
$S=\{(x_i,y_i)\}_{i=1}^n$, $y_i\in\{\pm1\}$; $\tilde S=\{(\iota_c(x_i),y_i)\}$.

$$\mathfrak R_S(\mathcal H)=\mathbb E_\sigma\sup_{h\in\mathcal H}\frac1n\sum_i\sigma_ih(x_i),\qquad
\mathcal V_S(\mathcal H):=\{(h(x_1),\dots,h(x_n)):h\in\mathcal H\}\subseteq\mathbb R^n .$$

$N:=\|\sum_i\sigma_ix_i\|_2$, $T:=\sum_i\sigma_i$.

> **Observation 0.** $\mathfrak R_S(\mathcal H)$ depends on $(\mathcal H,S)$ only through the value set $\mathcal V_S(\mathcal H)$.

---

## 1. The standard setting: everything is exactly invariant

**Lemma 1 (Padding absorption).** Let $h_{W,b,\varphi}(x)=g(Wx+b;\varphi)$, where the first layer is affine — fully connected *or convolutional* (a conv layer is affine with a per-channel bias) — and $g$ is an **arbitrary** measurable map carrying all remaining parameters $\varphi$ (any depth, nonlinearity, pooling, residual structure). Let

$$\mathcal H_1=\{x\mapsto g(Wx+b;\varphi)\},\qquad
\mathcal H_2=\{\tilde x\mapsto g([W,V]\tilde x+b;\varphi)\},$$

with $W\in\mathcal A$, $V\in\mathcal B\ni0$, $\varphi\in\Phi$, and the bias $b\in\mathbb R^h$ **free and not coupled** to $(W,V)$. Then

$$\boxed{\ \mathcal H_2\circ\iota_c=\mathcal H_1\ }\qquad\text{for every }c,K.$$

*Proof.* $[W,V]\iota_c(x)+b=Wx+(b+cV\mathbf 1_m)$. ($\subseteq$) the composite is the $\mathcal H_1$ member $(W,\,b+cV\mathbf 1_m,\,\varphi)$. ($\supseteq$) take $V=0$. $\square$

This is exactly the paper's own Eq. (11) — *"expanding the dimension of the input is indeed equivalent to adding the bias terms"* — stated as an identity of **function classes** rather than of energies.

**Theorem 1 (Exact invariance).** Under Lemma 1, for every $S$, $c$, $K$:

1. $\mathcal V_{\tilde S}(\mathcal H_2)=\mathcal V_S(\mathcal H_1)$, hence $\mathfrak R_{\tilde S}(\mathcal H_2)=\mathfrak R_S(\mathcal H_1)$ **exactly**;
2. $\gamma_2=\gamma_1$ **exactly**, for any margin functional depending on $h$ only through its values on the sample;
3. therefore $\dfrac{\mathfrak R_{\tilde S}(\mathcal H_2)}{\gamma_2}=\dfrac{\mathfrak R_S(\mathcal H_1)}{\gamma_1}$.

*Proof.* (1) Lemma 1 + Observation 0. (2) the same value vectors are realisable, so the same margins are attainable. (3) combine. $\square$

> **This is the case that covers the paper's experiments.** Every architecture in its Table 1 (torchvision ResNet-18/50, DenseNet-121, MobileNet-v3, EfficientNet) has first-layer biases. For all of them the margin bound of Problem 2 is **literally the same number** before and after expansion. Verified to $2.2\times10^{-16}$ on a 3-layer tanh MLP (CHECK 1).

**Remark.** Theorem 1 is not a gap a sharper proof could close. It says the conjecture's conclusion is *unreachable*: there is nothing to prove a strict inequality **about**, because both sides are the same real number. Any hope for (C1)/(C2) requires a class whose bias is **absent or constrained**. Everything below is that case.

**Theorem 2 (Geometric margin is preserved too).** Problem 2 defines the margin as "minimum distance to the decision boundary", so take that reading. For linear classifiers *with* bias,

$$\gamma_1=\max_{w,b}\min_i\frac{y_i(\langle w,x_i\rangle+b)}{\|w\|},\qquad
\gamma_2=\max_{w,v,b}\min_i\frac{y_i(\langle w,x_i\rangle+c\langle v,\mathbf 1\rangle+b)}{\|(w,v)\|}.$$

Then $\gamma_2=\gamma_1$ for every $c,K$.

*Proof.* ($\le$) given $(w,v,b)$ set $b'=b+c\langle v,\mathbf 1_m\rangle$: the numerator is unchanged and $\|(w,v)\|\ge\|w\|$. ($\ge$) take $v=0$. $\square$

**So under both readings of "margin", (C1) is false.** (CHECK 3.)

---

## 2. The bias-free setting — the only place padding does anything

**Lemma 2 ($K$, $d$, $c$ enter only through $A:=c\sqrt{(K-1)d}$).** For the linear ball $\|\tilde w\|_2\le B$, decompose $v=\tfrac{\alpha}{\sqrt m}\mathbf 1_m+v_\perp$. Only $\langle v,c\mathbf 1_m\rangle=\alpha c\sqrt m$ affects padded points, while $\|v\|^2=\alpha^2+\|v_\perp\|^2\ge\alpha^2$, so $v_\perp$ only wastes budget. Hence

$$\mathcal V_{\tilde S}(\mathcal H_2)=\bigl\{(\langle w,x_i\rangle+\alpha A)_i:\ \|w\|^2+\alpha^2\le B^2\bigr\},\qquad A:=c\sqrt{(K-1)d}.$$

Padding with $m$ constants $=$ appending **one** constant feature of value $A$. "More expanded dimensions" and "larger fill value" are the same knob.

**Theorem 3 (The margin gain is capped by the bias margin and never attains it).**

$$\gamma_2(A)=\max_{w,b}\ \min_i\ \frac{y_i(\langle w,x_i\rangle+b)}{\sqrt{\|w\|^2+b^2/A^2}}\qquad(b=\alpha A).$$

1. $\gamma_2(A)$ is nondecreasing in $A$ with $\sup_A\gamma_2(A)=\gamma_1^{\text{bias}}$, the original data's with-bias margin. The entire attainable gain is what one free scalar gives — at zero added dimensions and zero added compute.
2. If the optimal offset $b^\*\neq0$ then $\gamma_2(A)<\gamma_1^{\text{bias}}$ **strictly** for every finite $A$, since $b^2/A^2>0$ inflates the denominator.

Verified (CHECK 6): $\gamma_2=0.114099,\,0.207935,\,0.276616,\,0.277904,\,0.277917$ for $A=0,1,10,10^2,10^3$ against $\gamma_1^{\text{bias}}=0.277917$.

---

## 3. Counterexamples

**Theorem 4 ((C2) strictly reverses).** Let $n$ be even and
$S=\{(x_j,+1),(-x_j,-1)\}_{j=1}^{n/2}$ with all $x_j$ in a common open halfspace (so $\gamma_1>0$). Let $\mathcal H_1$ be the bias-free linear ball of radius $B$. Then for every $c\neq0$, $K\ge2$:

1. $\gamma_2=\gamma_1$. With $\beta:=c\langle v,\mathbf 1_m\rangle$, the pair $(x_j,+1)$ contributes $\langle w,x_j\rangle+\beta$ and $(-x_j,-1)$ contributes $-(-\langle w,x_j\rangle+\beta)=\langle w,x_j\rangle-\beta$, so
 $$\min_i y_i\langle\tilde w,\tilde x_i\rangle=\min_j\langle w,x_j\rangle-|\beta|,$$
 maximised at $\beta=0$. The padded dimensions are **provably useless**: the induced bias hurts one class exactly as much as it helps the other.
2. $\mathfrak R_{\tilde S}(\mathcal H_2)=\frac Bn\,\mathbb E\sqrt{N^2+A^2T^2}>\frac Bn\mathbb E N=\mathfrak R_S(\mathcal H_1)$, strict because $\sqrt{N^2+A^2T^2}>N$ on $\{T\neq0\}$ and $\mathbb P(T=n)=2^{-n}>0$.
3. Hence $\mathfrak R_{\tilde S}(\mathcal H_2)/\gamma_2>\mathfrak R_S(\mathcal H_1)/\gamma_1$ **strictly**.

The conjectured improvement is not merely unprovable — it is **backwards**. For $n=2$, $d=1$, $x_1=1$, $K=2$ the ratio multiplies by exactly $1+|c|$: at the paper's own fill value $c=0.5$ the bound is **50% worse**. Verified over 18 configurations (CHECK 2); e.g. $n=10,d=3,K=3,c=1$: margin unchanged to $1.1\times10^{-15}$, ratio $0.41183\to0.64755$.

**Theorem 5 (Universal large-$c$ failure — no dataset is safe).** For any $S$ with both classes present:

1. $\gamma_2\le\frac B2 D$, $D:=\min_{y_i\neq y_j}\|x_i-x_j\|$, **independent of $c$**.
 *Proof.* For any opposite-class pair,
 $$\min_k y_k\langle\tilde w,\tilde x_k\rangle\le\tfrac12\bigl[\langle\tilde w,\tilde x_i\rangle-\langle\tilde w,\tilde x_j\rangle\bigr]=\tfrac12\langle\tilde w,(x_i-x_j,\,\mathbf 0)\rangle\le\tfrac B2\|x_i-x_j\|,$$
 because **the constant padding cancels in every class-difference direction**. This is the geometric heart of the matter: padding adds no separating information, only norm. $\square$
2. $\mathfrak R_{\tilde S}(\mathcal H_2)\ge\frac Bn A\,\mathbb E|T|\ge BA/\sqrt{2n}$ (Khintchine, Szarek's constant $1/\sqrt2$).

Hence $\mathfrak R_{\tilde S}(\mathcal H_2)/\gamma_2\ge 2A/(\sqrt{2n}\,D)\to\infty$. **(C2) can never be a theorem "for all $c$".** Verified (CHECK 6): $\gamma_2$ saturates at $0.277917=\frac B2D$ to 6 digits while the ratio grows $2.35\to811.77$ as $A:0\to10^3$.

---

## 4. What survives: the exact criterion

**Theorem 6.** In the bias-free linear-ball setting write $\rho(A):=\gamma_2(A)/\gamma_1$ and $\nu(A):=\mathbb E\sqrt{N^2+A^2T^2}/\mathbb E N$. Then

$$\text{(C2) holds at }A\iff\rho(A)>\nu(A).$$

Both are $\ge1$ and increasing; $\rho$ is **bounded** (Theorem 3.1) while $\nu$ is **unbounded** ($\nu(A)\ge A\,\mathbb E|T|/\mathbb E N$). So the helping set is bounded — a genuine trade-off, which is exactly what "mathematically guaranteeing" denies.

**An explicit family where (C2) does hold** (so the conjecture is not vacuous). Take $d=2$, $x_+=(R,1+\delta)$ with $y=+1$, $x_-=(R,1-\delta)$ with $y=-1$, bias-free ball $B=1$, one padded feature $A$. Exactly (derived by hand and re-derived symbolically, CHECK 8):

$$\gamma_2(A)=\delta\sqrt{\frac{A^2+R^2}{A^2+R^2+1}},\qquad
\mathfrak R_{\tilde S}(\mathcal H_2)=\frac{\sqrt{R^2+1+A^2}+\delta}{2},$$

so with $u:=A^2+R^2$,

$$2\delta\cdot\frac{\mathfrak R_{\tilde S}}{\gamma_2}=\varphi(u):=\frac{u+1}{\sqrt u}+\delta\sqrt{1+\tfrac1u},
\qquad\text{(C2)}\iff\varphi(u)<\varphi(R^2).$$

$(u+1)/\sqrt u$ is strictly convex with its unique minimum at $u=1$; $\delta\sqrt{1+1/u}$ is strictly decreasing. Therefore:

* $\delta\to0$: the window is nonempty **iff $R<1$**, and equals $u\in(R^2,R^{-2})$, i.e. $0<A<\sqrt{R^{-2}-R^2}$. $R=1$ is exactly the tangency case.
* $\delta>0$ widens it slightly and opens a tiny one even at $R=1$.

| $R$ | $\delta$ | window |
|---|---|---|
| 0.5 | 0 | $0<A<1.9365=\sqrt{3.75}$ |
| 0.5 | 0.05 | $0<A<2.0130$ |
| 0.8 | 0 | $0<A<0.9605$ |
| 1.0 | 0 | collapses to a point (tangency) |
| 1.0 | 0.05 | $0<A<0.2671$ |
| 1.5 | 0 | **empty** — padding never helps |

Closed forms match an independent SVM solver to 8 decimals (CHECK 9); endpoints by `nsolve` (CHECK 10). Best improvement at $R=0.5$: bound ~20% tighter at $A\approx1$. Not an $n=2$ artifact — CHECK 5 repeats it with $n=12$ jittered points for a 27% tighter bound.

**Interpretation.** The window is nonempty exactly when $R<1$, i.e. when the data sit **close to the origin relative to the offset of the good separator** — precisely when the bias-free class is misspecified and an offset is worth a lot. The whole effect is:

> constant padding is a clumsy, budget-charged way of adding a bias term.

It helps only when (i) you had no bias, (ii) you needed one, and (iii) $c$ is small enough that the added norm has not eaten the gain. None of these is what Problem 2 claims, and **(i) is false for every network in the paper**. Optimising each over its own hyperparameter on the CHECK 5 dataset, an explicit bounded bias beats padding (ratio $1.6831$ at $\beta=0.75$ vs $1.7166$ at $A=0.90$).

---

## 5. Where the conjecture's own reasoning breaks

Problem 2 argues the complexity drops "because the augmented $(K-1)d$ dimensions have **zero variance**". For the linear ball the exact identity is

$$\mathfrak R_S(\mathcal H)=\frac Bn\,\mathbb E_\sigma\Bigl\|\sum_i\sigma_ix_i\Bigr\|_2,
\qquad\Bigl\|\sum_i\sigma_i\tilde x_i\Bigr\|^2=N^2+A^2T^2 .$$

The extra term does **not** vanish: $\mathbb E[T^2]=n$ exactly. Zero *variance* does not imply zero Rademacher contribution, because $\mathfrak R_S$ is driven by the **uncentered** Gram matrix, and a constant feature aligns perfectly with $\mathbf 1$ — the direction in which $\sigma$ has an $\Theta(\sqrt n)$ component. The intuition is valid for centered capacity measures (Gaussian width of centered data, PCA rank); the conjecture silently swaps a centered notion for an uncentered one, and the numerator moves in the **opposite** direction to the one asserted.

---
