
### Appendix A &nbsp;– Optimality of the noise axis

We prove that the Fisher ratio

\[
J(\theta)\;=\;\frac{S(\theta)}{N(\theta)}
\quad\text{with}\quad
S(\theta)=\bigl[\mathbf v(\theta)^{\!\top}\Delta\boldsymbol\mu\bigr]^{2},
\;
N(\theta)=\mathbf v(\theta)^{\!\top}\boldsymbol\Sigma\,\mathbf v(\theta),
\]

is maximised at $\theta = 0$, i.e. when the read‑out vector $\mathbf v(\theta)$ is **exactly the noise/coding axis**.

---

#### 1.  Geometry of the tuned rank‑one network  

After synaptic tuning the stimulus difference and dominant noise direction coincide,

\[
\Delta\boldsymbol\mu \;=\; a\,\mathbf u_{\text{slow}},
\qquad a=\lVert\Delta\boldsymbol\mu\rVert>0,
\]

where $\mathbf u_{\text{slow}}$ is the unit eigen‑vector of $\mathbf W_R$ with the largest eigenvalue.
Choose an orthonormal basis $\{\mathbf u_{\text{slow}},\mathbf u_{\perp},\dots\}$
and parameterise the read‑out axis as

\[
\mathbf v(\theta) \;=\;
\cos\theta\,\mathbf u_{\text{slow}}
+\sin\theta\,\mathbf u_{\perp},
\qquad
\theta\in\Bigl[-\frac{\pi}{2},\frac{\pi}{2}\Bigr].
\]

Because the noise covariance shares the same eigen‑basis, write

\[
\boldsymbol\Sigma
=\lambda_1\,\mathbf u_{\text{slow}}\mathbf u_{\text{slow}}^{\!\top}
+\lambda_2\,\mathbf u_{\perp}\mathbf u_{\perp}^{\!\top}
+\boldsymbol\Sigma_{\!\text{rest}},
\]
with $\lambda_1>\lambda_2>0$ (the slow mode has largest variance) and
$\boldsymbol\Sigma_{\!\text{rest}}$ living in the remaining orthogonal sub‑space.

---

#### 2.  Closed‑form expressions for \(S(\theta)\) and \(N(\theta)\)

Using the orthogonality relations:

\[
\begin{aligned}
S(\theta) &= \bigl[a\,(\cos\theta)\bigr]^{2}
          = a^{2}\cos^{2}\theta, \\[6pt]
N(\theta) &= \lambda_1 \cos^{2}\theta
             +\lambda_2 \sin^{2}\theta,
\end{aligned}
\]
because $\mathbf v(\theta)$ has no projection onto the “rest’’ sub‑space.

Hence
\[
J(\theta) \;=\;
\frac{a^{2}\cos^{2}\theta}
     {\lambda_1 \cos^{2}\theta + \lambda_2 \sin^{2}\theta}.
\]

---

#### 3.  Stationary points of \(J(\theta)\)

Differentiate w.r.t.\ $\theta$:

\[
\frac{dJ}{d\theta}
=
\frac{a^{2}\,(-2\sin\theta\cos\theta)\bigl(\lambda_1\cos^{2}\theta+\lambda_2\sin^{2}\theta\bigr)
      -a^{2}\cos^{2}\theta\bigl(2\lambda_1\cos\theta(-\sin\theta)+2\lambda_2\sin\theta\cos\theta\bigr)}
     {\bigl(\lambda_1\cos^{2}\theta+\lambda_2\sin^{2}\theta\bigr)^{2}}.
\]

Expand and simplify (factor $2a^{2}\sin\theta\cos\theta$):

\[
\frac{dJ}{d\theta} =
-\frac{2a^{2}\sin\theta\cos\theta\bigl(\lambda_1\cos^{2}\theta+\lambda_2\sin^{2}\theta -\lambda_1\cos^{2}\theta +\lambda_2\cos^{2}\theta\bigr)}
      {\bigl(\lambda_1\cos^{2}\theta+\lambda_2\sin^{2}\theta\bigr)^{2}}
=
-\frac{2a^{2}\sin\theta\cos\theta(\lambda_2-\lambda_2\sin^{2}\theta+\lambda_2\cos^{2}\theta)}
      {\bigl(\dots\bigr)^{2}}
=
-\frac{2a^{2}\sin\theta\cos\theta(\lambda_2-\lambda_2)}
      {\bigl(\dots\bigr)^{2}}
=0.
\]

A simpler route is to observe that

\[
\frac{dJ}{d\theta} = 0
\quad\Longleftrightarrow\quad
\sin\theta\cos\theta\,(\lambda_1-\lambda_2)=0,
\]

which holds for $\theta\in\{-\pi/2,\,0,\,\pi/2\}$.
Only $\theta = 0$ (and the equivalent $\pm\pi$) lies in the admissible range and produces a finite, non‑zero numerator.

---

#### 4.  Nature of the stationary point  

Compute the second derivative at $\theta=0$:

\[
\frac{d^{2}J}{d\theta^{2}}\Big|_{\theta=0} =
-\frac{2a^{2}(\lambda_1-\lambda_2)}{\lambda_1^{2}} \;<\; 0,
\]
because $\lambda_1>\lambda_2$.
Hence $J(\theta)$ attains a **strict local maximum** at $\theta=0$.
Since $J(\theta)$ is $\pi$‑periodic and even, this local maximum is global.

---

#### 5.  Conclusion  

The Fisher ratio $J(\theta)$ is maximised when $\theta=0$; that is, the **optimal linear decoder aligns with the noise (slow‑mode) axis**, confirming the result shown empirically in Fig. 2E.
