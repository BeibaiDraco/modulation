
### Alignment of stimulus and noise axes  

We focus on the feedback operator  

$$
A \;:=\; G\,W_R , \qquad \rho(A)<1 ,
$$

whose spectral radius $\rho(A)$ governs the stability of the linear‑rate dynamics.
Let  

$$
A\,u_i \;=\;\lambda_i\,u_i , \qquad  
1>\lambda_1>\lambda_2\ge\cdots\ge\lambda_N ,
$$

be its (right) eigen‑decomposition, with eigenvectors $\{u_i\}$ orthonormal in the $G^{-1}$ inner product.
The **slow‑mode axis** is the direction that relaxes back to baseline most slowly after a small perturbation; in discrete time its relaxation
constant is  

$$
\tau_i \;=\; -\frac{\Delta t}{\ln|\lambda_i|},
$$

so the slowest mode is $u_{\text{slow}}:=u_1$.

---

#### Noise axis  

Private noise $\eta(t)\!\sim\!\mathcal N(0,\sigma_\eta^2 I)$ bypasses $W_F$ and therefore propagates only through the recurrent loop.
Its steady‑state covariance is  

$$
\boldsymbol\Sigma \;=\;\sigma_\eta^{2}\sum_{k=0}^{\infty} (A A^{\!\top})^{k}
               \;=\;\sigma_\eta^{2}\bigl(I-A A^{\!\top}\bigr)^{-1}.
$$

Because this series weights $u_i$ as $(1-\lambda_i^2)^{-1}$, variance is maximal along $u_1$; hence the
**noise axis** coincides with the slow mode:  

$$
u_{\text{noise}} \;=\; u_{\text{slow}} \;=\; u_1 .
$$

---

#### Stimulus (coding) axis  

For a binary stimulus $s\!\in\!\{0,1\}$ the mean network response is  

$$
\boldsymbol\mu_s
    \;=\; (I-W_R)^{-1} W_F\,s
    \;=\; (I-A)^{-1} G\,w\,s .
$$

The discriminant therefore reads  

$$
\Delta\boldsymbol\mu
      =(I-A)^{-1}G\,w
      =\sum_{k=0}^{\infty} A^{k}\,G\,w .
$$

Expanding $G w$ in the eigenbasis gives  

$$
\Delta\boldsymbol\mu
      \;=\;\sum_{i=1}^{N} \frac{c_i}{1-\lambda_i}\,u_i,
      \qquad c_i := u_i^{\!\top} G w .
$$

Slow modes ($\lambda_i\!\approx\!1$) are thus preferentially amplified.  
If, in addition, $G w$ already points along $u_1$ (i.e.\ $c_1\!\neq\!0$ and $c_{i>1}\!=\!0$) then  

$$
u_{\text{stim}} := \frac{\Delta\boldsymbol\mu}{\|\Delta\boldsymbol\mu\|} \;=\; u_1 .
$$

This tuning rule matches the adaptive‑dynamics principle of Chadwick *et al.* (2023)fileciteturn1file0:
plasticity steers $W_F$ and $W_R$ so that high‑SNR feed‑forward drive excites the slowest recurrent mode.

---

#### Consequence for our rank‑one construction  

In our model both $W_F = w\,e^{\!\top}$ and $W_R = \rho\,z\,v^{\!\top}$ are rank‑one and chosen so that  
$G w \propto z$.  Consequently $G w$ is an eigenvector of $A$ with eigenvalue $\lambda_1=\rho$, and all of the above
conditions are satisfied.  Therefore  

$$
u_{\text{noise}} \;=\; u_{\text{slow}} \;=\; u_{\text{stim}},
$$

guaranteeing that the coding axis is perfectly aligned with the dominant noise direction—an essential prerequisite for the
optimal‑decoding result in Fig.​ 2E.  Appendix B lists the general algebraic conditions for this alignment and shows that
arbitrary rank‑one pairs $(W_F,W_R)$ do **not** guarantee it.
