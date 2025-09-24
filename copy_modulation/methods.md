
### Rank‑one E–I network and decoding‑axis analysis (Fig. 2)

#### Network architecture  
We modelled a linear rate network of \(N = 120\) neurons obeying Dale’s law (80 % excitatory, 20 % inhibitory).  
The population dynamics are  
\[
\tau \dot{\mathbf r}(t) \;=\; -\mathbf r(t) \;+\; \mathbf W_R \mathbf r(t) \;+\; \mathbf W_F\, s(t) \;+\; \boldsymbol\eta(t),
\]
where \(s(t) \in \{0,1\}\) is a binary stimulus, \(\boldsymbol\eta(t) \sim \mathcal N(\mathbf 0, \sigma_\eta^2 \mathbf I)\) is private Gaussian noise (\(\sigma_\eta = 1\)), and the time constant is absorbed into the unit‑time scaling (\(\tau = 1\)).  
Both the feed‑forward and recurrent weights are **rank‑one**:
\[
\mathbf W_F \;=\; \mathbf w\,\mathbf e^{\!\top},
\qquad
\mathbf W_R \;=\; \rho\, \mathbf z\,\mathbf v^{\!\top},\qquad \rho = 0.4.
\]
The feed‑forward driver  
\[
\mathbf w \;=\; \frac{\operatorname{sign}(\mathbf z) \odot \mathbf v}{\lVert \mathbf v \rVert},
\qquad
v_i \;=\; e^{-\kappa x_i^{2}},\; x_i \in [-1,1],\; \kappa = 4,
\]
ensures that each row of \(\mathbf W_R\) is non‑negative (E) or non‑positive (I).

Because \(\mathbf W_R\) is rank‑one, its spectrum contains a single non‑zero eigenvalue; the corresponding right eigenvector \(\mathbf u_{\text{slow}}\) defines the slowest dynamical mode and, as we show below, the dominant noise direction.

#### Steady‑state statistics  
For the linear system above the stimulus‑conditioned steady‑state mean and covariance are obtained in closed form:  
\[
\boldsymbol\mu_s = (\mathbf I - \mathbf W_R)^{-1} \mathbf W_F\, s,
\qquad
\boldsymbol\Sigma = \sigma_\eta^{2}\, (\mathbf I - \mathbf W_R)^{-1} (\mathbf I - \mathbf W_R)^{-\!\top}.
\]
The signal to be discriminated is the mean difference  
\[
\Delta \boldsymbol\mu \;=\; \boldsymbol\mu_{1} - \boldsymbol\mu_{0} \;=\; (\mathbf I - \mathbf W_R)^{-1} \mathbf w.
\]

#### Alignment of coding and noise axes  
Empirical and theoretical studies of tuned rank‑one networks (e.g. *Chadwick et al.*, 2023) show that plasticity drives the feed‑forward and recurrent singlet directions (\(\mathbf w\) and \(\mathbf z\)) to align with \(\mathbf u_{\text{slow}}\).  
Because decay time is inversely proportional to the dominant eigenvalue of \(\mathbf W_R\), the slow mode coincides with the direction of maximal private‑noise variance—the **noise axis**.  
When tuning has converged, the coding axis \(\Delta \boldsymbol\mu\) and noise axis \(\mathbf u_{\text{slow}}\) therefore coincide (Fig. 2B, bottom).

#### One‑parameter family of read‑out directions  
To quantify performance as a function of mis‑alignment we define  
\[
\mathbf v(\theta) \;=\;
\cos\theta\, \mathbf u_{\text{slow}}
\;+\;
\sin\theta\, \mathbf u_{\perp},
\qquad
\theta \in \Bigl[-\tfrac{\pi}{2},\tfrac{\pi}{2}\Bigr],
\]
where \(\mathbf u_{\perp}\) is any unit vector orthogonal to \(\mathbf u_{\text{slow}}\).  
For each \(\theta\) we compute
\[
S(\theta) \;=\; \bigl[\,\mathbf v(\theta)^{\!\top} \Delta \boldsymbol\mu \bigr]^{2},
\quad
N(\theta) \;=\; \mathbf v(\theta)^{\!\top} \boldsymbol\Sigma\, \mathbf v(\theta),
\quad
J(\theta) \;=\; \frac{S(\theta)}{N(\theta)},
\]
the signal power, noise variance, and Fisher ratio, respectively.

#### Optimality of the noise axis  
Because \(S(\theta)\) and \(N(\theta)\) are even in \(\theta\), their Maclaurin expansions begin with quadratic and constant terms, respectively.  
Thus \(S(\theta)\) decreases faster than \(N(\theta)\) as soon as \(\theta \neq 0\); the ratio \(J(\theta)\) attains its global maximum at \(\theta = 0\), i.e. on the noise/coding axis (Fig. 2E).  
A full algebraic proof is given in Appendix A.

#### Numerical implementation  
All quantities were evaluated analytically on a uniform grid of 181 angles (\(-90^{\circ}\!:\!+90^{\circ}\)).  
Code is provided in **Python 3.12** using *NumPy 1.26* and *Matplotlib 3.9*; a fixed random seed (42) renders every panel exactly reproducible.  
Although the script defines `trials_sim = 50 000`, no Monte‑Carlo sampling is required for Fig. 2.  
Outputs comprise  

* plots of \(S(\theta)\), \(N(\theta)\), and \(J(\theta)\);  
* visualisations of the Dale‑compliant recurrent matrix and its singular‑value spectrum; and  
* a CSV file (`information_dale_data.csv`) containing all numerical values.

#### Parameter summary  

| Parameter | Value | Description |
|-----------|:-----:|-------------|
| \(N\) | 120 | network size |
| \(f_E\) | 0.8 | excitatory fraction |
| \(\rho\) | 0.4 | recurrent spectral radius |
| \(\kappa\) | 4 | feed‑forward tuning width |
| \(\sigma_\eta\) | 1 | private‑noise s.d. |
