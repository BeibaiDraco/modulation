#!/usr/bin/env python3
"""
attentional_mod_1D.py
-------------------------------------------------
One‑dimensional feature version of the “attentional‑modulation” demo.

Author: ChatGPT
Date  : 2025‑05‑05
"""

import numpy as np
from numpy.linalg import inv, svd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings

# ------------------------------------------------------------
# 1.  Hyper‑parameters (edit here)
# ------------------------------------------------------------
N              = 60          # total neurons
frac_exc       = 0.8         # fraction excitatory
sigma_pref     = 0.12        # width of preferred‑feature distribution
sigma_ff       = 0.08        # width of feed‑forward tuning
sigma_rec      = 0.10        # width of recurrent tuning
w_ff_scale     = 1.0         # strength of W_F
w_rec_scale    = 0.8         # strength of W_R before row‑balance
noise_std      = 0.3         # s.d. of input noise
n_noise_trials = 3_000       # number of noise‑only trials
n_stim_trials  = 1_000       # number of stimulus trials per gain vector
search_iter    = 2_500       # random‑search budget for G optimisation
gain_low, gain_high = 0.2, 2.5  # allowed gain range
rng_seed       = 42
# ------------------------------------------------------------

rng = np.random.default_rng(rng_seed)


# ------------------------------------------------------------
# 2.  Helper functions
# ------------------------------------------------------------
def make_EI_masks(N, frac_exc, rng):
    """Return boolean masks for excitatory (E) and inhibitory (I) neurons."""
    idx = rng.permutation(N)
    n_exc = int(np.round(frac_exc * N))
    exc_mask          = np.zeros(N, dtype=bool)
    exc_mask[idx[:n_exc]] = True
    inh_mask = ~exc_mask
    return exc_mask, inh_mask


def feature_preference(N, rng, sigma_pref):
    """Assign each neuron a preferred feature μ_i ∈ [0,1]."""
    return np.clip(rng.normal(0.5, sigma_pref, size=N), 0.0, 1.0)


def gaussian_tuning_matrix(pref, sigma, scale, positive=True):
    """
    Build a (N × N) matrix where entry (i,j) depends on |μ_i – μ_j|.
    If positive=False, entries are negated (useful for I connections).
    """
    diff = np.abs(pref[:, None] - pref[None, :])
    mat  = scale * np.exp(-0.5 * (diff / sigma) ** 2)
    return mat if positive else -mat


def balanced_recurrent(pref, exc_mask, inh_mask, sigma, scale):
    """
    Build W_R with E➔* positive, I➔* negative, feature‑tuned,
    then subtract each row mean so that ∑_j W_R[i,j] ≈ 0.
    """
    W = np.zeros((len(pref), len(pref)))
    # E cells
    W[exc_mask] = gaussian_tuning_matrix(
        pref, sigma, scale, positive=True
    )[exc_mask]
    # I cells
    W[inh_mask] = gaussian_tuning_matrix(
        pref, sigma, scale, positive=False
    )[inh_mask]
    # Row balance
    W -= W.mean(axis=1, keepdims=True)
    return W


def feedforward_weights(pref, sigma, scale):
    """Return W_F (N × 1) so that neuron i gets w_i(s) = scale * exp(-(s-μ_i)²/2σ²)."""
    def tuning(s):
        return scale * np.exp(-0.5 * ((s - pref) / sigma) ** 2)
    return tuning


def steady_state(W_R, W_F_vec, G_vec, input_vec):
    """
    Solve (I - diag(G) W_R) x = diag(G) * W_F * input
    for x (N‑vector).
    """
    D = np.diag(G_vec)
    A = np.eye(W_R.shape[0]) - D @ W_R
    b = D @ W_F_vec * input_vec
    return inv(A) @ b


def generate_noise_trials(W_R, W_F_fun, G, n_trials, noise_std, rng):
    """Return [n_trials × N] matrix of steady‑state activity for noise‑only drives."""
    N = W_R.shape[0]
    xs = []
    for _ in range(n_trials):
        noise = rng.normal(0.0, noise_std, size=N)          # noise enters as 'input'
        # use zero stimulus (s=None) so W_F = 0 vector here
        x = steady_state(W_R, np.zeros(N), G, noise)
        xs.append(x)
    return np.stack(xs)


def generate_stim_trials(W_R, W_F_fun, G, n_trials, noise_std, rng):
    """Return trial matrix and list of stimulus values s used."""
    N = W_R.shape[0]
    xs, s_list = [], []
    for _ in range(n_trials):
        s = rng.uniform(0.0, 1.0)                    # random 1‑D feature
        ff = W_F_fun(s)                              # shape (N,)
        noise = rng.normal(0.0, noise_std, size=N)
        x = steady_state(W_R, ff, G, noise)
        xs.append(x)
        s_list.append(s)
    return np.stack(xs), np.array(s_list)


def alignment_score(pc_noise, pcs_stim):
    """
    Return |cos θ| between noise PC₁ and stimulus PC₁ (both N‑vectors).
    Larger → better alignment.
    """
    v1 = pc_noise / np.linalg.norm(pc_noise)
    v2 = pcs_stim / np.linalg.norm(pcs_stim)
    return np.abs(v1 @ v2)


def search_gain_vector(
    W_R, W_F_fun, target="aligned",
    n_random=2_500, refine=True, verbose=False
):
    """
    Find G that (mis)aligns stimulus PC₁ with noise PC₁.

    target = 'aligned'   → maximise |cos θ|
    target = 'misaligned'→ minimise |cos θ|
    """
    best_G, best_score = None, -np.inf if target == "aligned" else np.inf

    for i in range(n_random):
        G = rng.uniform(gain_low, gain_high, size=W_R.shape[0])

        # 1) noise PCA
        Xnoise = generate_noise_trials(W_R, W_F_fun, G,
                                       800, noise_std, rng)
        pca_noise = PCA(n_components=2).fit(Xnoise)
        pc1_noise = pca_noise.components_[0]

        # 2) stimulus PCA in the same G
        Xstim, _ = generate_stim_trials(W_R, W_F_fun, G,
                                        500, noise_std, rng)
        # Project onto same PC space
        Xstim_proj = (Xstim - pca_noise.mean_) @ pca_noise.components_.T
        pca_stim   = PCA(n_components=1).fit(Xstim_proj)
        pc1_stim_in_noise_space = pca_noise.components_.T @ pca_stim.components_[0]

        # Alignment
        score = alignment_score(pc1_noise, pc1_stim_in_noise_space)

        if (target == "aligned"   and score > best_score) or \
           (target == "misaligned" and score < best_score):
            best_score, best_G = score, G
            if verbose:
                print(f"[{target}] iter={i} → score={score:.3f}")

    # optional local refinement
    if refine:
        def objective(g):
            G = np.array(g)
            Xnoise = generate_noise_trials(W_R, W_F_fun, G,
                                           600, noise_std, rng)
            pca_noise = PCA(n_components=2).fit(Xnoise)
            pc1_noise = pca_noise.components_[0]

            Xstim, _ = generate_stim_trials(W_R, W_F_fun, G,
                                            300, noise_std, rng)
            Xstim_proj = (Xstim - pca_noise.mean_) @ pca_noise.components_.T
            pc1_stim = PCA(n_components=1).fit(Xstim_proj).components_[0]
            pc1_stim_back = pca_noise.components_.T @ pc1_stim

            score = alignment_score(pc1_noise, pc1_stim_back)
            return -score if target == "aligned" else score

        bounds = [(gain_low, gain_high)] * W_R.shape[0]
        try:
            res = minimize(objective, best_G, method="L-BFGS-B",
                           bounds=bounds, options={"maxiter": 200})
            if res.success:
                best_G = res.x
        except Exception as e:
            warnings.warn(f"local refinement failed: {e}")

    return best_G, best_score


def plot_results(Xnoise, Xstim1, Xstim2, title=""):
    """Scatter noise + 2 stimulus sets in noise‑PC space (first two PCs)."""
    pca_noise = PCA(n_components=2).fit(Xnoise)
    Xn  = (Xnoise - pca_noise.mean_) @ pca_noise.components_.T
    Xs1 = (Xstim1 - pca_noise.mean_) @ pca_noise.components_.T
    Xs2 = (Xstim2 - pca_noise.mean_) @ pca_noise.components_.T

    plt.figure(figsize=(6,6))
    plt.scatter(Xn[:,0],  Xn[:,1],  s=8,  alpha=0.15, label="noise",   marker=".")
    plt.scatter(Xs1[:,0], Xs1[:,1], s=16, alpha=0.6,  label="stim (mis‑aligned)")
    plt.scatter(Xs2[:,0], Xs2[:,1], s=16, alpha=0.6,  label="stim (aligned)")
    plt.axhline(0, lw=0.5); plt.axvline(0, lw=0.5)
    plt.xlabel("Noise PC1"); plt.ylabel("Noise PC2")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 3.  Build network
# ------------------------------------------------------------
# E/I identities and feature prefs
exc_mask, inh_mask = make_EI_masks(N, frac_exc, rng)
mu = feature_preference(N, rng, sigma_pref)

# Recurrent & feed‑forward weights
W_R = balanced_recurrent(mu, exc_mask, inh_mask,
                         sigma_rec, w_rec_scale)
W_F_fun = feedforward_weights(mu, sigma_ff, w_ff_scale)

print("Network built: W_R shape", W_R.shape)

# ------------------------------------------------------------
# 4.  Find G vectors
# ------------------------------------------------------------
print("Searching for mis‑aligned gain vector ...")
G_mis, sc_mis = search_gain_vector(W_R, W_F_fun,
                                   target="misaligned",
                                   n_random=search_iter,
                                   refine=True, verbose=False)
print("Searching for aligned gain vector ...")
G_aln, sc_aln = search_gain_vector(W_R, W_F_fun,
                                   target="aligned",
                                   n_random=search_iter,
                                   refine=True, verbose=False)
print(f"→ |cos θ|  aligned={sc_aln:.3f} , mis‑aligned={sc_mis:.3f}")

# ------------------------------------------------------------
# 5.  Generate final visualisation data
# ------------------------------------------------------------
Xnoise  = generate_noise_trials(W_R, W_F_fun, G_aln,
                                n_noise_trials, noise_std, rng)
Xstim_mis,  _ = generate_stim_trials(W_R, W_F_fun, G_mis,
                                     n_stim_trials, noise_std, rng)
Xstim_aln, _ = generate_stim_trials(W_R, W_F_fun, G_aln,
                                    n_stim_trials, noise_std, rng)

# ------------------------------------------------------------
# 6.  Plot
# ------------------------------------------------------------
plot_results(Xnoise, Xstim_mis, Xstim_aln,
             title="Stimulus vs Noise clouds in noise‑PC space")

