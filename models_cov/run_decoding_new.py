#!/usr/bin/env python3
"""
Decoding performance for the linear‑line network
saved by linear_line_network.py
"""
import numpy as np, matplotlib.pyplot as plt, time, os
from sklearn.linear_model import LinearRegression

# ───────── 1. parameters ─────────
in_file = "linear_line_network.npz"
num_noise_levels   = 300
n_features_decode  = 10
trials_per_feature = 200
min_sigma, max_sigma = 0.5, 100.0   # std‑dev range for PC‑1 noise

# ───────── 2. helpers ─────────
def response(W_R, WF, f, g, xi=None):
    N=len(g); G=np.diag(g)
    ff = G @ WF(f)
    xi = np.zeros((N,1)) if xi is None else xi.reshape(N,1)
    return np.linalg.solve(np.eye(N)-G@W_R, ff+xi).flatten()

def simulate_accuracy(pc1_range, sigma, n_feat, n_trials):
    lev = np.linspace(0,1,n_feat)
    means = lev*pc1_range
    total = n_feat*n_trials
    correct = 0
    for idx,µ in enumerate(means):
        noisy = µ + np.random.normal(0,sigma,n_trials)
        decoded = np.argmin(np.abs(noisy[:,None]-means), axis=1)
        correct += np.sum(decoded==idx)
    return correct/total

# ───────── 3. load network & rebuild WF(f) ─────────
if not os.path.exists(in_file):
    raise FileNotFoundError(f"{in_file} not found – run the setup script first.")

dat      = np.load(in_file)
W_R      = dat['W_R']
q        = dat['q']                      # neuron positions
pc1,pc2  = dat['pc1'], dat['pc2']
g_align  = dat['g_aligned']
g_mis    = dat['g_misaligned']
b        = (q - q.mean()).reshape(-1,1)
WF       = lambda f: b*f                 # rebuild linear drive
feature_range = (0.0,1.0)
N        = len(q)

# ───────── 4. compute PC‑1 slopes (noiseless) ─────────
f_grid = np.linspace(*feature_range, 21)
def pc1_projection(g):
    R = np.array([response(W_R, WF, f, g) for f in f_grid])
    R -= R.mean(0)
    proj = R @ np.vstack((pc1,pc2)).T
    return proj[:,0]                     # PC‑1 component
slope = lambda proj: LinearRegression().fit(f_grid.reshape(-1,1), proj).coef_[0]

pc1_range_align = slope(pc1_projection(g_align))
pc1_range_mis   = slope(pc1_projection(g_mis))

print(f"Slope (aligned)   = {pc1_range_align:.3f}")
print(f"Slope (misaligned)= {pc1_range_mis:.3f}")

# ───────── 5. accuracy vs noise level ─────────
sigmas = np.geomspace(min_sigma, max_sigma, num_noise_levels)
acc_align=[]; acc_mis=[]
t0=time.time()
for σ in sigmas:
    acc_align.append(simulate_accuracy(pc1_range_align, σ,
                                       n_features_decode,trials_per_feature))
    acc_mis.append(simulate_accuracy(pc1_range_mis,   σ,
                                     n_features_decode,trials_per_feature))
print(f"simulation done in {time.time()-t0:.1f}s")

acc_align=np.array(acc_align)*100
acc_mis  =np.array(acc_mis)*100

# ───────── 6. plot ─────────
plt.style.use('seaborn-v0_8-whitegrid')
fig,ax=plt.subplots(figsize=(8,6))
sc=ax.scatter(acc_mis, acc_align, c=sigmas, cmap='viridis_r',
              norm=plt.matplotlib.colors.LogNorm(),
              s=60, lw=0.4, edgecolors='k', alpha=0.6)
chance=100/n_features_decode
ax.plot([chance,100],[chance,100],'--',c='gray')
ax.set_xlabel("Least‑aligned accuracy (%)")
ax.set_ylabel("Most‑aligned accuracy (%)")
ax.set_title("Decoding accuracy vs. PC‑1 noise")
ax.set_xlim(chance-5,102); ax.set_ylim(chance-5,102)
fig.colorbar(sc,label="PC‑1 noise σ")
plt.tight_layout(); plt.show()
