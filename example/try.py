# %%
import numpy as np
from scipy.optimize import minimize, LinearConstraint, linprog
# %%
rng = np.random.default_rng()
k = rng.normal(size=(3, 4), scale=1, loc=0)
K_in = rng.normal(size=(3, 64), scale=1, loc=0)
k_out = rng.normal(size=(3, 120), scale=1, loc=0)
# %%
K_in.shape
# %%
out_in = k.T @ K_in
out_out = k.T @ k_out
# %%
epison = 1
bias = -out_in.min(axis=1) + epison
# %%
