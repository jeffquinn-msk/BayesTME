from utils import ilogit, stable_softmax, sample_mvn_from_precision
import numpy as np
from scipy.sparse import spdiags

def sample_pg(Omegas, rates, Y_igk, Theta, pg):
    Theta_r = np.transpose(Theta, [1,2,0])
    Y_r = np.transpose(Y_igk, [1,2,0])
    rates_r = np.transpose(rates, [1,2,0])
    Trials = Y_r + rates_r
    obs_mask = Trials > 1e-4 # PG is very unstable for small trials; just ignore these as they give virtually no evidence anyway.
    obs_mask_flat = obs_mask.reshape(-1)
    trials_flat = Trials.reshape(-1)[obs_mask_flat].astype(float)
    thetas_flat = Theta_r.reshape(-1)[obs_mask_flat]
    omegas_flat = Omegas.reshape(-1)[obs_mask_flat]
    pg.pgdrawv(trials_flat, thetas_flat, omegas_flat)
    Omegas[obs_mask] = np.clip(omegas_flat, 1e-13, None)
    Omegas[~obs_mask] = 0.

def sample_spatial_weights(C, V, n_obs, Y_igk, W, H, Omegas, prior_vars):
    # Cache repeatedly calculated local variables
    Prior_precision = np.diag(1/prior_vars)
    X = np.concatenate([np.ones(W.shape)[...,None], W[...,None]], axis=-1)

    # Update each gene one-by-one
    # TODO: vectorize this
    for g in range(V.shape[0]):
        for k in range(V.shape[1]):
            # Calculate the precision term
            h = H[g,k]
            Precision = (X[k,h].T * Omegas[None,g,k]).dot(X[k,h]) + Prior_precision

            # Calculate the mean term
            mu_part = X[k,h].T.dot((Y_igk[:,g,k] - n_obs[:,g,k])/2)
#             Precision = np.clip(Precision, 1e-6, 1-1e-6)
            # Sample the offset and spatial weights
            c, v = sample_mvn_from_precision(Precision, mu_part=mu_part, sparse=False)
            C[g,k] = c
            V[g,k] = v

def sample_spatial_patterns(W, n_obs, Y_igk, C, V, H, Omegas, Cov_mats):
    for k in range(W.shape[0]):
        for j in range(1,W.shape[1]):
            mask = H[:,k] == j
            if mask.sum() == 0:
                # Pick a random instance to use as data rather than sampling from the prior
                mask[np.random.choice(H.shape[0])] = True

            Y_masked = Y_igk[:,mask,k].T
            n_obs_masked = n_obs[:,mask,k].T
            Omegas_masked = Omegas[mask, k]
            V_masked = V[mask,k:k+1]
            C_masked = C[mask,k:k+1]

            # PG likelihood terms
            a_j = (Omegas_masked*V_masked**2).sum(axis=0)
            b_j = (V_masked*((Y_masked - n_obs_masked) / 2 - (Omegas_masked*C_masked))).sum(axis=0)

            # Posterior precision
            Precision = np.copy(Cov_mats[k, j])
            Precision[np.diag_indices(Precision.shape[0])] += a_j

            # Posterior mu term
            mu_part = b_j

            # Sample the spatial pattern
            W[k,j] = sample_mvn_from_precision(Precision, mu_part=mu_part, sparse=False)

def sample_sigmainv(W, Delta, DeltaT, Tau2_a, Tau2_b, Tau2_c, Cov_mats, stability=1e-6, lam2=1):
    for i in range(W.shape[0]):
        deltas = Delta.dot(W[i].reshape(-1))
        rate = deltas**2 / (2*lam2) + 1/Tau2_c.clip(stability, 1/stability)
        Tau2 = 1/np.random.gamma(1, 1/rate.clip(stability, 1/stability)).clip(stability, 1/stability)
        Tau2_c = 1/np.random.gamma(1, 1 / (1/Tau2 + 1/Tau2_b).clip(stability, 1/stability))
        Tau2_b = 1/np.random.gamma(1, 1 / (1/Tau2_c + 1/Tau2_a).clip(stability, 1/stability))
        Tau2_a = 1/np.random.gamma(1, 1 / (1/Tau2_b + 1).clip(stability, 1/stability))
        lam_Tau = spdiags(1 / (lam2 * Tau2), 0, Tau2.shape[0], Tau2.shape[0], format='csc')
        Sigma0_inv = DeltaT.dot(lam_Tau).dot(Delta)
        for j in range(1, W.shape[1]):
            Cov_mats[i, j] = Sigma0_inv[W.shape[-1]*j:W.shape[-1]*(j+1), W.shape[-1]*j:W.shape[-1]*(j+1)].todense()

def sample_spatial_assignments(H, n_obs, W, V, C, Y_igk, Gamma):
    from scipy.stats import nbinom
    n_signals, n_cell_types = V.shape
    n_spatial_patterns = Gamma.shape[1]
    
    # TODO: vectorize this
    for g in range(n_signals):
        for k in range(n_cell_types):
            thetas = np.clip(ilogit(W[k]*V[g,k] + C[g,k]), 1e-6, 1-1e-6)
#             print(thetas)
#             print(Y_igk[None,:,g,k])
            logprobs = nbinom.logpmf(Y_igk[None,:,g,k], np.clip(n_obs[None,:,g,k], 1e-6, None), 1-thetas)
#             print(logprobs)
            logprobs = logprobs.sum(axis=1)
            logprior = np.log(Gamma[k])
            p = stable_softmax(logprobs + logprior)
            H[g,k] = np.random.choice(n_spatial_patterns, p=p)

def sample_pattern_probs(Gamma, H, alpha):
    # Conjugate Dirichlet update
    # print(H.shape, alpha.shape, H.max())
    for k, H_k in enumerate(H.T):
        Gamma[k] = np.random.dirichlet(np.array([(H_k == s).sum() for s in range(alpha.shape[0])]) + alpha)