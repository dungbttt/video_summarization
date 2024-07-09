import numpy as np

from .cpd_nonlin import cpd_nonlin


def cpd_auto(K, ncp, vmax, desc_rate=1, **kwargs):
    
    m = ncp
    _, scores = cpd_nonlin(K, m, backtrack=False, **kwargs)

    N = K.shape[0]
    N2 = N * desc_rate  # length of the video before down-sampling

    penalties = np.zeros(m + 1)
    # Prevent division by zero (in case of 0 changes)
    ncp = np.arange(1, m + 1)
    penalties[1:] = (vmax * ncp / (2.0 * N2)) * (np.log(float(N2) / ncp) + 1)

    costs = scores / float(N) + penalties
    m_best = np.argmin(costs)
    cps, scores2 = cpd_nonlin(K, m_best, **kwargs)

    return cps, scores2
