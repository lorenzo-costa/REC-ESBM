import numpy as np
from scipy.special import gammaln, comb

def expected_cl_py(n, sigma, theta, H):
    n = int(n)
    if not (0 <= sigma < 1 and theta > -sigma and n > 0 and H > 1):
        raise ValueError("Invalid input: ensure 0 <= sigma < 1, theta > -sigma, n > 0, H > 1")

    if np.isinf(H):
        if sigma == 0:
            out = theta * np.sum(1 / (theta - 1 + np.arange(1, n + 1)))
        else:
            out = (1 / sigma) * np.exp(
                gammaln(theta + sigma + n) - gammaln(theta + sigma) -
                gammaln(theta + n) + gammaln(theta + 1)
            ) - theta / sigma
    else:
        if sigma == 0:
            index = np.arange(n)
            out = H - H * np.exp(np.sum(
                np.log(index + theta * (1 - 1/H)) - np.log(theta + index)
            ))
        else:
            raise NotImplementedError("Case with finite H and sigma > 0 is not implemented in the original R function.")
    
    return out

def HGnedin(V, h, gamma=0.5):
    out = np.exp(
        gammaln(V + 1) - gammaln(h + 1) - gammaln(V - h + 1)
        + gammaln(h - gamma) - gammaln(1 - gamma)
        + np.log(gamma)
        + gammaln(V + gamma - h) - gammaln(V + gamma)
    )
    return out