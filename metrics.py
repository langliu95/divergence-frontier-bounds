# import libraries
import time

import faiss
import numpy as np
from sklearn.preprocessing import normalize


def quantize(p, quant):
    """Quantize the distribution."""
    out = []
    for b in quant:
        out.append(np.sum(p[b]))
    return np.array(out)


def kl_div(p, q):
    """Compute the KL divergence."""
    if np.sum((p != 0) & (q == 0)) > 0:
        return np.inf
    nonzero = p != 0
    return np.sum(p[nonzero] * np.log(p[nonzero] / q[nonzero]))


def div_frontier(p, q, lambdas, quant=None):
    """Compute the DF at several points."""
    if quant is not None:
        p = quantize(p, quant)
        q = quantize(q, quant)

    div = []
    for lam in lambdas:
        comb = lam*p + (1-lam)*q
        div.append([kl_div(p, comb), kl_div(q, comb)])
    return np.array(div)


def frontier_integral(p, q, quant=None):
    """Compute the frontier integral (FI)."""
    if quant is not None:
        p = quantize(p, quant)
        q = quantize(q, quant)
    m = np.zeros(len(p))
    nonequal = p != q
    zero = (p == 0) | (q == 0)
    nonzero = ~zero
    linear = (p + q)/2
    ind = nonequal & zero
    m[ind] = linear[ind]
    ind = nonequal & nonzero
    if np.sum(ind) > 0:
        tmp = p[ind] * q[ind]/(p[ind] - q[ind])
        m[ind] = linear[ind] - tmp*(
            np.log(p[ind]) - np.log(q[ind]))
    return np.sum(m)


def frontier_integral_cont(p, q, qsample):
    """Compute the frontier integral between continuous distributions."""
    def f(t):
        return (t+1)/2 - t*np.log(t)/(t-1)
    
    fval = f(p(qsample)/q(qsample))
    return np.mean(fval)


def cluster_feats(p, q, num_clusters,
                  norm='none', num_redo=5,
                  max_iter=500, seed=0,
                  verbose=False):
    if verbose:
        print(f'seed = {seed}')
    assert norm in ['none', 'l2', 'l1', None]
    data1 = np.vstack([q, p])
    if norm in ['l2', 'l1']:
        data1 = normalize(data1, norm=norm, axis=1)
    # Cluster
    data1 = data1.astype(np.float32)
    t1 = time.time()
    kmeans = faiss.Kmeans(data1.shape[1], num_clusters, niter=max_iter,
                          verbose=verbose, nredo=num_redo, update_index=True,
                          seed=seed+2)
    kmeans.train(data1)
    _, labels = kmeans.index.search(data1, 1)
    labels = labels.reshape(-1)
    t2 = time.time()
    if verbose:
        print('kmeans time:', round(t2-t1, 2), 's')

    q_labels = labels[:len(q)]
    p_labels = labels[len(q):]

    q_bins = np.histogram(q_labels, bins=num_clusters,
                           range=[0, num_clusters], density=True)[0]
    p_bins = np.histogram(p_labels, bins=num_clusters,
                          range=[0, num_clusters], density=True)[0]
    return p_bins / p_bins.sum(), q_bins / q_bins.sum()
