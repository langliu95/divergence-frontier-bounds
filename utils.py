# import libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathos.multiprocessing import ProcessingPool as Pool
from scipy.stats import multivariate_t

from metrics import cluster_feats, div_frontier, frontier_integral

COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['axes.titlesize'] = 19
mpl.rcParams['lines.markersize'] = 7.5
mpl.rcParams["mathtext.fontset"] = 'cm'
plt.rcParams["font.family"] = "Times New Roman"

CORES = 4


################################################################
################################################################
# Computing divergences
################################################################
################################################################

def generate_data(size, dist, pars):
    if dist == 'normal':
        sample = np.random.multivariate_normal(pars[0], pars[1], size=size)
    if dist == 't':
        sample = multivariate_t.rvs(loc=pars[0], shape=pars[1], df=pars[2], size=size)
    return sample


################################################################
# divergence frontier
################################################################


def sup_error(psample, qsample, estimator, supp, p, q, lambdas):
    """Compute the sup error of the empirical DF."""
    phat = estimator(psample, supp)
    qhat = estimator(qsample, supp)
    emp_df = div_frontier(phat, qhat, lambdas)
    df = div_frontier(p, q, lambdas)
    diff = np.sum(np.abs(emp_df - df), axis=1)
    #diff = np.sqrt(np.sum((emp_df - df)**2, axis=1))
    return np.max(diff)


def construct_supe_dict(krange, nrange):
    error = {
        'Empirical': [], 'Good-Turing': [], 'Laplace': [],
        'Krichevsky-Trofimov': [], 'Braess-Sauer': [],
        'Bound': stat_error_bound(krange, nrange),
        'Oracle bound': []}
    return error


def compute_sup_error(p, q, n, error, lambdas):
    """Compute the absolute error of different distribution estimators.
    """
    supp = len(p)
    psample = np.random.choice(range(supp), size=n, p=p)
    qsample = np.random.choice(range(supp), size=n, p=q)
    error['Empirical'].append(sup_error(
        psample, qsample, empirical_est, supp, p, q, lambdas))
    error['Good-Turing'].append(sup_error(
        psample, qsample, good_turing_est, supp, p, q, lambdas))
    error['Laplace'].append(sup_error(
        psample, qsample, laplace_est, supp, p, q, lambdas))
    error['Krichevsky-Trofimov'].append(sup_error(
        psample, qsample, krichevsky_trofimov_est, supp, p, q, lambdas))
    error['Braess-Sauer'].append(sup_error(
        psample, qsample, braess_sauer_est, supp, p, q, lambdas))
    return error


def average_supe(error, supe):
    nrepeat = len(error)
    est = ['Empirical', 'Good-Turing', 'Laplace',
           'Krichevsky-Trofimov', 'Braess-Sauer']
    est_error = {name: [] for name in est}
    for err in error:
        for name in est:
            est_error[name].append(err[name])
    
    for name in est:
        supe[f'{name} avg'] = np.mean(est_error[name], axis=0)
        supe[f'{name} se'] = np.std(est_error[name], axis=0) / np.sqrt(nrepeat)
    return pd.DataFrame(data=supe)


################################################################
# frontier integral
################################################################


def stat_error_bound(krange, nrange):
    """Compute the upper bound for the statistical error."""
    bound = []
    for k in krange:
        for n in nrange:
            bound.append(
                (2*np.log(n)+1)*(k/n+np.sqrt(k/n)))
    return np.array(bound)


def stat_error_oracle_bound(nrange, p, q):
    """Compute the oracle bound for the statistical error."""
    # total variation
    const = np.log(nrange) + 0.5
    # const = 0.5
    bound = const * np.sum(np.sqrt(p*(1-p))) / np.sqrt(nrange)
    bound += const * np.sum(np.sqrt(q*(1-q))) / np.sqrt(nrange)
    # missing mass
    p_nonzero = p != 0
    q_nonzero = q != 0
    subp = p[p_nonzero]
    subq = q[q_nonzero]
    for i, n in enumerate(nrange):
        bound[i] += np.sum((1-subp)**n * subp * (
            np.maximum(1, np.log(1/subp)) + 0.5))
        bound[i] += np.sum((1-subq)**n * subq * (
            np.maximum(1, np.log(1/subq)) + 0.5))
    return bound


def absolute_error(psample, qsample, estimator, supp, p, q):
    """Compute the absolute error of the empirical FI."""
    phat = estimator(psample, supp)
    qhat = estimator(qsample, supp)
    emp_mr = frontier_integral(phat, qhat)
    mr = frontier_integral(p, q)
    return np.abs(emp_mr - mr)


def construct_mae_dict(krange, nrange):
    error = {
        'Empirical': [], 'Good-Turing': [], 'Laplace': [],
        'Krichevsky-Trofimov': [], 'Braess-Sauer': [],
        'Bound': stat_error_bound(krange, nrange),
        'Oracle bound': []}
    return error


def compute_absolute_error(p, q, n, error):
    """Compute the absolute error of different distribution estimators.
    """
    supp = len(p)
    psample = np.random.choice(range(supp), size=n, p=p)
    qsample = np.random.choice(range(supp), size=n, p=q)
    error['Empirical'].append(absolute_error(
        psample, qsample, empirical_est, supp, p, q))
    error['Good-Turing'].append(absolute_error(
        psample, qsample, good_turing_est, supp, p, q))
    error['Laplace'].append(absolute_error(
        psample, qsample, laplace_est, supp, p, q))
    error['Krichevsky-Trofimov'].append(absolute_error(
        psample, qsample, krichevsky_trofimov_est, supp, p, q))
    error['Braess-Sauer'].append(absolute_error(
        psample, qsample, braess_sauer_est, supp, p, q))
    return error


def average_mae(error, mae):
    nrepeat = len(error)
    est = ['Empirical', 'Good-Turing', 'Laplace',
           'Krichevsky-Trofimov', 'Braess-Sauer',
           'Bound', 'Oracle bound']
    est_error = {name: [] for name in est}
    for err in error:
        for name in est:
            est_error[name].append(err[name])
    
    for name in est:
        mae[f'{name} avg'] = np.mean(est_error[name], axis=0)
        mae[f'{name} se'] = np.std(est_error[name], axis=0) / np.sqrt(nrepeat)
    return pd.DataFrame(data=mae)


def average_quant(error, qe):
    nrepeat = len(error)
    strategy = ['Uniform', 'Greedy', 'Oracle']
    quant_error = {name: [] for name in strategy}
    for err in error:
        for name in strategy:
            quant_error[name].append(err[name])
    
    for name in strategy:
        qe[f'{name} avg'] = np.mean(quant_error[name], axis=0)
        qe[f'{name} se'] = np.std(quant_error[name], axis=0) / np.sqrt(nrepeat)
    return pd.DataFrame(data=qe)


def estimate_fi_cont(dist, pars, size, krange, quant='kmeans'):
    """Estimate the FI between continuous distributions."""
    psample = generate_data(size, dist, pars[0])
    qsample = generate_data(size, dist, pars[1])
    est_fi = []
    for k in krange:
        if quant == 'kmeans':
            phat, qhat = cluster_feats(psample, qsample, int(k))
        else:
            raise ValueError('Only k-means clustering is implemented.')
        est_fi.append(frontier_integral(phat, qhat))
    return np.array(est_fi)


################################################################
# distribution estimators
################################################################

def empirical_est(sample, supp):
    return np.bincount(sample, minlength=supp) / (len(sample)+0.0)


def good_turing_est(sample, supp):
    n = len(sample)
    freq = np.bincount(sample, minlength=supp)

    freq_hi = np.max(freq) + 1
    trange = np.arange(freq_hi)
    count = np.zeros(freq_hi, dtype=int)
    for t in trange:
        count[t] = np.sum(freq == t)
    shat = count * trange / (n+0.0)
    gt_ind = trange[:-1][trange[:-1] <= count[1:]]
    shat[gt_ind] = (count[gt_ind+1] + 1.0)*(gt_ind + 1)/n
    
    est = [shat[freq[a]]/count[freq[a]] for a in range(supp)]
    return est / np.sum(est)


def laplace_est(sample, supp):
    est = np.bincount(sample, minlength=supp) + 1
    return est / np.sum(est)


def krichevsky_trofimov_est(sample, supp):
    est = np.bincount(sample, minlength=supp) + 0.5
    return est / np.sum(est)


def braess_sauer_est(sample, supp):
    count = np.bincount(sample, minlength=supp)
    est = np.array(count) + 0.0
    est += 0.75
    est[count == 0] -= 0.25
    est[count == 1] += 0.25
    return est / np.sum(est)


################################################################
################################################################
# synthetic data
################################################################
################################################################

def generate_dist(supp, dist_type, order=None, shuffle=False):
    """Generate distributions for synthetic data."""
    if dist_type == 'step':
        p = np.zeros(supp)
        half = int(supp*0.5)
        p[:half] = 0.5
        p[half:] = 1.5
    if dist_type == 'zipf':
        p = 1/(np.arange(1, supp+1) + 0.0)**order
    if dist_type == 'uniform':
        p = np.random.dirichlet(np.ones(supp), size=1)[0]
    if dist_type == 'dirichlet':
        p = np.random.dirichlet(0.5*np.ones(supp), size=1)[0]

    p = p / np.sum(p)
    if shuffle:
        np.random.shuffle(p)
    return p


################################################################
# divergence frontier
################################################################

def _supe_varyn_synthetic(repeat, pdist, orderp, qdist, orderq,
                          supp, nrange, lambdas):
    np.random.seed(repeat)
    p = generate_dist(supp, pdist, order=orderp)
    q = generate_dist(supp, qdist, order=orderq)
    error = construct_supe_dict([supp], nrange)
    error['Oracle bound'] = stat_error_oracle_bound(nrange, p, q)
    for n in nrange:
        error = compute_sup_error(p, q, n, error, lambdas)
    return error


def supe_varyn_synthetic(supp, nrange, dist_pairs, lambdas, nrepeat=100,
                         prefix='../results/supe', save=True):
    """Compute sup error for varying sample size."""
    dfs = []
    for pair in dist_pairs:
        pdist, orderp = pair[0]
        qdist, orderq = pair[1]
        def worker(repeat):
            return _supe_varyn_synthetic(
                repeat, pdist, orderp, qdist, orderq, supp, nrange, lambdas)
 
        with Pool(processes=CORES) as pool:
            error = pool.map(worker, range(nrepeat))
        supe = {'Sample size': nrange}
        df = average_mae(error, supe)
        if save:
            fname = f'{prefix}/nvary-{pdist}{orderp}-{qdist}{orderq}-supp{supp}-df.pkl'
            df.to_pickle(fname)
        dfs.append(df)
    return dfs


def _supe_varyk_synthetic(repeat, pdist, orderp, qdist,
                          orderq, krange, n, lambdas):
    np.random.seed(repeat)
    error = construct_supe_dict(krange, [n])
    error['Oracle bound'] = np.zeros(len(krange))
    for i, k in enumerate(krange):
        p = generate_dist(k, pdist, order=orderp)
        q = generate_dist(k, qdist, order=orderq)
        error['Oracle bound'][i] = stat_error_oracle_bound([n], p, q)[0]
        error = compute_sup_error(p, q, n, error, lambdas)
    return error


def supe_varyk_synthetic(krange, n, dist_pairs, lambdas, nrepeat=100,
                         prefix='../results/supe', save=True):
    """Compute sup error for varying support size."""    
    dfs = []
    for pair in dist_pairs:
        pdist, orderp = pair[0]
        qdist, orderq = pair[1]
        def worker(repeat):
            return _supe_varyk_synthetic(
                repeat, pdist, orderp, qdist, orderq, krange, n, lambdas)
        
        with Pool(processes=CORES) as pool:
            error = pool.map(worker, range(nrepeat))
        supe = {'Support size': krange}
        df = average_mae(error, supe)
        if save:
            fname = f'{prefix}/kvary-{pdist}{orderp}-{qdist}{orderq}-size{n}-df.pkl'
            df.to_pickle(fname)
        dfs.append(df)
    return dfs


def _supe_tail_synthetic(repeat, pdist, orderp, qdist,
                         orderq, supp, n, lambdas):
    # only orderq can be a list
    np.random.seed(repeat)
    error = construct_supe_dict([supp], [n]*len(orderq))
    error['Oracle bound'] = np.zeros(len(orderq))
    for i, rq in enumerate(orderq):
        p = generate_dist(supp, pdist, order=orderp)
        q = generate_dist(supp, qdist, order=rq)
        error['Oracle bound'][i] = stat_error_oracle_bound([n], p, q)[0]
        error = compute_sup_error(p, q, n, error, lambdas)
    return error


def supe_tail_synthetic(supp, n, orderq, dists, lambdas, nrepeat=100,
                        prefix='../results/supe', save=True):
    """Compute sup for varying tail decay."""
    qdist = 'zipf'
    dfs = []
    for (pdist, orderp) in dists:
        def worker(repeat):
            return _supe_tail_synthetic(
                repeat, pdist, orderp, qdist, orderq, supp, n, lambdas)

        with Pool(processes=CORES) as pool:
            error = pool.map(worker, range(nrepeat))
        mae = {'Tail decay': orderq}
        df = average_mae(error, mae)
        if save:
            fname = f'{prefix}/qvary-{pdist}{orderp}-supp{supp}-size{n}-df.pkl'
            df.to_pickle(fname)
        dfs.append(df)
    return dfs


################################################################
# frontier integral
################################################################

def _mae_varyn_synthetic(repeat, pdist, orderp, qdist, orderq,
                         supp, nrange):
    np.random.seed(repeat)
    p = generate_dist(supp, pdist, order=orderp)
    q = generate_dist(supp, qdist, order=orderq)
    error = construct_mae_dict([supp], nrange)
    error['Oracle bound'] = stat_error_oracle_bound(nrange, p, q)
    for n in nrange:
        error = compute_absolute_error(p, q, n, error)
    return error


def mae_varyn_synthetic(supp, nrange, dist_pairs, nrepeat=100,
                        prefix='../results/mae', save=True):
    """Compute MAE for varying sample size."""
    dfs = []
    for pair in dist_pairs:
        pdist, orderp = pair[0]
        qdist, orderq = pair[1]
        error = []
            
        def worker(repeat):
            return _mae_varyn_synthetic(
                repeat, pdist, orderp, qdist, orderq, supp, nrange)

        with Pool(processes=CORES) as pool:
            error = pool.map(worker, range(nrepeat))

        mae = {'Sample size': nrange}
        df = average_mae(error, mae)
        if save:
            fname = f'{prefix}/nvary-{pdist}{orderp}-{qdist}{orderq}-supp{supp}.pkl'
            df.to_pickle(fname)
        dfs.append(df)
    return dfs


def _mae_varyk_synthetic(repeat, pdist, orderp, qdist,
                         orderq, krange, n):
    np.random.seed(repeat)
    error = construct_mae_dict(krange, [n])
    error['Oracle bound'] = np.zeros(len(krange))
    for i, k in enumerate(krange):
        p = generate_dist(k, pdist, order=orderp)
        q = generate_dist(k, qdist, order=orderq)
        error['Oracle bound'][i] = stat_error_oracle_bound([n], p, q)[0]
        error = compute_absolute_error(p, q, n, error)
    return error


def mae_varyk_synthetic(krange, n, dist_pairs, nrepeat=100,
                        prefix='../results/mae', save=True):
    """Compute MAE for varying support size."""    
    dfs = []
    for pair in dist_pairs:
        pdist, orderp = pair[0]
        qdist, orderq = pair[1]
            
        def worker(repeat):
            return _mae_varyk_synthetic(
                repeat, pdist, orderp, qdist, orderq, krange, n)

        with Pool(processes=CORES) as pool:
            error = pool.map(worker, range(nrepeat))

        mae = {'Support size': krange}
        df = average_mae(error, mae)
        if save:
            fname = f'{prefix}/kvary-{pdist}{orderp}-{qdist}{orderq}-size{n}.pkl'
            df.to_pickle(fname)
        dfs.append(df)
    return dfs


def _mae_tail_synthetic(repeat, pdist, orderp, qdist,
                        orderq, supp, n):
    # only orderq can be a list
    np.random.seed(repeat)
    error = construct_mae_dict([supp], [n]*len(orderq))
    error['Oracle bound'] = np.zeros(len(orderq))
    for i, rq in enumerate(orderq):
        p = generate_dist(supp, pdist, order=orderp)
        q = generate_dist(supp, qdist, order=rq)
        error['Oracle bound'][i] = stat_error_oracle_bound([n], p, q)[0]
        error = compute_absolute_error(p, q, n, error)
    return error


def mae_tail_synthetic(supp, n, orderq, dists, nrepeat=100,
                       prefix='../results/mae', save=True):
    """Compute MAE for varying tail decay."""
    qdist = 'zipf'
    dfs = []
    for (pdist, orderp) in dists:

        def worker(repeat):
            return _mae_tail_synthetic(
                 repeat, pdist, orderp, qdist, orderq, supp, n)

        with Pool(processes=CORES) as pool:
            error = pool.map(worker, range(nrepeat))

        mae = {'Tail decay': orderq}
        df = average_mae(error, mae)
        if save:
            fname = f'{prefix}/qvary-{pdist}{orderp}-supp{supp}-size{n}.pkl'
            df.to_pickle(fname)
        dfs.append(df)
    return dfs


def mae_quant_level(nrange, dists, pars, true_fis, nrepeat=100,
                    prefix='results/mae', fnames=None, save=True):
    """Compute MAE for varying sample size."""
    rates = np.arange(2, 6).astype(int)
    dfs = []
    for i, (dist, par) in enumerate(zip(dists, pars)):
        df = []
        if i % 2 == 0:
            const = 5
        else:
            const = 10
        for size in nrange:
            krange = (const * size**(1.0/rates)).astype(int)
            
            def worker(repeat):
                np.random.seed(repeat)
                est_fi = estimate_fi_cont(dist, par, size, krange)
                return np.abs(est_fi - true_fis[i])

            with Pool(processes=CORES) as pool:
                error = pool.map(worker, range(nrepeat))

            df.append(np.concatenate(
                [[size], np.mean(error, axis=0), np.std(error, axis=0)/nrepeat**0.5]))
            
        if save:
            fname = f'{prefix}/{fnames[i]}.txt'
            np.savetxt(fname, df)
        dfs.append(df)
    return dfs


################################################################
################################################################
# real data
################################################################
################################################################


################################################################
# frontier integral
################################################################

def _mae_varyn_real(repeat, p, q, nrange):
    np.random.seed(repeat)
    supp = len(p)
    error = construct_mae_dict([supp], nrange)
    error['Oracle bound'] = stat_error_oracle_bound(nrange, p, q)
    for n in nrange:
        error = compute_absolute_error(p, q, n, error)
    return error


def mae_varyn_real(p, q, nrange, nrepeat=100):

    def worker(repeat):
        return _mae_varyn_real(repeat, p, q, nrange)

    with Pool(processes=CORES) as pool:
        error = pool.map(worker, range(nrepeat))
    mae = {'Sample size': nrange}
    df = average_mae(error, mae)
    return df


def _mae_varyk_real(disc_dict, krange, repeat, n):
    # disc_dict = {k: mauve_obj}
    np.random.seed(repeat)
    error = construct_mae_dict(krange, [n])
    error['Oracle bound'] = np.zeros(len(krange))
    for i, k in enumerate(krange):
        # load p and q of size k
        p = disc_dict[k].p_hist
        q = disc_dict[k].q_hist
        error['Oracle bound'][i] = stat_error_oracle_bound([n], p, q)[0]
        error = compute_absolute_error(p, q, n, error)
    return error


def mae_varyk_real(disc_dict, n, nrepeat=100):
    krange = [k for k in list(disc_dict.keys()) if k >= 8]

    def worker(repeat):
        return _mae_varyk_real(disc_dict, krange, repeat, n)

    with Pool(processes=CORES) as pool:
        error = pool.map(worker, range(nrepeat))
    # compute avg and se
    mae = {'Support size': krange}
    df = average_mae(error, mae)
    return df


################################################################
# divergence frontiers
################################################################

def _supe_varyn_real(repeat, p, q, nrange, lambdas):
    np.random.seed(repeat)
    supp = len(p)
    error = construct_supe_dict([supp], nrange)
    error['Oracle bound'] = stat_error_oracle_bound(nrange, p, q)
    for n in nrange:
        error = compute_sup_error(p, q, n, error, lambdas)
    return error


def supe_varyn_real(p, q, nrange, lambdas, nrepeat=100):
    def worker(repeat):
        return _supe_varyn_real(repeat, p, q, nrange, lambdas)
    with Pool(processes=CORES) as pool:
        error = pool.map(worker, range(nrepeat))

    supe = {'Sample size': nrange}
    df = average_mae(error, supe)
    return df


def _supe_varyk_real(disc_dict, krange, repeat, n, lambdas):
    # disc_dict = {k: mauve_obj}
    np.random.seed(repeat)
    error = construct_supe_dict(krange, [n])
    error['Oracle bound'] = np.zeros(len(krange))
    for i, k in enumerate(krange):
        # load p and q of size k
        p = disc_dict[k].p_hist
        q = disc_dict[k].q_hist
        error['Oracle bound'][i] = stat_error_oracle_bound([n], p, q)[0]
        error = compute_sup_error(p, q, n, error, lambdas)
    return error


def supe_varyk_real(disc_dict, n, lambdas, nrepeat=100):
    krange = [k for k in list(disc_dict.keys()) if k >= 8]
    def worker(repeat):
        return _supe_varyk_real(disc_dict, krange, repeat, n, lambdas)
    with Pool(processes=CORES) as pool:
        error = pool.map(worker, range(nrepeat))

    # compute avg and se
    supe = {'Support size': krange}
    df = average_mae(error, supe)
    return df


################################################################
# quantization
################################################################

def quantize(p, quant):
    """Quantize the distribution."""
    out = []
    for b in quant:
        out.append(np.sum(p[b]))
    return np.array(out)


def uniform_quant(p, nbin):
    """Uniform quantization."""
    k = len(p)
    bin_size = int(k / nbin)
    quant = []
    for i in range(nbin):
        if i < nbin - 1:
            quant.append(range(i*bin_size, (i+1)*bin_size))
        else:
            quant.append(range(i*bin_size, k))
    return quant


def f_div(p, q):
    """Compute f-divergence."""
    k = len(p)
    out = np.zeros(k)
    nonequal = p != q
    pzero = (p == 0) & nonequal
    out[pzero] = 0.5
    qzero = (q == 0) & nonequal
    out[qzero] = np.inf
    ind = nonequal & (~pzero) & (~qzero)
    x = p[ind] / q[ind]
    out[ind] = (x + 1)*0.5 - x/(x-1)*np.log(x)
    return out


def f_div_conj(p, q):
    """Compute the conjugate f-divergence."""
    k = len(p)
    out = np.zeros(k)
    nonequal = p != q
    pzero = (p == 0) & nonequal
    out[pzero] = np.inf
    qzero = (q == 0) & nonequal
    out[qzero] = 0.5
    ind = nonequal & (~pzero) & (~qzero)
    x = q[ind] / p[ind]
    out[ind] = (x + 1)*0.5 - x/(x-1)*np.log(x)
    return out


def oracle_quant(p, q, nbin):
    """Oracle quantization."""
    k = int(nbin/2)
    # quantization based on p/q
    smaller = p <= q
    fdiv = f_div(p, q)
    eps = 0.5 / k
    quant1 = [[] for _ in range(k)]
    for i, f in enumerate(fdiv):
        if smaller[i]:
            b = int(f / eps)
            if b < k:
                quant1[b].append(i)
            if b == k:
                quant1[b-1].append(i)
    # quantization based on q/p
    larger = p > q
    fdiv = f_div_conj(p, q)
    eps = 0.5 / k
    quant2 = [[] for _ in range(k)]
    for i, f in enumerate(fdiv):
        if larger[i]:
            b = int(f / eps)
            if b < k:
                quant2[b].append(i)
            if b == k:
                quant2[b-1].append(i)
    return quant1 + quant2


def insert(edge, ind):
    """Insert an edge to the partition."""
    if ind in edge:
        raise ValueError('Index already exists.')
    if len(edge) == 0:
        return [ind]

    index = len(edge)
    for i, e in enumerate(edge):
        if e > ind:
            index = i
            break
    edge = edge[:index] + [ind] + edge[index:]
    return edge


def merge_bin_fi(p, q, edge):
    """Compute the FI for the quantized distributions."""
    N = len(p)
    mauve = []
    low = [0] + edge
    high = edge + [N]
    for lo, hi in zip(low, high):
        if lo < hi:
            mauve.append(frontier_integral(
                np.sum(p[lo:hi], keepdims=True),
                np.sum(q[lo:hi], keepdims=True)))
    return mauve


def sort_ratio(p, q):
    """Sort p and q according to their ratios."""
    zero = q == 0
    ind = np.arange(len(q))
    order = ind[zero]
    nonzero = ~zero
    ind = ind[nonzero]
    tmp = np.argsort(p[nonzero] / q[nonzero])
    order = np.concatenate([ind[tmp], order])
    return p[order], q[order]


def greedy_quant(p, q, nbin, log=False):
    """Greedy quantization."""
    N = len(p)
    p, q = sort_ratio(p, q)

    obj = []
    edge = []  # edge belongs to the right bin
    for k in range(nbin-1):
        bin_mauve = merge_bin_fi(p, q, edge)
        low = [0] + edge
        high = edge + [N]
        # add another edge
        new_mauve = np.zeros(N)
        for lo, hi in zip(low, high):
            for ind in range(lo+1, hi):  # tentative new edge
                new_bin_mauve = merge_bin_fi(
                    p[lo:hi], q[lo:hi], [ind])
                new_mauve[ind] = np.sum(bin_mauve) - merge_bin_fi(
                    p[lo:hi], q[lo:hi], [])[0] + np.sum(new_bin_mauve)
        edge = insert(edge, np.argmax(new_mauve))
        if log or (k == nbin-2):
            obj.append(np.sum(merge_bin_fi(p, q, edge)))
    return obj, edge


def _quant_synthetic(repeat, pdist, orderp, qdist,
                     orderq, supp, krange):
    np.random.seed(repeat)
    p = generate_dist(supp, pdist, order=orderp)
    q = generate_dist(supp, qdist, order=orderq)
    full = frontier_integral(p, q)

    error = {'Uniform': [], 'Oracle': []}
    greedy = np.array(greedy_quant(p, q, krange[-1], log=True)[0])
    error['Greedy'] = full - greedy[krange - 2]
    for k in krange:
        quant = uniform_quant(p, k)
        error['Uniform'].append(full - frontier_integral(p, q, quant))
        quant = oracle_quant(p, q, k)
        error['Oracle'].append(full - frontier_integral(p, q, quant))
    return error


def quant_synthetic(supp, krange, dist_pairs, nrepeat=100,
                    prefix='../results/mae', save=True):
    """Compute MAE for population level quantization with varying support size."""    
    dfs = []
    for pair in dist_pairs:
        pdist, orderp = pair[0]
        qdist, orderq = pair[1]
        def worker(repeat):
            return _quant_synthetic(
                repeat, pdist, orderp, qdist, orderq, supp, krange)
        with Pool(processes=CORES) as pool:
            error = pool.map(worker, range(nrepeat))

        qe = {'Number of bins': krange}
        df = average_quant(error, qe)
        if save:
            fname = f'{prefix}/quant-{pdist}{orderp}-{qdist}{orderq}-supp{supp}.pkl'
            df.to_pickle(fname)
        dfs.append(df)
    return dfs


################################################################
# plot
################################################################

def plot_stat_bound(dfs, xlabels, titles, const=100, fname=None, save=False, log_scale=True):
    fig, axes = plt.subplots(nrows=1, ncols=4, sharey=True)
    fig.set_figheight(4)
    fig.set_figwidth(15)
    axes[0].set_ylabel('Absolute error')

    for i, df in enumerate(dfs):
        xlabel = xlabels[i]
        x = df[xlabel]
        # empirical estimator
        y = df['Empirical avg']
        y_std = df['Empirical se']
        axes[i].plot(x, y, label='Monte Carlo', color=COLORS[0], marker='8')
        axes[i].fill_between(x, y-y_std, y+y_std, color=COLORS[0], alpha=0.3)
        # oracle bound
        y = df['Oracle bound avg']/const
        y_std = df['Oracle bound se']
        axes[i].plot(
            x, y, label='Oracle bound', color=COLORS[1],
            linestyle='-', marker='s')
        axes[i].fill_between(x, y-y_std, y+y_std, color=COLORS[1], alpha=0.3)
        # bound
        y = df['Bound avg']/const
        axes[i].plot(x, y, label='Bound', color=COLORS[2], linestyle='-.')
        axes[i].set_xlabel(xlabel)

        axes[i].set_title(titles[i])
        if log_scale:
            axes[i].set_yscale('log')
            if xlabel != 'Tail decay':
                axes[i].set_xscale('log')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    lgd = fig.legend(
        handles, labels, loc='lower center',
        bbox_to_anchor=(0.5, -0.14), ncol=3)

    if save:
        fig.savefig(
            fname, bbox_extra_artists=[lgd], bbox_inches='tight')
    else:
        plt.show()


def plot_dist_est(dfs, xlabels, titles, ylabel='Absolute error', fname=None, save=False, log_scale=True):
    linestyle = ['-', '--', '-.', (0, (1, 1)), '-']
    marker = ['8', 's', '', '', '^']
    est_name = ['Empirical', 'Braess-Sauer', 'Good-Turing', 'Krichevsky-Trofimov', 'Laplace']

    fig, axes = plt.subplots(nrows=1, ncols=4, sharey=True)
    fig.set_figheight(4)
    fig.set_figwidth(15)
    axes[0].set_ylabel(ylabel)

    for i, df in enumerate(dfs):
        xlabel = xlabels[i]
        for j, ename in enumerate(est_name):
            x = df[xlabel]
            y = df[f'{ename} avg']
            y_std = df[f'{ename} se']
            axes[i].plot(
                x, y, label=ename, color=COLORS[j],
                linestyle=linestyle[j], marker=marker[j])
            axes[i].fill_between(
                x, y-y_std, y+y_std, color=COLORS[j], alpha=0.3)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel(xlabel)
        if log_scale:
            axes[i].set_yscale('log')
            if xlabel != 'Tail decay':
                axes[i].set_xscale('log')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 1])  # L, B, R, T 
    lgd = fig.legend(
        handles, labels, loc='lower center',
        bbox_to_anchor=(0.5, -0.14), ncol=len(est_name))

    if save:
        fig.savefig(
            fname, bbox_extra_artists=[lgd], bbox_inches='tight')
    else:
        plt.show()


def plot_quant_error(dfs, xlabels, titles, fname=None, save=False):
    fig, axes = plt.subplots(nrows=1, ncols=4, sharey=True)
    fig.set_figheight(4)
    fig.set_figwidth(15)
    axes[0].set_ylabel('Absolute error')

    for i, df in enumerate(dfs):
        xlabel = xlabels[i]
        x = df[xlabel]
        # greedy quantization
        y = df['Greedy avg']
        y_std = df['Greedy se']
        axes[i].plot(x, y, label='Greedy', color=COLORS[0], marker='8')
        axes[i].fill_between(x, y-y_std, y+y_std, color=COLORS[1], alpha=0.3)
        # oracle quantization
        y = df['Oracle avg']
        y_std = df['Oracle se']
        axes[i].plot(x, y, label='Oracle', color=COLORS[1], linestyle='--', marker='s')
        axes[i].fill_between(x, y-y_std, y+y_std, color=COLORS[2], alpha=0.3)
        # uniform quantization
        y = df['Uniform avg']
        y_std = df['Uniform se']
        axes[i].plot(x, y, label='Uniform', color=COLORS[2], linestyle='-.')
        axes[i].fill_between(x, y-y_std, y+y_std, color=COLORS[0], alpha=0.3)

        axes[i].set_title(titles[i])
        axes[i].set_xlabel(xlabel)
        axes[i].set_yscale('log')
        axes[i].set_xscale('log')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    lgd = fig.legend(
        handles, labels, loc='lower center',
        bbox_to_anchor=(0.5, -0.14), ncol=3)

    if save:
        fig.savefig(
            fname, bbox_extra_artists=[lgd], bbox_inches='tight')
    else:
        plt.show()


def plot_quant_level(dfs, nrates, titles, xlabel='Sample size', ylabel='Absolute error', fname=None, save=False):
    linestyle = ['-', '--', '-.', (0, (1, 1)), '-']
    marker = ['8', 's', '', '', '^']

    fig, axes = plt.subplots(nrows=1, ncols=4, sharey=True)
    fig.set_figheight(4)
    fig.set_figwidth(15)
    axes[0].set_ylabel(ylabel)

    for i, df in enumerate(dfs):
        x = df[:, 0]
        for r in range(nrates):
            y = df[:, r+1]
            y_std = df[:, nrates+r+1]
            axes[i].plot(
                x, y, label=r'$r = $' + str(r+2), color=COLORS[r],
                linestyle=linestyle[r], marker=marker[r])
            axes[i].fill_between(
                x, y-y_std, y+y_std, color=COLORS[r], alpha=0.3)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel(xlabel)
        axes[i].set_yscale('log')
        axes[i].set_xscale('log')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 1])  # L, B, R, T 
    lgd = fig.legend(
        handles, labels, loc='lower center',
        bbox_to_anchor=(0.5, -0.14), ncol=nrates)

    if save:
        fig.savefig(
            fname, bbox_extra_artists=[lgd], bbox_inches='tight')
    else:
        plt.show()
