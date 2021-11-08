import numpy as np
from hw_utils import init_param, get_posterior, find_cluster
from validity_func import *


def MLE_classifier(x, C, sigma0=None, P_prior0=None, use_kmean=False, epsilon=1e-5):
    i = 1

    if (sigma0 is not None) and (P_prior0 is not None):

        mu0 = init_param(x, C, use_kmean=use_kmean)
        while True:

            P_posterior = get_posterior(x, mu0, sigma0, P_prior0, C)
            mu_new = np.array([np.sum(np.tile(P_posterior[i], (x.shape[-1], 1)).T * x, axis=0)
                               / np.sum(P_posterior[i]) for i in range(C)])

            if np.sum(abs(mu_new - mu0)) / (C * x.shape[-1]) < epsilon:
                break
            mu0 = mu_new
            i += 1

        return np.argmax(P_posterior, axis=0), mu0

    else:
        mu0, P_prior0, sigma0 = init_param(x, C, case2=True, use_kmean=use_kmean)
        while True:

            P_posterior = get_posterior(x, mu0, sigma0, P_prior0, C)
            P_prior_new = np.sum(P_posterior, axis=-1) / x.shape[0]
            mu_new = np.array([np.sum(np.tile(P_posterior[i], (x.shape[-1], 1)).T * x, axis=0)
                               / np.sum(P_posterior[i]) for i in range(C)])
            sigma_new = [
                np.sum(P_posterior[i][:, None, None] * ((x - mu_new[i])[:, :, None] @ (x - mu_new[i])[:, None, :]),
                       axis=0) / np.sum(P_posterior[i], axis=0) for i in range(C)]

            if np.sum(abs(mu_new - mu0)) / (C * x.shape[-1]) < epsilon:
                break
            mu0 = mu_new
            P_prior0 = P_prior_new
            P_posterior0 = P_posterior
            sigma0 = sigma_new
            i += 1

        return np.argmax(P_posterior0, axis=0), mu0, sigma0, P_prior0


def FKM(x, c, q, p, u, dist='euclidean', epsilon=1e-3, n_stds=10, max_iter=100, fuzzy_decay=False):
    '''
    :param x: data
    :param c: number of clusters
    :param q: scalar controlling the fuzzines of the clusters
    :param p: cluster centers
    :param u: membership matrix of shape number_of_clusters x number_of_samples
    :param dist: distance metric -  euclidean or exponential
    :param epsilon: scalar indicating minimal change required to continue iterating
    :param n_stds: number of standard deviations for final distance calculation
    :param max_iter: maximal number of iterations
    :param fuzzy_decay: boolean indicating if fuzziness parameter should decay
    :return: predicted cluster centers, membership matrix and fuzziness parameter
    '''
    j = 0
    change_old = np.inf

    if fuzzy_decay:
        a = q

    while j < max_iter:

        d = []
        for i in range(c):

            if len(p) == i:  # last iteration distances
                di = n_stds * np.trace(np.cov(x, rowvar=False)) * np.ones(di.shape)
                d.append(di ** (1 / (1 - q)))
                continue

            if (dist == 'euclidean'):
                di = np.linalg.norm(x - p[i], axis=-1) ** 2  # + 1e-10
            elif (dist == 'exponential'):
                Fi = fuzzy_cov(x, p[i], u[i])
                ai = np.mean(u[i])
                t = x - p[i]

                di = (np.linalg.det(Fi) ** 0.5) / ai * \
                     np.exp(((t @ np.linalg.inv(Fi))[:, None, :] @ t[:, :, None]) / 2).squeeze()

            d.append(di ** (1 / (1 - q)))

        d = np.array(d)
        new_u = d / np.sum(d, axis=0)

        p_old = p
        p = []
        all_u = np.sum(new_u ** q, axis=-1)

        for i in range(c):
            p.append(np.sum((new_u[i] ** q)[:, None] * x, axis=0) / all_u[i])

        change = np.mean(abs(new_u - u))
        if change < epsilon:
            break

        elif (dist == 'exponential'):
            F = np.array([np.linalg.det(fuzzy_cov(x, p[i], new_u[i])) for i in range(c)])
            if len(np.where(F <= 0)[0]):
                break
            else:
                u = new_u
                j += 1
                if fuzzy_decay and (a * np.exp(-j / 25) > 1) and (change_old - change >= 0.01):
                    q = a * np.exp(-j / 25)
                change_old = change
        else:
            u = new_u
            j += 1
            if fuzzy_decay and (a * np.exp(-j / 25) > 1) and ((change_old - change) >= 0.01):
                q = a * np.exp(-j / 25)
            change_old = change

    return p_old, u, q


def uofc(x, Cmax=8, q=2, method=None, fuzzy_decay=False):
    p = [np.mean(x, axis=0)]
    a = q

    u = [-np.inf] * len(x)
    Vinv = []
    Vhv = []
    Vpd = []
    Vapdc = []
    Vapdm = []
    Vnpi = []
    V = []
    for c in range(1, Cmax + 1):

        p, u, q = FKM(x, c, a, p, u, fuzzy_decay=False)
        p, u, q = FKM(x, c, q, p, u, dist='exponential', fuzzy_decay=fuzzy_decay)
        if method == 'all':
            Vinv.append(validity(x, u, p, c, method='inv', q=q))
            Vhv.append(validity(x, u, p, c, method='HV', q=q))
            Vpd.append(validity(x, u, p, c, method='PD', q=q))
            Vapdc.append(validity(x, u, p, c, method='APDC', q=q))
            Vapdm.append(validity(x, u, p, c, method='APDM', q=q))
            Vnpi.append(validity(x, u, p, c, method='NPI', q=q))
        elif method is not None:
            V.append(validity(x, u, p, c, method=method, q=q))
        u = np.concatenate([u, np.ones((1, len(x))) * -np.inf])

    if method == 'all':
        V = [Vinv, Vhv, Vpd, Vapdc, Vapdm, Vnpi]
    elif method is None:
        return u[:-1], p

    return V


def join_clusters(D, clusters, method='min', x=None):
    # agglomerative hierarchical clustering

    if method == 'min':
        # nearest-neighbor
        inds = np.unravel_index(np.argmin(D), D.shape)  # find 2D index in D using index from flat array
        Di, clusters = find_cluster(clusters, inds[0])
        Dj, clusters = find_cluster(clusters, inds[1])
        c = Di + Dj
        pairs = np.array(np.meshgrid(c, c)).T.reshape(-1, 2)  # get all pairs of cluster members
        D[pairs[:, 0], pairs[:, 1]] = np.inf  # ignore in next iteration

    elif method == 'max':
        # farthest neighbor
        inds = np.unravel_index(np.argmin(D), D.shape)
        Di, clusters = find_cluster(clusters, inds[0])
        Dj, clusters = find_cluster(clusters, inds[1])
        c = Di + Dj

        pairs = np.array(np.meshgrid(c, c)).T.reshape(-1, 2)
        tmp = D.copy()
        tmp[tmp == np.inf] = -np.inf
        max_d = np.max(tmp[c, :], axis=0)
        D[c, :] = max_d[None, :]
        D[:, c] = max_d[:, None]
        D[pairs[:, 0], pairs[:, 1]] = np.inf


    elif method == 'avg':
        # minimal average distance
        inds = np.unravel_index(np.argmin(D), D.shape)
        Di, clusters = find_cluster(clusters, inds[0])
        Dj, clusters = find_cluster(clusters, inds[1])
        c = Di + Dj
        D[Di, :] *= len(Di)
        D[Dj, :] *= len(Dj)
        tmp = D[c, :]
        avg = np.sum(tmp, axis=0) / (len(Di) * len(Dj))
        D[c, :] = avg[None, :]
        D[:, c] = avg[:, None]

    elif (method == 'mean') or (method == 'e'):
        # minimal distance between centers
        inds = np.unravel_index(np.argmin(D), D.shape)
        Di, clusters = find_cluster(clusters, inds[0])
        Dj, clusters = find_cluster(clusters, inds[1])
        c = Di + Dj

        cluster_mean = (x[Di[0], :] * len(Di) + x[Dj[0], :] * len(Dj)) / (len(c))
        x[c, :] = cluster_mean
        remove_ind = D == np.inf
        tmp = ((x - x[c, None]) ** 2).sum(axis=-1) ** 0.5
        D[c, :] = tmp
        D[:, c] = tmp.T
        D[remove_ind] = np.inf
        pairs = np.array(np.meshgrid(c, c)).T.reshape(-1, 2)
        D[pairs[:, 0], pairs[:, 1]] = np.inf
        if method == 'e':
            # minimize the sum of squared errors
            n = np.sum(D == np.inf, axis=0)
            n = n * len(c) / (n + len(c))
            n_vec = np.tile(np.sqrt(n), (len(c), 1))
            D[c, :] *= n_vec
            D[:, c] *= n_vec.T

    clusters.append(c)

    return clusters
