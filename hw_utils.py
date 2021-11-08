from scipy import stats
import numpy as np
from sklearn.cluster import KMeans
import random

def init_param(x, C, case2=False, use_kmean=False):
    if use_kmean:
        mu = KMeans(n_clusters=C).fit(x).cluster_centers_
    else:
        mu_ind = random.sample(range(len(x)), C)
        mu = x[mu_ind]

    if case2:
        P = [1 / C] * C
        sigma = [np.eye(x.shape[-1])] * C

        return mu, P, sigma

    return mu


def get_posterior(x, mu, sigma, P_prior, C):
    p_likelihood = [stats.multivariate_normal.pdf(x=x, mean=mu[i], cov=sigma[i]) for i in range(C)]
    P_evidence = np.sum(np.array([p_likelihood[i] * P_prior[i] for i in range(C)]), axis=0)
    P_posterior = np.array([p_likelihood[i] * P_prior[i] / P_evidence for i in range(C)])

    return P_posterior

def fuzzy_cov(x, p, u):
    return np.sum(u[:, None, None] * ((p - x)[:, :, None] @ (p - x)[:, None, :]), axis=0) / np.sum(u)

def find_cluster(clusters, val):
    for i, c in enumerate(clusters):
        if val in c:
            clusters.pop(i)
            return c, clusters
