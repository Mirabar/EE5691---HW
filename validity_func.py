import numpy as np
from hw_utils import fuzzy_cov


def invariant_criterion(x, u, C):

    #invariant to the number of clusters

    y = np.argmax(u, axis=0)
    m = np.mean(x, axis=0)

    Sb = 0
    Sw = 0

    for i in range(C):
        xi = x[y == i]
        mi = np.mean(xi, axis=0)
        d = xi - mi
        Si = np.sum(d[:, :, None] @ d[:, None, :], axis=0) #scatter matrix
        Sw += Si #within cluster scatter matrix
        d = mi - m
        d = d[:, None]
        Sb += len(xi) * (d @ d.T) #between cluster scatter matrix

    V = np.trace(np.matmul(np.linalg.inv(Sw), Sb))

    return V


def hypervolume_criterion(x, p, u, C):

    #average cluster volume

    h = 0

    for i in range(C):
        h += np.linalg.det(fuzzy_cov(x, p[i], u[i])) ** 0.5

    return h / C


def partition_density(x, p, u, C):

    #density of the central members

    h = 0
    c = 0

    for i in range(C):
        Fi = fuzzy_cov(x, p[i], u[i])
        h += np.linalg.det(Fi) ** 0.5
        d = x - p[i]
        sv = (d @ np.linalg.inv(Fi)) * d #Mahanalobis distance between x and the cluster centers
        m_ind = np.argwhere((sv[:, 0] < 1) * (sv[:, 1] < 1)) #define central members
        c += np.sum(u[i][m_ind])

    V = c / h

    return V


def average_partition_density(x, p, u, C, method='APDC', alpha=0.5):

    #average patition density of the central/maximal members

    ad = 0
    for i in range(C):
        Fi = fuzzy_cov(x, p[i], u[i])
        h = np.linalg.det(Fi) ** 0.5
        if method == 'APDC':
            d = x - p[i]
            sv = (d @ np.linalg.inv(Fi)) * d
            m_ind = np.argwhere((sv[:, 0] < 1) * (sv[:, 1] < 1))
        elif method == 'APDM':
            m_ind = np.argwhere(u[i] > alpha)

        if not len(m_ind):
            c = 0
        else:
            c = np.sum(u[i][m_ind])

        ad += c / h

    V = ad / C

    return V


def normalized_partition_indexes(x, u, p, C, q):
    #normalized partition density

    d = 0
    for i in range(C):
        d += (u[i] ** q) * np.linalg.norm(x - p[i], axis=-1) ** 2

    V = C * np.sum(d)

    return V


def validity(x, u, p, C, method='trace', q=None):
    if method == 'inv':  # maximize
        V = -invariant_criterion(x, u, C)

    elif method == 'HV':  # minimize
        V = hypervolume_criterion(x, p, u, C)

    elif method == 'PD':  # maximize
        V = -partition_density(x, p, u, C)

    elif (method == 'APDM') or (method == 'APDC'):  # maximize
        V = -average_partition_density(x, p, u, C, method=method)

    elif method == 'NPI':  # minimize
        V = normalized_partition_indexes(x, u, p, C, q)

    return V
