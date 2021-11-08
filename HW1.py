import numpy as np
import pickle
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.metrics import v_measure_score
from cluster_models import MLE_classifier
from hw_utils import get_posterior

###case1 - unknown mean

fig, ax = plt.subplots(3, 3)
ax = ax.flatten()
use_kmean = True
n_iter = 1

for d in range(8):

    with open(f'hw_data/data{d + 1}.pkl', 'rb') as f:
        data = pickle.load(f)

    x = np.concatenate(data[0], axis=0)
    y = []
    for i in range(len(data[0])):
        y = y + [i] * len(data[0][i])

    y = np.array(y)

    c, uc = np.unique(y, return_counts=True)
    C = data[-1]
    P_prior = uc / len(y)
    best_l = np.array([-np.inf] * C)
    best_mu = np.array([[-np.inf] * x.shape[-1]] * C)
    j = 0

    while j < n_iter:
        _, mu = MLE_classifier(x, C, sigma0=data[2], P_prior0=P_prior, use_kmean=use_kmean)

        l = np.array([np.sum(np.log(stats.multivariate_normal.pdf(x=x, mean=mu[i], cov=data[2][i])))
                      for i in range(C)])

        best_ind = np.where(l > best_l)[0]
        best_l[best_ind] = l[best_ind]
        best_mu[best_ind] = mu[best_ind]
        j += 1

    P_posterior = get_posterior(x, best_mu, data[2], P_prior, C)
    y_pred = np.argmax(P_posterior, axis=0)

    ax[d].scatter(x[:, 0], x[:, 1], c=y_pred, s=5, cmap='tab10', alpha=0.3)
    ax[d].scatter(best_mu[:, 0], best_mu[:, 1], c='black', marker='X')

    if len(data[0]) > C:
        l_noise = len(data[0][-1])
        y = y[:-l_noise]
        y_pred = y_pred[:-l_noise]

    v = v_measure_score(y, y_pred)
    ax[d].set_title('Dataset {} - nmi = {:.2f}'.format(d+1, v), fontsize=7)

    ax[d].tick_params(axis='both', which='major', labelsize=6)
    ax[d].tick_params(axis='both', which='minor', labelsize=6)

fig.delaxes(ax[-1])

###case2 - unknown mean, variance and priors

fig, ax = plt.subplots(4, 3)
ax = ax.flatten()
use_kmean = False


for d in range(10, 12):

    with open(f'hw_data/data{d + 1}.pkl', 'rb') as f:
        data = pickle.load(f)

    x = np.concatenate(data[0], axis=0)
    y = []
    for i in range(len(data[0])):
        y = y + [i] * len(data[0][i])

    y = np.array(y)

    c, uc = np.unique(y, return_counts=True)
    C = data[-1]

    y_pred, mu, sigma, P_prior = MLE_classifier(x, C, use_kmean=use_kmean)
    ax[d].scatter(x[:, 0], x[:, 1], c=y_pred, s=5, cmap='tab10', alpha=0.3)
    ax[d].scatter(mu[:, 0], mu[:, 1], c='black', marker='X')

    if len(data[0]) > C:
        l_noise = len(data[0][-1])
        y = y[:-l_noise]
        y_pred = y_pred[:-l_noise]

    v = v_measure_score(y, y_pred)
    ax[d].set_title('Dataset {} - nmi = {:.2f}'.format(d+1, v), fontsize=7)


    ax[d].tick_params(axis='both', which='major', labelsize=6)
    ax[d].tick_params(axis='both', which='minor', labelsize=6)
