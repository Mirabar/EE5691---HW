import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import v_measure_score
from cluster_models import uofc

#unsupervised optimal fuzzy clustering

t = ['-invariant', 'HP', '-PD', '-APDC', '-APDM', 'NPI'] #validity criteria
Cmax = 7 #max number of clusters
fd = True
q = 2
for i in range(12):

    with open(f'hw_data/data{i + 1}.pkl', 'rb') as f:
        data = pickle.load(f)

    x = np.concatenate(data[0], axis=0)

    y = []
    for k in range(data[-1]):
        y = y + [k] * len(data[0][k])
    y = np.array(y)

    V = uofc(x, Cmax=Cmax, method='all', q=q, fuzzy_decay=fd)

    fig, ax = plt.subplots(2, 4, figsize=(25, 5))
    gs = ax[0, -1].get_gridspec()
    for axe in ax[:, -1]:
        axe.remove()
    axbig = fig.add_subplot(gs[:, -1])

    ax = ax.flatten()

    j = 0

    for a in ax:
        if a.axes is None:
            continue
        a.plot(range(1, Cmax + 1), V[j])
        a.set_title(t[j], fontsize=12)
        a.set_xlabel('Number of clusters', fontsize=8)
        a.set_xticks(range(1, Cmax + 1))

        a.tick_params(axis='both', which='major', labelsize=8)
        a.tick_params(axis='both', which='minor', labelsize=8)
        j += 1

    plt.tight_layout()
    plt.savefig(f'figs/validity{i + 1}_q2_fd.png')
    plt.close(fig)

    valid_clusters = np.argmin(V, axis=-1) + 1
    n_clusters, n_valid = np.unique(valid_clusters, return_counts=True)
    n_clusters = n_clusters[np.argmax(n_valid)]


    u, p = uofc(x, Cmax=n_clusters, q=q, fuzzy_decay=fd)
    p = np.array(p)

    y_pred = np.argmax(u, axis=0)


    fig, ax = plt.subplots(1, 1)
    ax.scatter(x[:, 0], x[:, 1], c=y_pred, s=5, cmap='tab10', alpha=0.3)
    ax.scatter(p[:, 0], p[:, 1], c='black', marker='X')

    if len(data[0]) > data[-1]:
        l_noise = len(data[0][-1])
        y = y[:-l_noise]
        y_pred = y_pred[:-l_noise]

    v = v_measure_score(y, y_pred)

    ax.set_title('Dataset {} - Validity - nmi = {:.2f}'.format(i + 1, v))
    plt.tight_layout()
    plt.savefig(f'figs/uofc{i + 1}_q2_fd.png')
    plt.close(fig)

    u, p = uofc(x, Cmax=data[-1], q=q, fuzzy_decay=fd)
    p = np.array(p)

    y_pred = np.argmax(u, axis=0)

    # fig, ax = plt.subplots(1, 1)
    axbig.scatter(x[:, 0], x[:, 1], c=y_pred, s=5, cmap='tab10', alpha=0.3)
    axbig.scatter(p[:, 0], p[:, 1], c='black', marker='X')

    if len(y_pred) > len(y):
        y_pred = y_pred[:len(y)]

    v = v_measure_score(y, y_pred)

    axbig.set_title('Dataset {} - True clusters - nmi = {:.2f}'.format(i + 1, v))
    plt.tight_layout()
    plt.savefig(f'figs/uofc{i + 1}_q2_fd.png')
    plt.close(fig)
