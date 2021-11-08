import pickle
import numpy as np
from scipy import spatial
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import v_measure_score
from cluster_models import join_clusters


for k in range(12):

    with open(f'hw_data/data{k + 1}.pkl', 'rb') as f:
        data = pickle.load(f)

    x = np.concatenate(data[0], axis=0)
    x2 = x.copy()

    n_clusters = data[-1]

    y = []
    for k in range(data[-1]):
        y = y + [k] * len(data[0][k])
    y = np.array(y)


    colors = cm.tab10(np.linspace(0, 1, n_clusters))
    fig, ax = plt.subplots(1, 5, figsize=(30, 5))

    for j, m in enumerate(['min', 'max', 'avg', 'mean', 'e']):
        i = x.shape[0]
        D = spatial.distance.squareform(spatial.distance.pdist(x))
        D[range(D.shape[0]), range(D.shape[0])] = np.inf
        clusters = [(i,) for i in range(len(x))]

        if m == 'e':
            D /= np.sqrt(2)

        while i > n_clusters:
            # merge iteratively
            clusters = join_clusters(D, clusters, m, x2)

            i -= 1

        y_pred = np.ones(len(x))*np.inf
        cn = 0
        for i, c in zip(clusters, colors):
            ax[j].scatter(x[i, 0], x[i, 1], c=c, s=5, cmap='tab10', alpha=0.3)
            y_pred[list(i)] = cn
            cn += 1

        if len(y_pred) > len(y):
            y_pred = y_pred[:len(y)]

        v = v_measure_score(y, y_pred)

        ax[j].set_title('Dist {}, Dataset {} - Validity - nmi = {:.2f}'.format(m, k + 1, v), fontsize=8)
        ax[j].tick_params(axis='both', which='major', labelsize=8)
        ax[j].tick_params(axis='both', which='minor', labelsize=8)
    plt.tight_layout()
    plt.savefig(f'figs/hi_{k + 1}.png')
    plt.close(fig)

a = 1
