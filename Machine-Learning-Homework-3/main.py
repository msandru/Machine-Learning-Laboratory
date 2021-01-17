from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_samples = 1500
random_state = 160
X, y = make_blobs(n_samples=n_samples, centers=6, random_state=random_state)
d = pd.DataFrame(X, columns=['X1', 'X2'])

j = list()
for i in range(1, 15):
    km = KMeans(n_clusters=i, random_state=1)
    clusters = km.fit_predict(d)

    x_centroids = km.cluster_centers_[:, 0]
    y_centroids = km.cluster_centers_[:, 1]

    plt.scatter(d['X1'], d['X2'], c=clusters)
    plt.scatter(x_centroids, y_centroids, s=100, marker='x',
                c=range(len(x_centroids)))

    #plt.show()
    j.append(km.inertia_)

range = range(1, 1 + len(j))
fig, ax = plt.subplots(figsize=(10, 5))
plt.plot(range, j, label='Inertia')
plt.legend()
plt.xlabel("k")
plt.ylabel("J (inertia)")
plt.show()
