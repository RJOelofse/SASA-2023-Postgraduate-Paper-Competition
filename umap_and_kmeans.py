import pandas as pd
import numpy as np
import umap
import umap.plot
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.utils import resample

mnist = fetch_openml('mnist_784')
mnist_subsample, mnist_subsample_labels = resample(mnist.data,
                                                   mnist.target,
                                                   n_samples=10000,
                                                   stratify=mnist.target,
                                                   random_state=1)

fmnist = fetch_openml('Fashion-MNIST')
fmnist_subsample, fmnist_subsample_labels = resample(fmnist.data,
                                                     fmnist.target,
                                                     n_samples=10000,
                                                     stratify=fmnist.target,
                                                     random_state=1)

mapper_mnist = umap.UMAP(n_neighbors=15,
                   n_components=2,
                   metric="euclidean",
                   min_dist=0).fit(mnist_subsample)
umap.plot.points(mapper_mnist, labels=mnist_subsample_labels)

mapper_fmnist = umap.UMAP(n_neighbors=15,
                         n_components=2,
                         metric="euclidean",
                         min_dist=0).fit(fmnist_subsample)
umap.plot.points(mapper_fmnist, labels=fmnist_subsample_labels)

#%%
kmeans_mnist = KMeans(n_clusters=10,
                      init="k-means++",
                      n_init="auto",
                      random_state=2023).fit(mnist_subsample)

print("kmeans mnist")
adjusted_rand = metrics.adjusted_rand_score(mnist_subsample_labels, kmeans.labels_)
print(adjusted_rand)
adjusted_mutual_info = metrics.adjusted_mutual_info_score(mnist_subsample_labels, kmeans.labels_)
print(adjusted_mutual_info)

kmeans_fmnist = KMeans(n_clusters=10,
                       init="k-means++",
                       n_init="auto",
                       random_state=2023).fit(fmnist_subsample)

print("kmeans fmnist")
adjusted_rand = metrics.adjusted_rand_score(fmnist_subsample_labels, kmeans_fmnist.labels_)
print(adjusted_rand)
adjusted_mutual_info = metrics.adjusted_mutual_info_score(fmnist_subsample_labels, kmeans_fmnist.labels_)
print(adjusted_mutual_info)

kmeans_umap_mnist = KMeans(n_clusters=10,
                           init="k-means++",
                           n_init="auto",
                           random_state=2023).fit(mapper_mnist.embedding_)

print("kmeans + umap mnist")
adjusted_rand = metrics.adjusted_rand_score(mnist_subsample_labels, kmeans_umap_mnist.labels_)
print(adjusted_rand)
adjusted_mutual_info = metrics.adjusted_mutual_info_score(mnist_subsample_labels, kmeans_umap_mnist.labels_)
print(adjusted_mutual_info)

kmeans_umap_fmnist = KMeans(n_clusters=10,
                            init="k-means++",
                            n_init="auto",
                            random_state=2023).fit(mapper_fmnist.embedding_)

print("kmeans + umap fmnist")
adjusted_rand = metrics.adjusted_rand_score(fmnist_subsample_labels, kmeans_umap_fmnist.labels_)
print(adjusted_rand)
adjusted_mutual_info = metrics.adjusted_mutual_info_score(fmnist_subsample_labels, kmeans_umap_fmnist.labels_)
print(adjusted_mutual_info)