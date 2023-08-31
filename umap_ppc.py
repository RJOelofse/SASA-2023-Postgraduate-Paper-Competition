import importlib
import time
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', FutureWarning)

import umappc
# importlib.reload(umappc)
import umappc.plot
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.utils import resample

print('Downloading mnist dataset')
mnist = fetch_openml('mnist_784')
mnist_subsample, mnist_subsample_labels = resample(mnist.data,
                                                   mnist.target,
                                                   n_samples=10000,
                                                   stratify=mnist.target,
                                                   random_state=1)
print('Downloaded mnist dataset')

print('Downloading fashion-mnist dataset')
fmnist = fetch_openml('Fashion-MNIST')
fmnist_subsample, fmnist_subsample_labels = resample(fmnist.data,
                                                     fmnist.target,
                                                     n_samples=10000,
                                                     stratify=fmnist.target,
                                                     random_state=1)
print('Downloaded fashion-mnist dataset')

#%%
n_neighbors_vals = np.array([5,15,60,120])
lagrange_vals = np.array([0.15,0.25,0.5,1,1.5])
repeat = 5

adjusted_rand_mean = np.zeros((n_neighbors_vals.shape[0],lagrange_vals.shape[0]))
adjusted_rand_sd = np.zeros((n_neighbors_vals.shape[0],lagrange_vals.shape[0]))

adjusted_mutual_info_mean = np.zeros((n_neighbors_vals.shape[0],lagrange_vals.shape[0]))
adjusted_mutual_sd = np.zeros((n_neighbors_vals.shape[0],lagrange_vals.shape[0]))

computation_time_mean = np.zeros((n_neighbors_vals.shape[0],lagrange_vals.shape[0]))
computation_time_sd = np.zeros((n_neighbors_vals.shape[0],lagrange_vals.shape[0]))

#%%
print('Started grid search of hyperparameters for MNIST')
for i in range(n_neighbors_vals.shape[0]):
    for j in range(lagrange_vals.shape[0]):
        adjusted_rand = np.zeros(repeat)
        adjusted_mutual_info = np.zeros(repeat)
        computation_time = np.zeros(repeat)
        for k in range(repeat):
            print('n_neighbors: ' + str(n_neighbors_vals[i]) + ', lagrange: ' + str(lagrange_vals[j]) + ', repetition: ' + str(k))
            # Get the start time
            start_time = time.time()
            mapper = umappc.UMAP(n_neighbors=n_neighbors_vals[i],
                                 n_components=2,
                                 metric="euclidean",
                                 min_dist=0,
                                 # random_state=2023,
                                 verbose=False,
                                 clustering=True,
                                 n_clusters=10,
                                 lagrange=lagrange_vals[j]).fit(mnist_subsample)

            adjusted_rand[k] = metrics.adjusted_rand_score(mnist_subsample_labels, mapper.cluster_labels)
            adjusted_mutual_info[k] = metrics.adjusted_mutual_info_score(mnist_subsample_labels, mapper.cluster_labels)

            # get the end time
            end_time = time.time()
            elapsed_time = end_time - start_time
            final_time = elapsed_time / 60

            computation_time[k] = final_time

        adjusted_rand_mean[i,j] = np.mean(adjusted_rand)
        adjusted_rand_sd[i,j] = np.std(adjusted_rand)

        adjusted_mutual_info_mean[i,j] = np.mean(adjusted_mutual_info)
        adjusted_mutual_sd[i,j] = np.std(adjusted_mutual_info)

        computation_time_mean[i,j] = np.mean(computation_time)
        computation_time_sd[i,j] = np.std(computation_time)

print('Completed grid search of hyperparameters for MNIST')

#%%
colindex = lagrange_vals
rowindex = n_neighbors_vals
print('adjusted_rand_mean')
print(tabulate(adjusted_rand_mean, headers=colindex, showindex=rowindex))
print('')
print('adjusted_rand_sd')
print(tabulate(adjusted_rand_sd, headers=colindex, showindex=rowindex))
print('')
print('adjusted_mutual_info_mean')
print(tabulate(adjusted_mutual_info_mean, headers=colindex, showindex=rowindex))
print('')
print('adjusted_mutual_sd')
print(tabulate(adjusted_mutual_sd, headers=colindex, showindex=rowindex))
print('')
print('computation_time_mean')
print(tabulate(computation_time_mean, headers=colindex, showindex=rowindex))
print('')
print('computation_time_sd')
print(tabulate(computation_time_sd, headers=colindex, showindex=rowindex))

file1 = open("umap_ppc_performance_mnist.txt","w")
file1.write('adjusted_rand_mean \n')
file1.write(tabulate(adjusted_rand_mean, headers=colindex, showindex=rowindex,tablefmt='latex')+'\n')
file1.write('adjusted_rand_sd \n')
file1.write(tabulate(adjusted_rand_sd, headers=colindex, showindex=rowindex,tablefmt='latex')+'\n')
file1.write('adjusted_mutual_info_mean \n')
file1.write(tabulate(adjusted_mutual_info_mean, headers=colindex, showindex=rowindex,tablefmt='latex')+'\n')
file1.write('adjusted_mutual_sd \n')
file1.write(tabulate(adjusted_mutual_sd, headers=colindex, showindex=rowindex,tablefmt='latex')+'\n')
file1.write('computation_time_mean \n')
file1.write(tabulate(computation_time_mean, headers=colindex, showindex=rowindex,tablefmt='latex')+'\n')
file1.write('computation_time_sd \n')
file1.write(tabulate(computation_time_sd, headers=colindex, showindex=rowindex,tablefmt='latex')+'\n')
file1.close() #to change file access modes

#%%
# Use the optimal hyperparameters
mapper_mnist_optimal = umappc.UMAP(n_neighbors=15,
                     n_components=2,
                     metric="euclidean",
                     min_dist=0,
                     verbose=False,
                     clustering=True,
                     n_clusters=10,
                     lagrange=0.25).fit(mnist_subsample)
# True Cluster Labels
umappc.plot.points(mapper_mnist_optimal, labels=mnist_subsample_labels)
# Assigned Cluster Labels
umappc.plot.points(mapper_mnist_optimal, labels=mapper_mnist_optimal.cluster_labels)

print(metrics.adjusted_rand_score(mnist_subsample_labels, mapper_mnist_optimal.cluster_labels))
print(metrics.mutual_info_score(mnist_subsample_labels, mapper_mnist_optimal.cluster_labels))
print(metrics.adjusted_mutual_info_score(mnist_subsample_labels, mapper_mnist_optimal.cluster_labels))

    #%%
adjusted_rand_mean = np.zeros((n_neighbors_vals.shape[0],lagrange_vals.shape[0]))
adjusted_rand_sd = np.zeros((n_neighbors_vals.shape[0],lagrange_vals.shape[0]))

adjusted_mutual_info_mean = np.zeros((n_neighbors_vals.shape[0],lagrange_vals.shape[0]))
adjusted_mutual_sd = np.zeros((n_neighbors_vals.shape[0],lagrange_vals.shape[0]))

computation_time_mean = np.zeros((n_neighbors_vals.shape[0],lagrange_vals.shape[0]))
computation_time_sd = np.zeros((n_neighbors_vals.shape[0],lagrange_vals.shape[0]))

print('Started grid search of hyperparameters for Fashion-MNIST')
for i in range(n_neighbors_vals.shape[0]):
    for j in range(lagrange_vals.shape[0]):
        adjusted_rand = np.zeros(repeat)
        adjusted_mutual_info = np.zeros(repeat)
        computation_time = np.zeros(repeat)
        for k in range(repeat):
            print('n_neighbors: ' + str(n_neighbors_vals[i]) + ', lagrange: ' + str(lagrange_vals[j]) + ', repetition: ' + str(k))
            # Get the start time
            start_time = time.time()
            mapper = umappc.UMAP(n_neighbors=n_neighbors_vals[i],
                                 n_components=2,
                                 metric="euclidean",
                                 min_dist=0,
                                 # random_state=2023,
                                 verbose=False,
                                 clustering=True,
                                 n_clusters=10,
                                 lagrange=lagrange_vals[j]).fit(fmnist_subsample)

            adjusted_rand[k] = metrics.adjusted_rand_score(fmnist_subsample_labels, mapper.cluster_labels)
            adjusted_mutual_info[k] = metrics.adjusted_mutual_info_score(fmnist_subsample_labels, mapper.cluster_labels)

            # get the end time
            end_time = time.time()
            elapsed_time = end_time - start_time
            final_time = elapsed_time / 60

            computation_time[k] = final_time

        adjusted_rand_mean[i,j] = np.mean(adjusted_rand)
        adjusted_rand_sd[i,j] = np.std(adjusted_rand)

        adjusted_mutual_info_mean[i,j] = np.mean(adjusted_mutual_info)
        adjusted_mutual_sd[i,j] = np.std(adjusted_mutual_info)

        computation_time_mean[i,j] = np.mean(computation_time)
        computation_time_sd[i,j] = np.std(computation_time)

print('Completed grid search of hyperparameters for Fashion-MNIST')
#%%
colindex = lagrange_vals
rowindex = n_neighbors_vals
print('adjusted_rand_mean')
print(tabulate(adjusted_rand_mean, headers=colindex, showindex=rowindex))
print('')
print('adjusted_rand_sd')
print(tabulate(adjusted_rand_sd, headers=colindex, showindex=rowindex))
print('')
print('adjusted_mutual_info_mean')
print(tabulate(adjusted_mutual_info_mean, headers=colindex, showindex=rowindex))
print('')
print('adjusted_mutual_sd')
print(tabulate(adjusted_mutual_sd, headers=colindex, showindex=rowindex))
print('')
print('computation_time_mean')
print(tabulate(computation_time_mean, headers=colindex, showindex=rowindex))
print('')
print('computation_time_sd')
print(tabulate(computation_time_sd, headers=colindex, showindex=rowindex))

file1 = open("umap_ppc_performance_fmnist.txt","w")
file1.write('adjusted_rand_mean \n')
file1.write(tabulate(adjusted_rand_mean, headers=colindex, showindex=rowindex,tablefmt='latex')+'\n')
file1.write('adjusted_rand_sd \n')
file1.write(tabulate(adjusted_rand_sd, headers=colindex, showindex=rowindex,tablefmt='latex')+'\n')
file1.write('adjusted_mutual_info_mean \n')
file1.write(tabulate(adjusted_mutual_info_mean, headers=colindex, showindex=rowindex,tablefmt='latex')+'\n')
file1.write('adjusted_mutual_sd \n')
file1.write(tabulate(adjusted_mutual_sd, headers=colindex, showindex=rowindex,tablefmt='latex')+'\n')
file1.write('computation_time_mean \n')
file1.write(tabulate(computation_time_mean, headers=colindex, showindex=rowindex,tablefmt='latex')+'\n')
file1.write('computation_time_sd \n')
file1.write(tabulate(computation_time_sd, headers=colindex, showindex=rowindex,tablefmt='latex')+'\n')
file1.close() #to change file access modes

#%%
# Use the optimal hyperparameters
mapper_fmnist_optimal = umappc.UMAP(n_neighbors=15,
                                   n_components=2,
                                   metric="euclidean",
                                   min_dist=0,
                                   verbose=False,
                                   clustering=True,
                                   n_clusters=10,
                                   lagrange=0.25).fit(fmnist_subsample)

# True Cluster Labels
umappc.plot.points(mapper_fmnist_optimal, labels=fmnist_subsample_labels)
# Assigned Cluster Labels
umappc.plot.points(mapper_fmnist_optimal, labels=mapper_fmnist_optimal.cluster_labels)

print(metrics.adjusted_rand_score(fmnist_subsample_labels, mapper_fmnist_optimal.cluster_labels))
print(metrics.mutual_info_score(fmnist_subsample_labels, mapper_fmnist_optimal.cluster_labels))
print(metrics.adjusted_mutual_info_score(fmnist_subsample_labels, mapper_fmnist_optimal.cluster_labels))

