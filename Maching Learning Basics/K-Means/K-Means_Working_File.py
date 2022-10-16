import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
data = scale(digits.data) # data are the features and we're scaling them down to between -1 and 1 to save computation time
y = digits.target

# k = len(np.unique(y))   # dynamic way to get k but we don't need this since we're just using 10
k = 10
samples, features = data.shape # define how many samples/features we have by getting the data set shape


def bench_k_means(estimator, name, data):
    """This was taken from the sklearn website and allows us to train a bunch of different classifiers and score them
    calling this function"""
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


clf = KMeans(n_clusters=k, init="random", n_init=10) # classifier, number of clusters = 10, place centroids, number of times we algorithm runs w/ different centroid seeds
bench_k_means(clf, "1", data)

