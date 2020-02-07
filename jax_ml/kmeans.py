'''Kmeans in jax/numpy
'''
import random
from sklearn.base import ClassifierMixin
from tqdm.notebook import tqdm
import jax.numpy as jnp
from jax import jit, vmap
from .metrics import euclidean


class KMeans(ClassifierMixin):
    '''
    Kmeans implementation in jax/numpy.

    - Kmeans++-like initialization
    - Allows defining custom distance metrics
    '''
    def __init__(self, k, n_iter=100, dist_fun=euclidean):
        '''
        Parameters:
        -----------
        k : the number of clusters
        n_iter : the number of iterations [100]
        dist_fun : the distance function; should be a function
            dist(x: vec, y: vec)-> float with two vectors, x
            and y, returning a scalar.
        '''
        self.k = k
        self.n_iter = n_iter
        self.dist_fun = jit(
            vmap(
                dist_fun, in_axes=(0, None), out_axes=0
            )
        )

    def adjust_centers(self, X):
        '''Adjust centers given cluster assignments
        '''
        jnp.row_stack([
            X[self.clusters == c].mean(axis=0)
            for c in self.clusters
        ])

    @staticmethod
    def __get_center(X, weights=None):
        '''Randomly draw a new center from X
        '''
        return jnp.array(
            random.choices(X, weights=weights, k=1)[0],
            ndmin=2
        )

    def initialize_centers(self, X):
        '''Roughly the kmeans++ initialization
        '''
        self.centers = self.__get_center(X)
        for c in range(1, self.k):
            print(c)
            weights = self.dist_fun(X, self.centers)
            if c > 1:
                weights = jnp.mean(weights, axis=-1)
            print(weights.shape)
            new_center = self.__get_center(X, weights)
            self.centers = jnp.row_stack(
                (self.centers, new_center)
            )

    def fit(self, X, y=None):
        self.initialize_centers(X)
        for _ in tqdm(range(self.n_iter)):
            dists = self.dist_fun(X, self.centers)
            self.clusters = jnp.argmin(dists, axis=-1)
            self.adjust_centers(X)
        return self.clusters
