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
    - Allows specifying other measures for average; Arithmetic mean
        by default: jax.numpy.mean; use jax.numpy.median for kmedians
        clustering.

    '''
    def __init__(self, k, n_iter=100, dist_fun=euclidean, mean=jnp.mean):
        '''
        Parameters:
        -----------
        k : the number of clusters
        n_iter : the number of iterations [100]
        dist_fun : the distance function; should be a function
            dist(x: vec, y: vec)-> float with two vectors, x
            and y, returning a scalar.
        mean : average measure [jax.numpy.mean]

        Attributes:
        -----------
        k : parameter k for the number of clusters
        n_iter : number of iterations
        dist_fun : distance function used
        _mean : average function used
        clusters : crisp cluster labels for each point
        centers : centers
        inertia_ : distance of each point to their respective center
        '''
        self.k = k
        self.n_iter = n_iter
        self.dist_fun = vmap(
                jit(dist_fun), in_axes=(0, None), out_axes=0
        )
        self._mean = mean

    def adjust_centers(self, X):
        '''Adjust centers given cluster assignments
        '''
        jnp.row_stack([
            self._mean(X[self.clusters == c], axis=0)
            for c in self.clusters
        ])

    @staticmethod
    def _get_center(X, weights=None):
        '''Randomly draw a new center from X
        '''
        return jnp.array(
            random.choices(X, weights=weights, k=1)[0],
            ndmin=2
        )

    def initialize_centers(self, X):
        '''Roughly the kmeans++ initialization
        '''
        self.centers = self._get_center(X)
        for c in range(1, self.k):
            weights = self.dist_fun(X, self.centers)
            if c > 1:
                # harmonic mean gives error
                weights = jnp.mean(weights, axis=-1)
            new_center = self._get_center(X, weights)
            self.centers = jnp.row_stack(
                (self.centers, new_center)
            )

    def fit(self, X, y=None):
        self.initialize_centers(X)
        for _ in tqdm(range(self.n_iter)):
            self.inertia_ = self.dist_fun(X, self.centers)
            self.clusters = jnp.argmin(self.inertia_, axis=-1)
            self.adjust_centers(X)
        return self.clusters
