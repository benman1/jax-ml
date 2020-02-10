'''
Some metrics and distance functions.

Vectors are expected to be column vectors.
'''
from functools import partial
import jax.numpy as jnp


def get_mahalanobis(conv_inv):
    '''Given an inverse covariance matrix, return the mahalanobis
    distance function of signature (n),(n) -> ()
    '''
    def mahalanobis(x, y, VI):
        '''mahalanobis function
        '''
        diff = x - y
        a = jnp.dot(diff, VI)
        return jnp.sqrt(jnp.dot(a, diff.T))[0]

    return partial(mahalanobis, VI=conv_inv)


def euclidean(x, y):
    '''Euclidean distance function
    '''
    return jnp.linalg.norm(
        x - y, ord=2, axis=-1, keepdims=False
    )
