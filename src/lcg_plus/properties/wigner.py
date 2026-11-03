# Copyright © 2025 Technical University of Denmark

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from numba import njit
hbar = 2

def compute_wigner_function(means, covs, weights, xvec, pvec):
    X, P = np.meshgrid(xvec, -pvec, sparse=True) #Use -pvec because of matplotlib.imshow y axis convention. Can cause issues if comparing with analytical Wigner functions..
            
    wigner = 0
    for i, weight in enumerate(weights):
    
        if X.shape == P.shape:
            arr = np.array([X - means[i, 0], P - means[i, 1]])
            arr = arr.squeeze()
            
        else:
            # need to specify dtype for creating an ndarray from ragged
            # sequences
            arr = np.array([X - means[i, 0], P - means[i, 1]], dtype=object)

        if len(covs) ==1:
            exp_arg = arr @ np.linalg.inv(covs[0]) @ arr
            prefactor = 1 / (np.sqrt(np.linalg.det(2 * np.pi * covs[0])))
        else: 
            exp_arg = arr @ np.linalg.inv(covs[i]) @ arr
            prefactor = 1 / (np.sqrt(np.linalg.det(2 * np.pi * covs[i])))

        wigner += (weight * prefactor) * np.exp(-0.5 * (exp_arg))
    return wigner

def compute_wigner_function_log(means, covs, log_weights, xvec, pvec):
    X, P = np.meshgrid(xvec, -pvec, sparse=False) #Use -p because of matplotlib.imshow y axis convention. Can cause issues if comparing with analytical Wigner functions..
            
    Q = np.array([X,P])
        
    arr = Q[np.newaxis,:] - means[:,:, np.newaxis,np.newaxis]
    arr=np.transpose(arr, [2,3,0,1])
    
    if len(covs) == 1:
        
        exp_arg = -0.5 * np.einsum("...j,...jk,...k", arr, np.linalg.inv(covs[0])[np.newaxis,np.newaxis,:,:], arr)
        prefactor = 1 / (np.sqrt(np.linalg.det(2 * np.pi * covs[0])))
    
    else:

        exp_arg = -0.5 * np.einsum("...j,...jk,...k", arr, np.linalg.inv(covs)[np.newaxis, np.newaxis,:, : ,:], arr)
        exp_arg -= np.log(np.sqrt(np.linalg.det(2 * np.pi * covs)))[np.newaxis,np.newaxis,:] #the prefactor is handled here
        prefactor = 1
    
    wigner_exp_arg = np.transpose(exp_arg, [2,0,1])
    logwig = logsumexp(log_weights[:,np.newaxis,np.newaxis] + wigner_exp_arg, axis = 0 )

    return prefactor*np.exp(logwig)

@njit 
def make_grid(xvec,pvec):
    r"""Returns two coordinate matrices `X` and `P` from coordinate vectors
    `xvec` and `pvec`
    """
    X = np.outer(xvec, np.ones_like(pvec))
    P = np.outer(np.ones_like(xvec), pvec)
    return X,P


def get_wigner_real(data, xvec, pvec):
    """Returns wigner function, but only for real means
    """
    means, covs, weights, norm = data
    X, P = make_grid(xvec,pvec)
    grid = np.empty(X.shape+(2,))
    grid[:, :,0] = X
    grid[:, :,0] = P
    W=0
    for i, mu in enumerate(means):
        if len(covs) == 1:
            mvn  = multivariate_normal(mu.real, covs[0], allow_singular=False) #Only likes real means
        else:
            mvn  = multivariate_normal(mu.real, covs[i], allow_singular=False) #Only likes real means
            
        W += weights[i]*mvn.pdf(grid)
    return W/norm

#Currently unused and slow
def Gauss(sigma, mu, xvec, pvec, MP = False):
    """Returns the Gaussian in phase space point (x,p), or on a grid
    To do: Rethink MP method
    """

    if len(pvec)==1:
        xi  = xvec
    else:
        X, P = make_grid(xvec,pvec)
        xi = np.empty((2,)+X.shape)
        xi[0,:, :] = X
        xi[1,:, :] = P

    sigma_inv = np.linalg.inv(sigma)

    delta = xi - mu[:,np.newaxis, np.newaxis]

    exparg = - 0.5 * np.einsum("j...,...j", delta, np.einsum("...jk,k...",  sigma_inv, delta))

    Norm = 1/np.sqrt(np.linalg.det(sigma*2*np.pi))
    
    return Norm * np.exp(exparg)  



def Gaussian(sigma, mu, xvec, pvec):
    
    X, P = make_grid(xvec,pvec)
    xi = np.empty((2,)+X.shape)
    xi[0,:, :] = X
    xi[1,:, :] = P
                                    
    delta = xi - mu[:,np.newaxis, np.newaxis]
    sigma_inv= np.linalg.inv(sigma)
    #exparg = -0.5 * (delta @ sigma_inv @ delta)
    exparg = - 0.5 * np.einsum("j...,...j", delta, np.einsum("...jk,k...",  sigma_inv, delta))
    Norm = 1/np.sqrt(np.linalg.det(sigma*2*np.pi))
    return Norm * np.exp(exparg)

