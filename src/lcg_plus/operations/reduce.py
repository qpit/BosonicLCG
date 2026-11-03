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
from lcg_plus.states.coherent import outer_coherent, gen_indices, gen_indices_full, gen_fock_superpos_coherent
from scipy.special import logsumexp, factorial


def find_unique_means_and_merge_weights(means, log_weights, sort=False):
    
    unique_means, idx, idx_inv  = np.unique(np.round(means,10),axis = 0, return_index = True, return_inverse=True)
    unique_means = means[idx] 

    unique_log_weights = np.zeros(len(idx), dtype='complex')
    
    #Compute the new weights with a logsumexp
    for i in range(len(idx)):
        lw = log_weights[idx_inv == i]
        if len(lw) != 1:
            unique_log_weights[i] = logsumexp(lw)
        else:
            unique_log_weights[i] = lw[0] #to scalar
        num_k = len(unique_log_weights)
        
    #Find number of real Gaussians, and sort them as [Re G, Im G]
    if sort: 
        
        reals = unique_means.imag == 0
        reals = reals[:,0]*reals[:,1]
        
        #Sort unique means, weights where real means go first
        means_re = unique_means[reals==True]
        weights_re = unique_log_weights[reals==True]
        means_imag = unique_means[reals==False]
        weights_imag = unique_log_weights[reals==False]

        unique_means = np.vstack((means_re, means_imag))
        unique_log_weights = np.hstack((weights_re, weights_imag))
        num_k = np.sum(reals == True)
    
    return unique_means, unique_log_weights, num_k

def mu_to_alphas(mu):
    """Compute the |alpha><beta| values from the mean
    Args:
        mu : means
    Returns:
        alpha : 
        beta : 
        exparg : log coefficient 
    """
    alpha = 0.5*(mu[:,0].real-mu[:,1].imag)+1j*0.5*(mu[:,0].imag+mu[:,1].real)
    beta = 0.5*(mu[:,0].real+mu[:,1].imag)+1j*0.5*(mu[:,1].real-mu[:,0].imag)
    exparg = -0.5*(mu[:,0].imag)**2 - 0.5*(mu[:,1].imag)**2 + 0.5*1j*(
        mu[:,1].real*mu[:,1].imag+mu[:,0].real*mu[:,0].imag)

    return alpha, beta, exparg

def reduce(nmax:int, eps:float, data:tuple):
    """Find the weights of the reduced state with stellar rank nmax. The number of new weights is of order (nmax+1)^2.
    

    Args: 
        nmax : max number of photons in reduced state
        eps : radius of target ring
        data : input state parameters
    Returns: 
        new_data : output state parameters
    """
    means, covs, weights, num_k, norm = data
    
    #Normalise weights
    #weights /= norm 
    
    k1 = nmax+1
    
    alpha, beta, d = mu_to_alphas(means)
    
    weights /= np.exp(d)
   
    weights *= np.exp(-0.5 * np.abs(alpha)**2 - 0.5 * np.abs(beta)**2)
    
    w1 = weights[0:num_k][np.newaxis,:]
    w2 = weights[num_k:][np.newaxis,:]/2 #Because we use 2 Re ( weights[num_k:] ) 
    a1 = alpha[0:num_k][np.newaxis,:]
    a2 = alpha[num_k:][np.newaxis,:]
    b2 = beta[num_k:][np.newaxis,:]
    
    ns, ms = gen_indices(nmax)
    n1 = ns[:,np.newaxis]
    m1 = ms[:,np.newaxis]
    n2 = ns[np.newaxis,:]
    m2 = ms[np.newaxis,:]

    #First compute kl sum
    akk = a1**n1 * np.conjugate(a1)**m1
    akl = a2**n1 * np.conjugate(b2)**m1
    alk = b2**n1 * np.conjugate(a2)**m1
 
    sumkk = np.sum(w1*akk,axis=1)
    sumkl = np.sum(w2*akl,axis=1)
    sumlk = np.sum(np.conjugate(w2)*alk,axis=1)
    tot = sumkk + sumkl + sumlk
    
    #Now compute the nm sum for all ij
    n11 = n1[0:k1,:]
    
    exparg1 = -2*np.pi*1j*n11*(n2-m2)/k1
    cnn = 1/eps**(2*n11) * np.exp(exparg1)*tot[0:k1,np.newaxis]

    exparg2 = -2*np.pi*1j*(n1[k1:,:]*n2 - m1[k1:,:]*m2)/k1
    exparg3 = -2*np.pi*1j*(m1[k1:,:]*n2 - n1[k1:,:]*m2)/k1
    
    cnm = 1/eps**(n1[k1:,:]+m1[k1:,:])*np.exp(exparg2)*tot[k1:,np.newaxis]
    cmn = 1/eps**(n1[k1:,:]+m1[k1:,:])*np.exp(exparg3)*np.conjugate(tot[k1:,np.newaxis])
    
    nw1 = np.sum(cnn,axis=0)
    nw2 = np.sum(cnm+cmn,axis=0)
    nw = nw1+nw2

    #Generate the new means, and multiply new weights by the coefficients
      
    theta = 2*np.pi /k1
    alpha = eps * np.exp(1j*theta*ns)
    beta = eps * np.exp(1j*theta*ms)
    means, cov, d = outer_coherent(alpha,beta)

    nw*=np.exp(d)
    nw[k1:]*=2

    #normalise the new weights
    #nw /= np.sum(nw.real)
    
    #return means.T, covs, nw, k1, norm
    return means.T, covs, np.log(nw), k1


def reduce_log(nmax:int, eps:float, data:tuple):
    """Find the weights of the reduced state with stellar rank nmax. The number of new weights is of order (nmax+1)^2.
    

    Args: 
        nmax : max number of photons in reduced state
        eps : radius of target ring
        data : input state parameters
    Returns: 
        new_data : output state parameters
    """
    #raise ValueError('Not tested with log_weights. Review.')
    means, covs, log_weights, num_k = data
    
    #Normalise weights
    #weights /= norm 
    
    k1 = nmax+1
    
    alpha, beta, exparg = mu_to_alphas(means)
    
    log_weights += -exparg 
   
    log_weights += -0.5 * np.abs(alpha)**2 - 0.5 * np.abs(beta)**2
    
    w1 = log_weights[0:num_k][np.newaxis,:]
    w2 = log_weights[num_k:][np.newaxis,:] - np.log(2) #Because we use 2 Re ( weights[num_k:] ) 
    a1 = alpha[0:num_k][np.newaxis,:]
    a2 = alpha[num_k:][np.newaxis,:]
    b2 = beta[num_k:][np.newaxis,:]
    
    ns, ms = gen_indices(nmax)
    n1 = ns[:,np.newaxis]
    m1 = ms[:,np.newaxis]
    n2 = ns[np.newaxis,:]
    m2 = ms[np.newaxis,:]

    #First compute kl sum
    akk = n1 * np.log(a1) + m1*np.log(np.conjugate(a1))
    #akk = a1**n1 * np.conjugate(a1)**m1
    akl = n1 * np.log(a2) + m1 * np.log(np.conjugate(b2))
    #akl = a2**n1 * np.conjugate(b2)**m1
    alk = n1 * np.log(b2) + m1 * np.log(np.conjugate(a2))
    #alk = b2**n1 * np.conjugate(a2)**m1

    #sumkk = logsumexp(w1 + akk, axis = 1)
    #sumkk = np.sum(np.exp(w1)*akk,axis=1)
    #sumkl = logsumexp(w2 + akl, axis = 1)
    #sumkl = np.sum(np.exp(w2)*akl,axis=1)
    #sumlk = logsumexp(np.conjugate(w2) + alk, axis = 1)
    #sumlk = np.sum(np.conjugate(np.exp(w2))*alk,axis=1)
    sumtot = np.concatenate((w1+akk, w2+akl, np.conjugate(w2)+alk), axis =1)
    
    tot = logsumexp(sumtot, axis = 1)
    
    #Now compute the nm sum for all ij
    n11 = n1[0:k1,:]
    
    exparg1 = -2*np.pi*1j*n11*(n2-m2)/k1
    cnn = exparg1 - 2*n11*np.log(eps) + tot[0:k1,np.newaxis]
    #cnn = 1/eps**(2*n11) * np.exp(exparg1)*tot[0:k1,np.newaxis]

    exparg2 = -2*np.pi*1j*(n1[k1:,:]*n2 - m1[k1:,:]*m2)/k1
    exparg3 = -2*np.pi*1j*(m1[k1:,:]*n2 - n1[k1:,:]*m2)/k1
    cnm = exparg2 - (n1[k1:,:]+m1[k1:,:])*np.log(eps) + tot[k1:,np.newaxis]
    #cnm = 1/eps**(n1[k1:,:]+m1[k1:,:])*np.exp(exparg2)*tot[k1:,np.newaxis]
    cmn = exparg3 - (n1[k1:,:]+m1[k1:,:])*np.log(eps) + np.conjugate(tot[k1:,np.newaxis])
    #cmn = 1/eps**(n1[k1:,:]+m1[k1:,:])*np.exp(exparg3)*np.conjugate(tot[k1:,np.newaxis])

    nw = logsumexp(np.concatenate((cnn, cnm, cmn), axis = 0),axis = 0)
    
    #Generate the new means, and multiply new weights by the coefficients
      
    theta = 2*np.pi /k1
    alpha = eps * np.exp(1j*theta*ns)
    beta = eps * np.exp(1j*theta*ms)
    means, cov, exparg = outer_coherent(alpha,beta)

    nw += exparg
    nw[k1:] += np.log(2)

    #normalise the new weights
    #norm = np.exp(logsumexp(nw)).real
    #nw /= np.sum(nw.real)
    
    #return means.T, covs, nw, k1, norm
    return means.T, covs, nw, k1


def reduce_log_full(nmax:int, eps:float, data:tuple):
    """Find the weights of the reduced state with stellar rank nmax. The number of new weights is of order (nmax+1)^2.

    Args: 
        nmax : max number of photons in reduced state
        eps : radius of target ring
        data : input state parameters
    Returns: 
        new_data : output state parameters
    """
    means, covs, log_weights, num_k = data
    
    
    k1 = nmax+1
    
    alpha, beta, exparg = mu_to_alphas(means)

    #Remove the terms where alpha.real + alpha.imag = 0, and also beta
    id1= np.where(alpha == 0)
    id2= np.where(beta == 0) 
    id3 = np.unique(np.concatenate((id1,id2)))
    #Remove those entries
    alpha = np.delete(alpha, id3)
    beta = np.delete(beta, id3)
    log_weights = np.delete(log_weights, id3)
    exparg = np.delete(exparg, id3)
    
    log_weights += -exparg 
   
    log_weights += -0.5 * np.abs(alpha)**2 - 0.5 * np.abs(beta)**2
    
    w  = log_weights[np.newaxis,:]
    a = alpha[np.newaxis,:]
    b = beta[np.newaxis,:]

    ns, ms = gen_indices_full(nmax)
    n1 = ns[:,np.newaxis]
    m1 = ms[:,np.newaxis]
    n2 = ns[np.newaxis,:]
    m2 = ms[np.newaxis,:]

    #First compute j sum
        
    aj = n1 * np.log(a) + m1*np.log(np.conjugate(b))

    tot = logsumexp(w + aj, axis = 1)
    
    #Now compute the nm sum for all j

    exparg1 = -2*np.pi*1j*(n1*n2 - m1*m2)/k1
    cnm = exparg1 - (n1+m1)*np.log(eps) + tot[:,np.newaxis]
    
    nw = logsumexp(cnm, axis = 0)
    
    #Generate the new means, and multiply new weights by the coefficients
      
    theta = 2*np.pi /k1
    alpha = eps * np.exp(1j*theta*ns)
    beta = eps * np.exp(1j*theta*ms)
    means, cov, exparg = outer_coherent(alpha,beta)

    nw += exparg
    
    return means.T, covs, nw, len(nw)


def reduce_log_pure(nmax:int, eps:float, data:tuple):
    """Find the weights of the reduced state with stellar rank nmax. The number of new weights is of order (nmax+1)^2.

    Args: 
        nmax : max number of photons in reduced state
        eps : radius of target ring
        data : input state parameters
    Returns: 
        new_data : output state parameters
    """
    #raise ValueError('Not tested with log_weights. Review.')
    means, covs, log_weights, num_k = data
    
    k1 = nmax+1
    
    alpha, beta, exparg = mu_to_alphas(means)
    
    log_weights += -exparg 
   
    log_weights += -0.5 * np.abs(alpha)**2 - 0.5 * np.abs(beta)**2

    ns = np.arange(k1)
    n1 = ns[:,np.newaxis]
    m1 = nmax

    if num_k == len(log_weights):
        fast = False
        w = log_weights[np.newaxis,:]
        a = alpha[np.newaxis,:]
        b = beta[np.newaxis,:]
    
        akl = n1 * np.log(a) + m1 * np.log(np.conjugate(b))
        sumtot = w + akl
    
    else:
        fast = True
        w1 = log_weights[0:num_k][np.newaxis,:]
        w2 = log_weights[num_k:][np.newaxis,:] - np.log(2) #Because we use 2 Re ( weights[num_k:] ) 
        a1 = alpha[0:num_k][np.newaxis,:]
        a2 = alpha[num_k:][np.newaxis,:]
        b2 = beta[num_k:][np.newaxis,:]

        #First compute kl sum
        akk = n1 * np.log(a1) + m1 * np.log(np.conjugate(a1)) 
        akl = n1 * np.log(a2) + m1 * np.log(np.conjugate(b2)) 
        alk = n1 * np.log(b2) + m1 * np.log(np.conjugate(a2))
        sumtot = np.concatenate((w1+akk, w2+akl, np.conjugate(w2)+alk), axis =1)
    
    rho = logsumexp(sumtot, axis = 1) - 0.5 * np.log(factorial(ns))
    coeffs = np.exp(rho) 
    
    return gen_fock_superpos_coherent(coeffs, 1e-3, eps = eps, fast = fast)