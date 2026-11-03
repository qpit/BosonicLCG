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


# COHERENT STATE SIMULATION MODULE
# ----------------------------------
# Simulation tools for approximating Fock state superpositions as
# superpositions of coherent states from Marshall & Anand http://arxiv.org/abs/2305.17099.

# The Wigner function of the superposition of coherent states is a weighted sum of Gaussians.

import numpy as np
from scipy.special import  factorial, comb, logsumexp

def gen_indices(nmax:int):
    """Generate the upper triangular n,m index list, with diagonal indices in front
    """
    k1 = nmax+1
    ns, ms = np.triu_indices(k1,1)
    ns = np.concatenate([range(k1),ns])
    ms = np.concatenate([range(k1),ms])
    return ns, ms

def gen_indices_full(nmax:int):
    k1 = nmax+1
    k1 = nmax+1
    ns, ms = np.triu_indices(k1,1)
    ks, ls = np.tril_indices(k1,-1)
    ns = np.concatenate([range(k1),ns, ks])
    ms = np.concatenate([range(k1),ms, ls])
    return ns, ms

def outer_coherent(alpha, beta, hbar =2):
    """ Returns the coefficient, displacement vector and covariance matrix (vacuum) of the Gaussian that
    describes the Wigner function of the outer product of two coherent states |alpha><beta| derived 
    in Appendix A of https://arxiv.org/abs/2103.05530.
    """
    cov = hbar/2 * np.eye(2)
    re_alpha = alpha.real
    im_alpha = alpha.imag
    re_beta = beta.real
    im_beta = beta.imag

    mu = np.sqrt(hbar/2) * np.array([re_alpha + re_beta 
                                        + 1j *(im_alpha - im_beta), 
                                        im_alpha + im_beta
                                        + 1j * (re_beta - re_alpha)])
    log_coeff = -0.5 * (im_alpha - im_beta) **2- 0.5 * (re_alpha - re_beta) **2- 1j * im_beta * re_alpha + 1j * im_alpha * re_beta

    return mu, cov, log_coeff

def eps_fock_coherent(N, inf):
    """Returns the amplitude $\\eps = |\\alpha|$ of the coherent states giving the desired fidelity  
    to the N photon number state in the approximation - Eq? In M&A.
    """
    return (factorial(2*N+1)/(factorial(N)) * inf)**(1/(2*(N+1)))


def gen_fock_coherent(N, infid, eps = None, norm = True, fast = True, hbar =2):
    """Generate the Bosonic state data for a Fock state N in the coherent state representation.
    
    Args:
        N (int): fock number
        infid (float): infidelity of approximation
        eps (float): coherent state amplitude, takes presidence over infid if specified.

    See Eq 28 of http://arxiv.org/abs/2305.17099 for expansion into coherent states.

    Returns: 
        means, covs, weights, k
    """
    
    theta = 2*np.pi/(N+1)
    if not eps:
        eps = eps_fock_coherent(N, infid)

    if fast:
        ns, ms = gen_indices(N)
    else:
        ns, ms = gen_indices_full(N)

    alphas = eps * np.exp(1j*theta*ns)
    betas = eps * np.exp(1j*theta*ms)
    
    means, cov, d = outer_coherent(alphas,betas)
    log_weights = d - 1j*theta*N*(ns-ms)
    num_k = len(log_weights)

    if fast:
        log_weights[N+1:] += np.log(2) #For real parts
        num_k = N+1
    
    if norm:
        log_norm = logsumexp(log_weights)
        norm = np.exp(log_norm).real #Extract the real part. Should be real anyway if not fast
        log_weights -= np.log(norm)
    
    return means.T, np.array([cov]), log_weights, num_k
    

def eps_superpos_coherent(N, inf):
    """Returns the magnitude of the coherent states giving for the desired 
    infidelity of the Fock superposition up to photon number N.
    """
    return (factorial(N+1)*inf)**(1/(2*(N+1)))

def gen_fock_superpos_coherent(coeffs, infid, eps = None, norm = True, fast =True):
    """Returns the weights, means and covariance matrix of the state |psi> = c0 |0> + c1 |1> + c2 |2> + ... + c_max |n_max>
    in the coherent-fock representation.

    Args:
        coeffs (list/array):  the coefficients in front of the number states, coeff = [c0, c1, c2, ... c_nmax] 
        infid (float): infidelity of approx
    Returns: 
        means (ndarray): list of means
        cov (array): vacuum cov
        log_weights (ndarray): list of weights 
        k (int) : nmax + 1
    """
    #raise ValueError('Not fixed for log weights form (log(-1) = 1j*np.pi)')

    N = len(coeffs)-1
    if not eps:
        eps = eps_superpos_coherent(N, infid)

    theta = 2*np.pi /(N+1) 
        
    ns = np.array(range(N+1))[:,np.newaxis]
    ks = np.array(range(N+1))[np.newaxis,:]

    bn = np.log(np.sqrt(factorial(ns)) / eps** ns * coeffs[:,np.newaxis])
    ckn = bn - 1j*ks*ns*theta
    
    #ckn  = np.sqrt(factorial(ns)) / eps** ns * coeffs[:,np.newaxis] * np.exp(-1j*ks*ns*theta)
    ck = logsumexp(ckn, axis =0)
    #ck = np.sum(ckn, axis = 0)

    if fast:
        ns, ms = gen_indices(N)
    else:
        ns, ms = gen_indices_full(N)
   
    
    alphas = eps * np.exp(1j * theta * ns)
    betas = eps * np.exp(1j * theta * ms)
    
    means, cov, log_weights = outer_coherent(alphas,betas)

    num_k = len(log_weights)
    
    if fast:
        log_weights[N+1:] += np.log(2) #For the real parts
        num_k = N+1
        
    log_weights += ck[ns] + np.conjugate(ck[ms])
    #log_weights[0:N+1] += 2*ck
    #log_weights[N+1:] += ck[ns[N+1:]] + np.conjugate(ck[ms[N+1:]])
    
    if norm:
        log_norm = logsumexp(log_weights)
        norm = np.exp(log_norm).real #Extract the real part. Should be real anyway if not fast
        log_weights -= np.log(norm)
        
    return means.T, cov, log_weights, num_k

def norm_coherent(N, eps):
    """REVISE
    """
    norm = 1 + eps**(2*(N+1))/factorial(N+1)
    return norm
    
def order_infidelity_fock_coherent(N, alpha):
    """give infidelity of N fock approximation using given alpha - Eq 28 of M&A
    """
    return factorial(N)/factorial(2*N+1)*alpha**(2*(N+1))

#Old generating functions
def gen_fock_coherent_old(N, infid, eps = None, fast = False, norm = True, hbar =2):
    """Generate the Bosonic state data for a Fock state N in the coherent state representation.
    
    Args:
        N (int): fock number
        infid (float): infidelity of approximation
        eps (float): coherent state amplitude, takes presidence over infid if specified.
        fast (bool): whether to invoke the fast representation, which uses N(N+1)/2 Gaussians

    See Eq 28 of http://arxiv.org/abs/2305.17099 for expansion into coherent states.

    Returns: 
        means, covs, weights, +(k if fast == True)
    """
    
    cov = 0.5*hbar * np.eye(2)
    means = []
    theta = 2*np.pi/(N+1)
    log_weights = []
    if not eps:
        eps = eps_fock_coherent(N, infid)

    if fast: 
        log_weights_re = []
        means_re = []
    
    for k in np.arange(N+1):

        alpha = eps * np.exp(1j * theta * k) 

        if fast:
            muk, cov, ck = outer_coherent(alpha, alpha)
        
            means.append(muk)
            log_weights.append(ck)

        for l in np.arange(N+1):
            if fast:
                if l>k: 
                    beta = eps * np.exp(1j * theta * l)
                    mukl, cov, ckl = outer_coherent(alpha, beta)
                    ckl += -theta * 1j* N*(k-l)
                    means_re.append(mukl)
                    log_weights_re.append(ckl+np.log(2))
            else:
                beta = eps * np.exp(1j * theta * l)
                mukl, cov, ckl = outer_coherent(alpha, beta)
                ckl += -theta * 1j* N*(k-l)
                means.append(mukl)
                log_weights.append(ckl)
                
    if N == 0:
        means = []
        means.append(np.array([0,0]))
        
    if fast:
        k = len(weights)
        log_weights = np.concatenate([log_weights, log_weights_re], axis = 0)
        means = np.concatenate([means, means_re], axis = 0)

    else:
        log_weights = np.array(log_weights)
        means = np.array(means)
    
    #factor = factorial(N)/(N+1)**2 * np.exp(eps**2)/eps**(2*N)

    #weights += np.log(factor)
    

    if fast:
         #   Norm = np.sum(np.exp(log_weights).real)
            
        return means, cov, log_weights, k
    else:
        if norm:
         #   Norm = np.sum(np.exp(log_weights))
            log_weights -= logsumexp(log_weights)
        return means, cov, log_weights, len(log_weights)
        
def gen_fock_superpos_coherent_old(coeffs, infid, eps = None, fast = False):
    """Returns the weights, means and covariance matrix of the state |psi> = c0 |0> + c1 |1> + c2 |2> + ... + c_max |n_max>
    in the coherent-fock representation.

    Args:
        coeff (list/array):  the coefficients in front of the number states, coeff = [c0, c1, c2, ... c_nmax] 
        infid (float): infidelity of approx
    Returns: 
        means_new (ndarray): list of means
        cov (array): vacuum cov
        weights_new (ndarray): list of weights 
    """
    raise ValueError('Not fixed for log weights form (log(-1) = 1j*np.pi)')
    def get_ck(k, N, coeffs, eps):
        """Get coefficient ck in Eq (22) in http://arxiv.org/abs/2305.17099 for a superposition of Fock states
    
        Args:
            k (int) : coefficient number
            N (int) : max Fock number in superposition
            coeffs (list/array) 
            eps (float) 
        Returns:
            ck (complex) : k'th coefficient
        """
        ck = 0
        for n, an in enumerate(coeffs):
            ck += np.exp(eps**2 / 2)/(N+1) * np.sqrt(factorial(n)*1.0) / eps**n * an * np.exp(-2*np.pi * 1j * n * k /(N+1))
            
        return ck
    
    log_weights = []
    means = []

    if fast: 
        log_weights_re = []
        means_re = []

    N = len(coeffs)-1
    
    ck = np.zeros(N+1, dtype = 'complex128')
    if not eps:
        eps = eps_superpos_coherent(N, infid)
    
    #Obtain new coefficients
    for i in np.arange(N+1):
        ck[i] = get_ck(i, N, coeffs, eps)
        
    theta = 2*np.pi /(N+1) 

    for i, cn in enumerate(ck):
        alpha = eps * np.exp(1j * theta * i)
        if fast: 
            mui, cov, ci = outer_coherent(alpha, alpha)
            means.append(mui)
            log_weights.append(np.log(np.abs(cn)**2)+ci)
        
        for j, cm in enumerate(ck):
            if fast:
                if j > i:
                    cm = cm.conjugate()
                    beta = eps * np.exp(1j * theta * j) 
                    muij, cov, cij = outer_coherent(alpha, beta)
        
                    log_weights_re.append(np.log(2*cn*cm)+cij)
                    means_re.append(muij)
            else:
                cm = cm.conjugate()
                beta = eps * np.exp(1j * theta * j) 
                muij, cov, cij = outer_coherent(alpha, beta)
    
                log_weights.append(np.log(cn*cm)+cij)
                means.append(muij)
    if fast:
        k = len(log_weights)
        log_weights = np.concatenate([log_weights, log_weights_re], axis = 0)
        means = np.concatenate([means, means_re], axis = 0)
        Norm = np.sum(np.exp(log_weights).real)
        log_weights -= np.log(Norm)
        return means, cov, log_weights, k, np.sum(np.exp(log_weights).real)
    else:
        Norm = np.sum(np.exp(log_weights))
        log_weights -= np.log(Norm)
        return np.array(means), cov, np.array(log_weights), len(log_weights)

# |N><M| operator
# ---------------------------------------

def fock_outer_coherent(N, M, eps1, eps2, hbar =2):
    """Return |N><M| operator in bosonic representation
    WORK IN PROGRESS
    OBS: Purity error
    
    Obtain the weights, means and covaraince matrix in the coherent-fock representation of the |N><M| state
    
    for M > N, calculate |M><N| and take complex conjugate 

    Args: 
        N (int)
        M (int)
        eps1 (float): Displacement of |N> approx
        eps2 (float): Displacement of |M> approx

    Returns: 
        weights_new (ndarray): list of weights
        means_new (ndarray): list of means
        cov (array): vacuum cov
    
    """
    comp = 0
    if M > N:
        #Compute |M><N| and take complex conjugate at the end
        N, M = M, N
        eps1, eps2 = eps2, eps1
        comp = 1
        
    cov = hbar /2 * np.eye(2)
    means = []
    theta_N = 2*np.pi/(N+1)
    theta_M = 2*np.pi/(M+1)
    log_weights = []

    
    for k in np.arange(N+1):
        Re_k = eps1*np.cos(theta_N * k)
        Im_k = eps1*np.sin(theta_N * k)


        for l in np.arange(M+1):
            
            Re_l = eps2*np.cos(theta_M * l)
            Im_l = eps2*np.sin(theta_M * l)
            
            mulk = np.sqrt(hbar/2) * np.array([Re_k + Re_l + 1j*(Im_k - Im_l), Im_k + Im_l + 1j*(Re_l - Re_k)])
            means.append(mulk)
            

            clk = -theta_N * 1j* N*k +theta_M *1j* l* M + -0.5* (Im_k - Im_l)**2 - 0.5 *(Re_k - Re_l)**2-1j*Im_l*Re_k + 1j*Im_k*Re_l 
            log_weights.append(clk)
            
    log_weights = np.array(log_weights)

    #Here do a small simplifcation of the factorial in order to be able to compute the sqrt
    K = N - M
    if K != N:
        factorial_simplify = np.sqrt(factorial(N)/factorial(N-K-1))*factorial(M)
    else:
        factorial_simplify = np.sqrt(factorial(N)*factorial(M)/1.0)
        
    #factor = np.sqrt(factorial(N)*factorial(M))/((N+1)*(M+1)) * np.exp(eps1**2/2+eps2**2/2)/(eps1**N * eps2**M)
    
    #factor = factorial_simplify/((N+1)*(M+1)) * np.exp(eps1**2/2+eps2**2/2)/(eps1**N * eps2**M)

    #weights *= factor
    means = np.array(means)
    log_weights = np.array(weights)
    
    if comp:
        means = means.conjugate()
        weights = weights.conjugate()

    ##Quick and dirty solution to fix purity issue - Generate norm of |N> and |M> and divide by sqrts of norms
    norm1 = norm_coherent(N, eps1)
    norm2 = norm_coherent(M, eps2)
    #print(f"{N,M}", 1-1/norm1, 1-1/norm2, eps1, eps2)

    Norm = np.sum(np.exp(log_weights))
    
    log_weights -= np.log(Norm)

    
    return means, cov, log_weights

def outer_sqz_coherent(r, alpha, beta, hbar =2):
    """ Returns the coefficient, displacement vector and covariance matrix (vacuum) of the Gaussian that
    describes the Wigner function of the outer product of two coherent states |alpha><beta| derived 
    in Appendix A of https://arxiv.org/abs/2103.05530.
    r>0: Squeezing in x
    r<0: Squeezing in p
    """

    cov = hbar /2 * np.array([[np.exp(-2*r),0],[0,np.exp(2*r)]])
    gamma = alpha/(np.cosh(r)+np.sinh(r))
    delta = beta/(np.cosh(r)+np.sinh(r))
    
    re_gamma = gamma.real
    im_gamma = gamma.imag
    re_delta = delta.real
    im_delta = delta.imag
    
    mu = np.sqrt(hbar/2) * np.array([re_gamma + re_delta
                                        + 1j *np.exp(-2*r)*(im_gamma - im_delta), 
                                        im_gamma + im_delta
                                        + 1j * np.exp(2*r)* (re_delta - re_gamma)])

   

    log_coeff =  -0.5 * np.exp(-2*r)* (im_gamma - im_delta) **2- 0.5 * np.exp(2*r)*(re_gamma - re_delta) **2- 1j * im_delta * re_gamma + 1j * im_gamma * re_delta

    return mu, cov, log_coeff

def gen_sqz_cat_coherent(r, alpha, k, fast = False):
    """Prepare a squeezed cat

    Args: 
        r : squeezing of the cat
        alpha: displacement of the cat (pre-squeezing)
        k : parity
    Returns:
        tuple
    """
    if fast: 
        params = [(1.0, alpha,alpha), (1.0,-alpha,-alpha), (2.0*(-1)**k,alpha,-alpha)]
    else:
        params = [(1, alpha,alpha), (1,-alpha,-alpha), ((-1)**k,alpha,-alpha), ((-1)**k,-alpha,alpha)]
        
    means = []
    log_weights = []

    for a in params:
        means_a, cov, log_weights_a = outer_sqz_coherent(r, a[1], a[2])
        means.append(means_a)
        log = log_weights_a + np.log(np.abs(a[0]))
        if np.sign(a[0]) == -1:
            log += 1j*np.pi #For negative coefficients
        
        log_weights.append(log)

    means = np.array(means)
    log_weights = np.array(log_weights)
    
    if fast:
      
        return means, cov, log_weights, 2
    else:
        return means, cov, log_weights, len(log_weights)
    


def mu_to_alphas(mu):
    alpha = 0.5*(mu[0].real-mu[1].imag)+1j*0.5*(mu[0].imag+mu[1].real)
    beta = 0.5*(mu[0].real+mu[1].imag)+1j*0.5*(mu[1].real-mu[0].imag)
    exparg = -0.5*(mu[0].imag)**2 - 0.5*(mu[1].imag)**2 + 0.5*1j*(mu[1].real*mu[1].imag+mu[0].real*mu[0].imag)
    
    d = np.exp(exparg)

    #mu2, cov, coeff = outer_coherent(alpha,beta)
    #print(np.allclose(mu2,mu))
    #print(np.allclose(coeff,d))
    
    return alpha, beta, d


def get_cnm(n,m, data):
    c=0
    
    means, covs, weights, k = data
    
    for i in range(len(weights)):
        alpha, beta, d = mu_to_alphas(means[i])
        exparg = -0.5*np.abs(alpha)**2-0.5*np.abs(beta)**2
        if i < k:
            c += weights[i]*np.exp(exparg)* alpha** n * np.conjugate(beta)**m /d
        else:
            c += 0.5 * weights[i]*np.exp(exparg)* alpha** n * np.conjugate(beta)**m /d
            c += 0.5 * np.conjugate(weights[i])*np.exp(exparg)* np.conjugate(alpha)**m * beta**n /np.conjugate(d)
        
    return  c/np.sqrt(factorial(n)*factorial(m))