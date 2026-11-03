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
from scipy.special import comb, logsumexp
from lcg_plus.states.coherent import gen_fock_coherent
from lcg_plus.helper.from_sf import chop_in_blocks_multi, chop_in_blocks_vector_multi, chop_in_blocks_multi_v2, chop_in_blocks_vector_multi_v2
from lcg_plus.helper.matrices import xxpp_to_xpxp, xpxp_to_xxpp

def transform_data_due_to_measurement(state_data : tuple, modes : list, povm_data : tuple, hbar = 2.0):
    """Compute the new set of displacement vectors, covariance matrices and log_weights after post-selecting 
    some of the modes in the state stored in state_data on a measurement result whose povm element can be written as a 
    linear combination of gaussians, the data of which is stored in povm_data.

    To DO: Add dimension compatibility checks
    To DO: assertions that covariance matrices are real, and there is only one in the red-lcg case

    Args:
        state_data (tuple): [means, covs, log_weights, num_k] data of the state being measured
        modes (list): modes being measured
        povm_data (tuple): [means, covs, log_weights, num_k] data of povm element
        hbar (float): 
    Returns
        tuple: data of post-selected state computed via Eq. 26 in https://arxiv.org/abs/2508.06175
    """

    means, covs, log_weights, k1 = state_data

    #Partition the system into measured (B) and non-measured (A) subsystem
    mode_ind = np.concatenate((2 * np.array(modes), 2 * np.array(modes) + 1))
        
    V_A, V_AB, V_B = chop_in_blocks_multi(covs, mode_ind)
    r_A, r_B = chop_in_blocks_vector_multi(means, mode_ind)

    num_modes_A = r_A.shape[1] // 2 #number of modes in A
    num_modes_B = r_B.shape[1] // 2 #number of modes in B
    num_covs = covs.shape[0]
    num_weights = len(log_weights)

    means_povm, covs_povm, log_weights_povm, k2 = povm_data
    num_covs_povm = covs_povm.shape[0]

    num_weights_povm = len(log_weights_povm)

    if k2 != num_weights_povm and covs_povm.shape[0] != 1:
        raise ValueError("red-LCOG rep is not yet compatible with several covariance matrices.")

    # Use broadcasting to add arrays of different dimensions
    C = V_B[:,np.newaxis,:,:] + covs_povm[np.newaxis,:,:,:]
    C_inv = np.linalg.inv(C)

    V_AB_C_inv = np.einsum("...jk,...kl", V_AB[:,np.newaxis,:,:], C_inv)

    #Compute the new covariance matrices, Eq. 27
    new_covs = V_A[:,np.newaxis,:,:] - np.einsum("...jk,...kl", V_AB_C_inv, V_AB.transpose((0,2,1))[:,np.newaxis,:,:])
    
    delta_B = means_povm[np.newaxis,:,:] - r_B[:,np.newaxis,:]

    #Compute the new displacment vector, Eq. 28
    new_means = r_A[:,np.newaxis,:] + np.einsum("...jk,...k", V_AB_C_inv, delta_B) 
    
    #Compute the new log_weights, Eq. 29
    gaussian_norm = 0.5 * np.log(np.linalg.det(2*np.pi*C))
    prefactor = num_modes_B * np.log(2*np.pi*hbar)
    reweights_exp_arg = -0.5 * np.einsum("...j,...jk,...k", delta_B, C_inv, delta_B) - gaussian_norm + prefactor
    new_log_weights = log_weights_povm[np.newaxis,:] + log_weights[:,np.newaxis] + reweights_exp_arg 
    
    if k2 == num_weights_povm:
        new_means = new_means.reshape([num_weights * num_weights_povm,2*num_modes_A])
        new_log_weights = new_log_weights.reshape([num_weights * num_weights_povm])
        if num_covs_povm != 1 and num_covs != 1:
            new_covs = new_covs.reshape([num_weights * num_weights_povm, 2*num_modes_A, 2*num_modes_A])
        else:
            new_covs = new_covs.reshape([num_covs * num_covs_povm, 2*num_modes_A, 2*num_modes_A])

        #The output is automatically ordered with the first k1*n Gaussians being the real ones with our choice of broadcasting
    
    else: 
        #If there are special Gaussians, i.e. one part of a pair of complex conjugates,
        #see appendix: the reduced linear combination of Gaussians formalism, 
        delta_B_conj = means_povm[np.newaxis,k2:] - np.conjugate(r_B[k1:,np.newaxis,:])
        new_means_conj = np.conjugate(r_A[k1:,np.newaxis,:]) + np.einsum("...jk,...k", V_AB_C_inv, delta_B_conj)
        gaussian_norm_conj = 0.5 * np.log(np.linalg.det(2*np.pi*C))
        reweights_exp_arg_conj = -0.5*np.einsum("...j,...jk,...k", delta_B_conj, C_inv[np.newaxis,:,:], delta_B_conj) - gaussian_norm_conj + prefactor
        new_log_weights_conj = log_weights_povm[np.newaxis,k2:] + np.conjugate(log_weights[k1:,np.newaxis]) + reweights_exp_arg_conj

        #Order the new data so that the k1 * k2 regular gaussians are listed first
        weight_list = [
            new_log_weights[0:k1, 0:k2],
            new_log_weights[0:k1, k2:],
            new_log_weights[k1:, 0:k2],
            new_log_weights[k1:, k2:] - np.log(2),
            new_log_weights_conj - np.log(2)
            ]
        new_log_weights = np.concatenate([w.ravel() for w in weight_list])
        mean_list = [
            new_means[0:k1, 0:k2,:],
            new_means[0:k1,k2:,:],
            new_means[k1:,0:k2,:],
            new_means[k1:,k2:,:],
            new_means_conj
            ]
        new_means = np.concatenate([m.reshape(-1,m.shape[-1]) for m in mean_list], axis = 0)

        new_covs = new_covs.reshape([1, 2*num_modes_A, 2*num_modes_A])
        
    return new_means, new_covs, new_log_weights, k1*k2

def transform_data_due_to_homodyne_measurement(state_data : tuple, modes : list, phases : list, results : list):
    """Compute the new set of displacement vectors, covariance matrices and log_weights after post-selecting 
    some of the modes in the state stored in state_data on a homodyne measurement result.

    Args: 
        state_data (tuple): [means, covs, weights, num_k] of measured state
        modes (list): modes being measured
        phases (list): list of angles of the local oscillator
        results (list): list of measurement results
    
    Returns: 
        tuple: data of post-selected state
    """
    means, covs, log_weights, num_k = state_data

    mode_ind = np.concatenate((2 * np.array(modes), 2 * np.array(modes) + 1))
    
    V_A, V_AB, V_B = chop_in_blocks_multi(covs, mode_ind)
    r_A, r_B = chop_in_blocks_vector_multi(means, mode_ind)
    c = np.diag(np.cos(phases))
    s = np.diag(np.sin(phases))
    R = np.block([[c, s],[-s,c]]) #Symplectic matrix of the rotation unitary (with opposite phase)

    num_modes_A = r_A.shape[1] // 2 #number of modes in A
    num_modes_B = r_B.shape[1] // 2 #number of modes in B
    num_weights = len(log_weights)
    num_covs = covs.shape[0]

    means_povm = np.pad(results, (0,num_modes_B)) #xxpp disp vector

    #Convert to xxpp and apply rotation on B
    V_A = xpxp_to_xxpp(V_A)
    V_B = R @ xpxp_to_xxpp(V_B) @ R.T 
    V_AB = xpxp_to_xxpp(V_AB) @ R.T 
    r_A = xpxp_to_xxpp(r_A, is_vector = True)
    r_B = np.einsum("...jk,...k",R, xpxp_to_xxpp(r_B, is_vector = True))

    delta_B = means_povm[np.newaxis,:] - r_B
    delta_B_x = delta_B[:, :num_modes_B] #x component

    V_B_xx = V_B[:,:num_modes_B,:num_modes_B] #xx entries
    V_B_xx_inv = np.linalg.inv(V_B_xx)
    V_B_xx_inv_block = np.zeros((num_covs,2*num_modes_B,2*num_modes_B), dtype = V_B_xx.dtype)
    V_B_xx_inv_block[:, :num_modes_B, :num_modes_B] = V_B_xx_inv
    V_AB_block = V_AB @ V_B_xx_inv_block

    #Compute transformed data and switch ordering back to xpxp
    new_covs = xxpp_to_xpxp(V_A - V_AB_block @ V_AB.transpose([0,2,1]))
    new_means = xxpp_to_xpxp(r_A + np.einsum("...kl,...l",V_AB_block, delta_B), is_vector=True)
    reweights_exp_arg = -0.5 * np.einsum("...j,...jk,...k", delta_B_x, V_B_xx_inv, delta_B_x)
    gaussian_norm = -0.5 * np.log(np.linalg.det(2*np.pi*V_B_xx))
    new_log_weights = log_weights + reweights_exp_arg + gaussian_norm

    return new_means, new_covs, new_log_weights, num_k

def transform_gradient_data_due_to_measurement(state_data : tuple, gradient_data : tuple, modes : list, povm_data : tuple, hbar = 2.0):
    """Compute the new set of the partial derivatives of the displacement vectors, covariance matrices and log weights after the
    partial measurement of the state according to Eq. G4-G6. 

    Args: 
        state_data (tuple): [means, covs, log_weights, num_k] of the state
        gradient_data (tuple): [means_grad, covs_grad, log_weights_grad] of the state
        modes (list): list of measured modes
        povm_data (tuple): [means_grad, covs_grad, log_weights_grad, num_k] of the povm element
        hbar (float):
    Returns
        tuple: with the new gradient data
    """
    #Unpack
    means, covs, log_weights, k1 = state_data
    means_grad, covs_grad, log_weights_grad = gradient_data
    means_povm, covs_povm, log_weights_povm, k2 = povm_data

    if k1 != len(log_weights) or k2 != len(log_weights_povm):
        raise ValueError("Gradient transform is only compatible with the full LCOG formalism.")
    
    num_gradients = covs_grad.shape[0] #Number of gradients

    #Partition state data and gradient data into A and B blocks
    mode_ind = np.concatenate((2 * np.array(modes), 2 * np.array(modes) + 1))
    V_A, V_AB, V_B = chop_in_blocks_multi(covs, mode_ind)
    r_A, r_B = chop_in_blocks_vector_multi(means, mode_ind)

    num_modes_A = r_A.shape[1] // 2 #number of modes in A
    num_modes_B = r_B.shape[1] // 2 #number of modes in B
    
    grad_V_A, grad_V_AB, grad_V_B = chop_in_blocks_multi_v2(covs_grad, mode_ind)
    grad_r_A, grad_r_B = chop_in_blocks_vector_multi_v2(means_grad, mode_ind)
    
    # Use broadcasting to manipulate numpy arrays of different shapes
    C = V_B[:,np.newaxis,:,:] + covs_povm[np.newaxis,:,:,:]
    C_inv = np.linalg.inv(C)

    V_AB_C_inv = np.einsum("...jk,...kl", V_AB[:,np.newaxis,:,:], C_inv)

    delta_B = means_povm[np.newaxis,:,:] - r_B[:,np.newaxis,:]

    grad_V_B_inv = -C_inv @ grad_V_B @ C_inv

    grad_V_B_inv_delta_B = np.einsum("...jk,...k", grad_V_B_inv[:,np.newaxis,:,:], delta_B[np.newaxis,:,:])
    
    C_inv_delta_B = np.einsum("...jk,...k", C_inv, delta_B)

    #Update the partial derivative of the displacement vectors
    grad_V_AB = np.einsum("...jk,...k", grad_V_AB[:,np.newaxis,:,:], C_inv_delta_B[np.newaxis,:,:])
    grad_C_inv = np.einsum("...jk,...k", V_AB, grad_V_B_inv_delta_B)
    grad_r_B = np.einsum("...jk,...k", V_AB_C_inv, grad_r_B)

    new_means_grad = grad_r_A[:,:,np.newaxis,:] + grad_V_AB + grad_C_inv - grad_r_B[:,:,np.newaxis,:]
    new_means_grad = new_means_grad.reshape([num_gradients, k1*k2, 2 * num_modes_A])
    

    #Update the partial derivative of the covariance matrix
    new_covs_grad = grad_V_A + (- grad_V_AB @ V_AB_C_inv.transpose([0,2,1])[np.newaxis,:,:,:]
                                                - V_AB[np.newaxis,:,:,:] @ grad_V_B_inv @ V_AB.transpose([0,2,1])[np.newaxis,:,:,:]
                                                - V_AB_C_inv[np.newaxis,:,:,:] @ grad_V_AB.transpose([0,1,3,2]))

    #Log-weights of the partial derivatives
    grad_r_B = np.einsum("...j,...j", grad_r_B[:,:, np.newaxis,:], C_inv_delta_B[np.newaxis,:,:])
    delta_B_C_inv_delta_B = -0.5 * np.einsum("...j,...j", delta_B[np.newaxis,:,:], grad_V_B_inv_delta_B)

    reweights_exparg_grad = grad_r_B + delta_B_C_inv_delta_B

    prefactor = -0.5 * np.trace(C_inv @ grad_V_B, axis1 = 2, axis2 = 3) #Just a number
    
    new_log_weights_grad = log_weights_grad[:,:,np.newaxis] + reweights_exparg_grad + prefactor
    new_log_weights_grad = new_log_weights_grad.reshape([num_gradients, k1*k2])
     
    return new_means_grad, new_covs_grad, new_log_weights_grad

# Photon counting measurements
# ------------------------------------
def measure_fock_coherent(n : int, data : tuple, mode : int, inf=1e-4, red_lcg = True, hbar = 2.0) -> tuple:  
    """Returns data tuple after projecting a mode on fock state n (which is approximated via coherent states
    https://arxiv.org/abs/2305.17099)
    
    Args:
        n (int): photon number
        data (tuple) : [means, covs, weights, num_k] of measured state
        mode (int): mode that is measured
        inf (float): infidelity of the fock approx
        red_lcg (bool): to use reduced-lcg formalism
        hbar (float)
        
    Returns:
       tuple: data of post-selected state
    """
    #Approximate the fock state as a superposition of coherent states
    if red_lcg: #Use reduced lcog formalism
        povm_data = gen_fock_coherent(n, inf, fast=True)
    else:
        povm_data = gen_fock_coherent(n, inf, fast=False)
    
    return transform_data_due_to_measurement(data, [mode], povm_data, hbar)

def measure_fock_thermal(n: int, data : tuple, mode : int, r = 0.05, hbar = 2.0):
    """Returns data tuple after projecting a mode on a fock state n (which is approximated via thermal states
    from https://arxiv.org/abs/2103.05530)

    Args:
            n (int): photon number
            data (tuple) : [means, covs, weights, num_k] of measured state
            mode (int): mode that is measured
            r (float): squeezing parameter in the approximation
            hbar (int)
            
        Returns:
           tuple: data of post-selected state
    """
    #Approximate the fock state as a linear combination of thermal states
    povm_data = gen_fock_thermal(n, r)
    return transform_data_due_to_measurement(data, [mode], povm_data, hbar)

def measure_ppnrd(n : int, data : tuple, mode : int, M : int, hbar = 2.0):
    """Returns data tuple after measuring a mode with a pseudo photon number resolving detector,
    which is a linear combination of thermal states
        
    Args:
        n (int): number of clicks
        data (tuple) : [means, covs, weights, num_k] of measured state
        mode (int): mode that is measured
        M (int): number of on/off detectors in the ppnrd
        hbar (float)

    Returns:
        tuple" data of post-selected state
    """
    povm_data = ppnrd_povm_element(n, M, hbar)
    return transform_data_due_to_measurement(data, [mode], povm_data, hbar)

# Gaussian measurements

def measure_homodyne(data : tuple, modes : list, phases : list, results : list):
    """Returns data tuple after measuring some of the modes with a homodyne detector.

    Args: 
        data (tuple): [means, covs, weights, num_k] of measured state
        modes (list): modes being measured
        phases (list): list of angles of the local oscillator
        results (list): list of measurement results

    Returns: 
        tuple: data of post-selected state
    """
    
    return transform_data_due_to_homodyne_measurement(data, modes, phases, results)

def measure_generaldyne(data, modes, cov_povm, mean_povm, hbar = 2.0):
    """Returns data tuple after measuring some of the modes with a generaldyne detector.

    Args:
        data (tuple): [means, covs, weights, num_k] of measured state
        modes (list): modes being measured
        cov_povm (np.ndarray): even dimensional covariance matrix of the povm element
        mean_povm (np.ndarray): even dimensional displacement vector of the povm element
        hbar (float)
    """
    #Add extra dimension to povm covariance matrix and displacement vector
    log_weight_povm = np.array([- len(modes) * np.log(2*np.pi*hbar)])
    cov_povm = np.array([cov_povm])
    mean_povm = np.array([mean_povm])
    povm_data = mean_povm, cov_povm, log_weight_povm, 1
    new_data = transform_data_due_to_measurement(data, modes, povm_data, hbar)
    return new_data

def measure_fock_coherent_gradients(n, data_in, data_gradients, mode, inf, hbar=2.0):
    """
    """
    povm_data = gen_fock_coherent(n, inf, fast=False)
    new_data = transform_data_due_to_measurement(data_in, [mode], povm_data, hbar)
    new_data_gradients = transform_gradient_data_due_to_measurement(data_in, data_gradients, [mode], povm_data, hbar)
    return new_data, new_data_gradients


def ppnrd_povm_element(k : int, M : int, hbar = 2.0):
    """Returns data tuple of the pseudo-pnrd povm, which is a weighted sum of thermal states given in Eq. 37-40

    Args: 
        k (int): Number of clicks
        M (int): Number of on/off detectors
        hbar (float)
    Returns:
        data (tuple): data tuple of POVM (sum of Gaussians)
    """

    ls = np.arange(k+1)
    eta = (M - k + ls)/M
    if k == M:
        eta[0] = 1e-20 #close to zero to avoid inf
    
    nbars = (1 - eta)/eta
    log_weights = np.log(comb(M,k)) + np.log(comb(k,ls)) - np.log(eta) + 1j*np.pi*(ls % 2)
    covs = np.array([np.eye(2)*hbar/2*(1+2*nbar) for nbar in nbars])
    means = np.repeat( np.zeros(2)[np.newaxis,:], k+1, axis = 0)

    data = means, covs, log_weights, len(log_weights)
    
    return data

def gen_fock_thermal(n, r = 0.05, hbar = 2.0):

    if 1 / r**2 < n:
            raise ValueError(f"The parameter 1 / r ** 2={1 / r ** 2} is smaller than n={n}")
    # A simple function to calculate the parity
    parity = lambda n: 1 if n % 2 == 0 else -1
    # All the means are zero
    means = np.zeros([n + 1, 2])
    covs = np.array(
        [
            0.5 * hbar
            * np.identity(2)
            * (1 + (n - j) * r**2)
            / (1 - (n - j) * r**2)
            for j in range(n + 1)
        ]
    )
    log_weights = np.array(
        [np.log(1 - n * (r**2)) - np.log(1 - (n - j) * (r**2)) + np.log(comb(n,j)) + 1j*np.pi*(j%2)
            #(1 - n * (r**2)) / (1 - (n - j) * (r**2)) * comb(n, j) * parity(j)
            for j in range(n + 1)
        ],
    )

    log_norm = logsumexp(log_weights)
    return means, covs, log_weights-log_norm, len(log_weights)


    
