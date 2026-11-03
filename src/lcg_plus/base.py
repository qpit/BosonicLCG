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
from thewalrus.decompositions import williamson
from lcg_plus.operations.measurements import measure_fock_coherent, measure_ppnrd, measure_homodyne, measure_fock_thermal, measure_generaldyne
from lcg_plus.operations.measurements import measure_fock_coherent_gradients
from lcg_plus.operations.gaussian_unitaries import apply_symplectic_on_subsystem, apply_symplectic_full
from lcg_plus.operations.channels import apply_gaussian_channel_full
from lcg_plus.properties.normalisation import calculate_norm
from lcg_plus.properties.wigner import compute_wigner_function, compute_wigner_function_log
from lcg_plus.states.coherent import eps_superpos_coherent
from lcg_plus.operations.reduce import reduce_log, reduce_log_full, reduce_log_pure, find_unique_means_and_merge_weights


from lcg_plus.operations.sampling import *

from lcg_plus.helper.from_sf import chop_in_blocks_multi, chop_in_blocks_vector_multi
from lcg_plus.helper.matrices import is_symplectic
from scipy.linalg import block_diag

import itertools as it


hbar = 2

class State:
    """Simulate a bosonic state by representing its Wigner function as a linear combination of multivariate Gaussians https://arxiv.org/abs/2103.05530
    by keeping track of the (xpxp-ordered) means, (xpxp-ordered) covariance matrices, and coefficients in front of each Gaussian.
    """
    def __init__(self, num_modes = 1, hbar = 2):
        """Initialise single-mode vacuum by default.
        """
        self.hbar = hbar
        self.num_modes = num_modes
        self.means = np.array([np.zeros(self.num_modes * 2)])
        self.covs = np.array([np.eye(self.num_modes * 2)]) * self.hbar / 2
        self.log_weights = np.array([0]) #log(weights)
        self.weights = np.array([1]) # weights
        self.num_weights = len(self.weights)
        self.num_covs = len(self.covs) #Relevant for faster calculations
        self.num_k = self.num_weights #The first num_k Gaussians are treated "normally" i.e. those in left sum in Eq. (22) of https://arxiv.org/abs/2508.06175
        self.norm = 1 #Normalisation constant
        self.log_norm = 0


    def update_data(self, new_data : tuple):
        """Insert a custom data tuple, new_data = [means, covs, log_weights, num_k]. 
        This overrides the existing state data completely.
        """
        if len(new_data) != 4:
            raise ValueError('new_data must be [means, covs, log_weights, num_k] tuple.')
            
        self.means, self.covs, self.log_weights, self.num_k = new_data
        self.num_weights = len(self.log_weights)
        self.num_modes = int(np.shape(self.means)[-1]/2)
        
        if len(self.covs.shape) != 3: 
            self.covs = np.array([self.covs]) #Quick fix for places where covs has shape (2N,2N), not (1,2N,2N)
            
        self.num_covs = len(self.covs)
        self.weights = np.exp(self.log_weights)

    def update_gradients(self, new_gradients : tuple):
        """Insert a custom gradient tuple, new_gradients = [means_partial, covs_partial, log_weights_partial]
        Overrides any existing gradient data.
        """
        if len(new_gradients) != 3:
            raise ValueError('new_gradients must be [means_partial, covs_partial, log_weights_partial] tuple.')
        
        self.means_partial, self.covs_partial, self.log_weights_partial = new_gradients
        
    def normalise(self):
        """Subtract log_norm from log_weights
        """
        norm, log_norm = calculate_norm(self)

        if self.num_k != self.num_weights:
            self.log_weights -= np.log(norm) 
        else:
            self.log_weights -= log_norm 
            
        self.weights /= norm

    def apply_gaussian_unitary(self, symp_mat: np.ndarray, modes = None): 
        """Apply a Gaussian unitary to modes with symplectic matrix (in xpxp ordering) to the covs and means.
        """
        if not is_symplectic(symp_mat):
            raise Warning("symp_mat is not symplectic. Check its ordering.")
        if modes: 
            self.means, self.covs = apply_symplectic_on_subsystem(self.means, self.covs, self.num_modes, self.num_weights, symp_mat, modes)
        else:
            self.means, self.covs = apply_symplectic_full(self.means, self.covs, symp_mat) 
        
    def apply_displacement(self, d: np.ndarray):
        """Shift the means by d (in xpxp ordering)
        TO DO: add modes arg, i.e. displacements on some of the modes
        """
        if d.size != self.means[0].size:
            raise ValueError('d must be 2 x nmodes.')
        self.means += d

    def apply_gaussian_channel(self, X, Y):
        """Apply an (X,Y) Gaussian channel to the covs and means
        TO DO: add modes arg i.e. gaussian channel on some of the modes
        """
        self.means, self.covs = apply_gaussian_channel_full(self.means, self.covs, X, Y)
        #Update the gradients if any
        if hasattr(self, "means_partial"):
            self.means_partial, self.covs_partial = apply_gaussian_channel_full(self.means_partial,self.covs_partial, X , Y)

    def post_select_fock(self, mode : int, photon_number : int, method = 'coherent', infidelity = 1e-4, gradients = False, squeezing = None):
        """Post select on a photon number in the given mode. The measured mode is traced out.
        """
        data_in = self.means, self.covs, self.log_weights, self.num_k

        if gradients:
            data_partial = self.means_partial, self.covs_partial, self.log_weights_partial

        if method == 'coherent':
            
            if gradients == False:
                data_out = measure_fock_coherent(photon_number, data_in, mode, infidelity)
            else: 
                raise ValueError('Gradients not compatible with reduced coherent representation. Use the coherent_full method.')

        elif method == 'coherent_full':
            
            if gradients == False:
                data_out = measure_fock_coherent(photon_number, data_in, mode, infidelity, red_lcg=False)
            else: 
                data_out, data_gradients = measure_fock_coherent_gradients(photon_number, data_in, data_partial, mode, infidelity)

        elif method == 'thermal':
            if gradients == False:
                data_out = measure_fock_thermal(photon_number, data_in, mode, squeezing)
            else: 
                raise ValueError("Gradients not compatible with thermal fock representation. Use the coherent_full method.")

        self.update_data(data_out)

        if gradients:
            self.update_gradients(data_gradients)

        self.norm, self.log_norm = calculate_norm(self)

    def post_select_ppnrd(self, mode : int, click_number : int, total_detectors : int):
        """
        Detect mode wth pseudo photon-number resolving detection registering n clicks by demultiplexing into M on/off detectors.
        The pPNRD POVM is written as a linear combination of Gaussians (thermal states) and the
        state's Gaussian means, covs and weights are updated according to the Gaussian transformation rules of 
        Bourassa et al. 10.1103/PRXQuantum.2.040315. 
        """
        if click_number > total_detectors:
            raise ValueError("Number of clicks can't exceed number of on/off detectors in the fanout.")

        if self.num_k != self.num_weights:
            raise ValueError('This measurement is not yet compatible with fast gaussian rep.')
        
        data_in = self.means, self.covs, self.log_weights, self.num_k
        
        data_out = measure_ppnrd(click_number, data_in, mode, total_detectors)
        
        self.update_data(data_out)
        self.norm, self.log_norm = calculate_norm(self)
       
    
    def post_select_homodyne(self, modes : list, phases : float, results: float):
        """
        Post select mode with a homodyne detector with local oscillator phase.
        """

        data_in = self.means, self.covs, self.log_weights, self.num_k
        
        data_out = measure_homodyne(data_in, modes, phases, results)
        self.update_data(data_out)
        self.norm, self.log_norm = calculate_norm(self)


    def post_select_generaldyne(self, modes : list, povm_cov : np.ndarray, povm_mean : np.ndarray):
        """
        Post-select modes on a Gaussian POVM element parametrized by a covariance matrix and displacement vector
        """
        data_in = self.means, self.covs, self.log_weights, self.num_k
        data_out = measure_generaldyne(data_in, modes, povm_cov, povm_mean)
        self.update_data(data_out)
        self.norm, self.log_norm = calculate_norm(self)


    def get_wigner(self, xvec : np.array, pvec : np.array, indices = None, use_log_weights = True) -> np.ndarray:
        """Return the Wigner function evaluated on the 2D grid of points given by xvec and pvec.
        See https://github.com/XanaduAI/strawberryfields/blob/master/strawberryfields/backends/states.py#L1608.
        """

        self.normalise()

        if indices is not None: 
            sigmaA, sigmaAB, covs = chop_in_blocks_multi(self.covs, indices)
            muA, means = chop_in_blocks_vector_multi(self.means, indices)
        else:
            if self.num_modes != 1:
                raise ValueError('State has multiple modes, please specify indices.')
            means = self.means
            covs = self.covs

        if use_log_weights:
            W = compute_wigner_function_log(means, covs, self.log_weights, xvec, pvec)
        else:
            W = compute_wigner_function(means, covs, self.weights, xvec, pvec)

        return np.real_if_close(W)

    def tensor_product(self, num_copies : int):
        """Tensor product State num times
        """
        # Check number of modes in state
        if self.num_modes != 1:
            raise ValueError('This is a multimode state. So far, we can only tensor product single mode states.')

        if self.num_k != self.num_weights:
            raise Warning('I didnt check whether it handles the fast rep correctly.')
        
        new_means, new_covs, new_log_weights = compute_nfold_tensor_product(self.means, self.covs, self.log_weights, num_copies)
        self.update_data([new_means, new_covs, new_log_weights, len(new_log_weights)])


    def add_state(self, state):
        """Tensor product of a state with a user-specified state in sum of Gaussian representation
        """

        data1 = self.means, self.covs, self.log_weights, self.num_k
        data2 = state.means, state.covs, state.log_weights, state.num_k

        data_out = compute_tensor_product(data1, data2)
        self.update_data(data_out)
        
    def reduce_equal_means(self):
        """Merge Gaussians with equal means into a single Gaussian
        """
        if self.num_covs != 1:
            raise ValueError("Cannot merge means if Gaussians have different covariance matrices.")
        
        sort = False
        if self.num_k != self.num_weights:
            sort = True
            
        reduced_means, reduced_log_weights, num_k = find_unique_means_and_merge_weights(self.means, self.log_weights, sort)
        self.update_data([reduced_means, self.covs, reduced_log_weights, num_k])
        
        
    def reduce_pure(self, max_photons : int, infid = 1e-6):
        """Map the state to a superposition of (max_photons+1) coherent states.
        """
        if self.num_modes != 1:
            raise ValueError("Only single-mode states can currently be reduced.")
        if self.num_covs != 1:
            raise ValueError("The state has several covariance matrices.")
        
        #Perform Wiliamson decomposition on the covariance matrix and undo the symplectic transform.
        D, S = williamson(self.covs[0])
        if np.trace(D) != self.hbar:
            raise ValueError('State is not pure.')
        
        #Undo the Gaussian unitary gate from Williamson
        self.apply_gaussian_unitary(np.linalg.inv(S))

        data = self.means, self.covs, self.log_weights, self.num_k
        
        radius = eps_superpos_coherent(max_photons, infid) #amplitude of coherent state for target fidelity
        new_data = reduce_log_pure(max_photons, radius, data)

        self.update_data(new_data)
        self.norm, self.log_norm = calculate_norm()
        self.normalise()
    
        #Re-apply the Gaussian unitary
        self.apply_gaussian_unitary(S)


    def reduce_mixed(self, sd = 6, infid = 1e-6):
        """Map the state to a linear combination of ~((max_photons+1)^2) Gaussians
        """
        #Perform Wiliamson decomposition on the covariance matrix and undo the symplectic transform.
        D, S = williamson(self.covs[0])
        nu = D - np.eye(2)
        
        #Undo the Gaussian unitary gate from Williamson
        self.apply_gaussian_unitary(np.linalg.inv(S))
        
        #Remove thermal terms from cov
        data = self.means, self.covs - nu, self.log_weights, self.num_k
        self.update_data(data)
        
        #Find nmax from first two photon number moments
        nbar, nvar = self.get_photon_number_moments()
        nmax = int( np.ceil(nbar.real+sd*np.sqrt(nvar.real)))

        eps = eps_superpos_coherent(nmax, infid)
        
        if self.num_k != self.num_weights:
            #Perform the reduction with fast rep
            new_data = reduce_log(nmax, eps, data)
  
        else:
            #Perform the reduction with full rep
            new_data = reduce_log_full(nmax, eps, data)

        self.update_data(new_data)
        self.norm, self.log_norm = calculate_norm()
        self.normalise()
        
        #Re-apply the random Gaussian random displacement and the Gaussian unitary
        self.covs += nu
        self.apply_gaussian_unitary(S)

    def sample_dyne(self, modes, shots=1, covmat = [], method = 'normal'):
        """Performs general-dyne measurements on a set of modes. 
        """
            
        means_quad, covs_quad, quad_ind = select_quads(self, modes, covmat)
            
        ub_ind, ub_weights, ub_weights_prob = get_upbnd_weights(means_quad, covs_quad, self.log_weights, method)

        # Perform the rejection sampling technique until the desired number of shots
        # are acquired
    
        vals = np.zeros((shots, len(modes)))
        reject_vals = []
        
        for i in range(shots):
            drawn = False
            while not drawn:
                
                # Sample an index for a peak from the upperbounding function
                # according to ub_weights_prob
                peak_ind_sample = np.random.choice(ub_ind, size=1, p=ub_weights_prob)[0]
                # Get the associated mean covariance for that peak
                mean_sample = means_quad[peak_ind_sample].real
                
                if len(covs_quad) != 1:
                    cov_sample = covs_quad[peak_ind_sample]
                else: 
                    cov_sample = covs_quad[0]
                # Sample a phase space value from the peak
                peak_sample = np.random.multivariate_normal(mean_sample, cov_sample, size =1)[0]
    
                # Calculate the probability at the sampled point
                prob_dist_val = generaldyne_probability(peak_sample, means_quad, covs_quad, self.log_weights)
    
                #Calculate the upper bounding function at the sampled point
                prob_upbnd = generaldyne_probability(peak_sample, means_quad[ub_ind,:].real, covs_quad, ub_weights)
                
                # Sample point between 0 and upperbound function at the phase space sample
                vertical_sample = np.random.random(size=1) * prob_upbnd
                # Keep or reject phase space sample based on whether vertical_sample falls
                # above or below the value of the probability distribution
    
                if vertical_sample > prob_dist_val:
                    reject_vals.append(peak_sample)
                if vertical_sample <= prob_dist_val:
                    drawn = True
                    vals[i] = peak_sample
        
        return vals, np.array(reject_vals)
            
    
    def sample_dyne_gaussian(self, modes, shots = 1, covmat = [], factor = 0):
        """Performs general-dyne measurements on a set of modes using a Gaussian 
        upper bounding function based on the first and second moments of the state. 
        
        """
        means_quad, covs_quad, quad_ind = select_quads(self, modes, covmat)
        cov_ub, mean_ub, scale = get_upbnd_gaussian(self, means_quad, covs_quad, quad_ind)
            
        #Perform rejection sampling with the single guassian upper bounding function
        vals = np.zeros((shots, len(modes)))
        reject_vals = []        
        
        for i in range(shots):
            drawn = False
            while not drawn:
              
                #Draw a sample from the Gaussian
                sample = np.random.multivariate_normal(mean_ub, cov_ub, size =1)[0]
                if factor ==0:
                    prefactor = 1/np.sqrt(2*np.pi*np.linalg.det(cov_ub))
                    prob_upbnd = generaldyne_probability(sample, mean_ub, cov_ub, np.array([np.log(scale/prefactor)])) 
                else:
                    prob_upbnd = generaldyne_probability(sample, mean_ub, cov_ub, np.array([np.log(factor)]))
                    
                
                y = np.random.random(size=1)*prob_upbnd 
    
                prob_dist_val = generaldyne_probability(sample, means_quad, covs_quad, self.log_weights)
             
                if y > prob_dist_val:
                    reject_vals.append(sample)
                    
                elif y <= prob_dist_val:
                    drawn =True
                    vals[i] = sample
    
        return vals, np.array(reject_vals)


def compute_nfold_tensor_product(means, covs, log_weights, num_copies):
    """Comptue the new means, covs and log_weights when the tensor product of the same state is taken num times. 
    """
    if np.shape(covs)[0]==1:
        new_covs = np.array([block_diag(*np.repeat(covs, num_copies, axis = 0))])

    new_log_weights = np.sum(np.array(list(it.product(log_weights.tolist(), repeat = num_copies))), axis = 1)
    
    new_means = np.reshape(np.array(list(it.product(means, repeat = num_copies))), (len(log_weights)**num_copies, num_copies*2) )

    return new_means, new_covs, new_log_weights

def compute_tensor_product(data1, data2):
    """Compute the means, covs and log_weights of the tensor product of two different states. 
    """
    #unpack data
    means1, covs1, log_weights1, k1 = data1
    means2, covs2, log_weights2, k2 = data2

    num_weights1 = len(log_weights1)
    num_weights2 = len(log_weights2)

    #In coherent picture, covariances are the same for every weight
    if len(covs1) != 1 or len(covs2) != 1:
        new_covs = np.array([block_diag(*i) for i in list(it.product(covs1,covs2))])
    else:
        
        new_covs = np.array([block_diag(*list([np.squeeze(covs1),np.squeeze(covs2)]))])

    #Deal with different fast rep scenarios separately to obtain the correct ordering

    if k1 != num_weights1 and k2 != num_weights2:  #Both are in fast rep
        
        nw1 = np.sum(np.array(list(it.product(log_weights1[0:k1], log_weights2[0:k2]))),axis=1) 
        nw2 = np.sum(np.array(list(it.product(log_weights1[0:k1], log_weights2[k2::]))),axis=1) 
        nw3 = np.sum(np.array(list(it.product(log_weights1[k1::], log_weights2[0:k2]))),axis=1) 
        nw4 = np.sum(np.array(list(it.product(log_weights1[k1::], log_weights2[k2::]))),axis=1) - np.log(2) #To counteract +2*np.log(2)
        nw5 = np.sum(np.array(list(it.product(log_weights1[k1::], log_weights2[k2::].conjugate()))),axis=1) - np.log(2)

        nm1 = np.array([np.concatenate(i) for i in list(it.product(means1[0:k1],means2[0:k2]))])
        nm2 = np.array([np.concatenate(i) for i in list(it.product(means1[0:k1],means2[k2::]))])
        nm3 = np.array([np.concatenate(i) for i in list(it.product(means1[k1::],means2[0:k2]))])
        nm4 = np.array([np.concatenate(i) for i in list(it.product(means1[k1::],means2[k2::]))])
        nm5 = np.array([np.concatenate(i) for i in list(it.product(means1[k1::],means2[k2::].conjugate()))])

        new_weights = np.concatenate((nw1,nw2,nw3,nw4,nw5))
        new_means = np.concatenate((nm1,nm2,nm3,nm4,nm5))
        
        num = k1*k2

    elif k1 != num_weights1: #state1 is in fast rep

        nw1 = np.sum(np.array(list(it.product(log_weights1[0:k1], log_weights2))),axis=1) 
        nw2 = np.sum(np.array(list(it.product(log_weights1[k1::], log_weights2))),axis=1)

        nm1 = np.array([np.concatenate(i) for i in list(it.product(means1[0:k1],means2))])
        nm2 = np.array([np.concatenate(i) for i in list(it.product(means1[k1::],means2))])
        
        new_weights = np.concatenate((nw1,nw2))
        new_means = np.concatenate((nm1,nm2))
        
        num = k1

    elif k2 != num_weights2: #state2 is in fast rep
        nw1 = np.sum(np.array(list(it.product(log_weights1, log_weights2[0:k2]))),axis=1) 
        nw2 = np.sum(np.array(list(it.product(log_weights1, log_weights2[k2::]))),axis=1)

        nm1 = np.array([np.concatenate(i) for i in list(it.product(means1,means2[0:k2]))])
        nm2 = np.array([np.concatenate(i) for i in list(it.product(means1,means2[k2::]))])
        
        new_weights = np.concatenate((nw1,nw2))
        new_means = np.concatenate((nm1,nm2))
        
        num = k2
        
    else: #Neither are in fast rep
        
        new_weights = np.sum(np.array(list(it.product(log_weights1, log_weights2))),axis=1) 
        #Hack to fix list of list problem
        new_means = list(it.product(means1,means2))
        new_means = np.array([np.concatenate(i) for i in new_means])

        num = len(new_weights)

        
    return new_means, new_covs, new_weights, num
