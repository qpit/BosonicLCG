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
from lcg_plus.helper.matrices import has_even_dimensions, symplectic_form

def is_valid_gaussian_channel(X, Y, hbar = 2, atol = 1e-08):
    """Check if (X,Y) is a valid Gaussian quantum channel."""
    #Check dimensions
    has_even_dimensions(X)
    has_even_dimensions(Y)
    m,n = X.shape
    k,l = Y.shape
    if m != k: 
        return False
    #Check for violation of uncertainty relations
    num_modes = m // 2
    Omega = symplectic_form(num_modes)
    vals = np.linalg.eigvalsh(Y - 0.5j * hbar * X @ Omega @ X.T + 0.5j * hbar * Omega)
    vals[np.abs(vals)< atol] = 0.0
    if np.all(vals >=0):
        return True

def loss_channel_matrices(etas : np.ndarray, nbars : np.ndarray, hbar = 2):
    """Get X and Y matrices (xpxp ordering) for the multimode loss channel with transmissivity eta and nbar in each mode.

    Args:
        etas : transmissivity in each mode
        nbars : average photon number of coupling environment in each mode

    Returns:
        X : sqrt(eta)*I
        Y : (1-eta) * hbar / 2 * (2*nbar+1) * I 
    """
    if etas.shape != nbars.shape:
        raise ValueError("etas and nbars must have same length.")
    
    X = np.diag(np.repeat(np.sqrt(etas),2))
    Y = np.diag(np.repeat( (1-etas) * hbar / 2 * (2*nbars + 1) , 2))

    return X, Y 

def gain_channel_matrices(Gs : np.array, hbar =2):
    """Get X and Y matrices (xpxp ordering) for the multimode gain channel.

    Args:
        Gs : gain in each mode
    Returns:
        X : sqrt(G)*I
        Y : (G-1) * hbar / 2 
    """
    X = np.diag(np.repeat(np.sqrt(Gs), 2))
    Y = np.diag(np.repeat( (Gs-1) * hbar / 2 , 2))
    return X, Y


def apply_gaussian_channel_full(means, covs, X : np.ndarray, Y : np.ndarray):
    """Transform the covarance matrices and displacement vectors according to the Gaussian channel parameterized by X and Y (in xpxp ordering).
    V -> X V X.T + Y
    r -> X r
    """
    means = np.einsum("...jk,...k", X, means)

    if covs.shape[0] == 1:
        covs = X @ covs @ X.T + Y
    else:
        covs = np.einsum("...jk,...kl,...lm", X, covs, X.T) + Y[np.newaxis, :]
    return means, covs