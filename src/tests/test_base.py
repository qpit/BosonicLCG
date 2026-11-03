
import pytest
import numpy as np
from scipy.stats import unitary_group
from lcg_plus.base import State
from thewalrus.random import random_symplectic
from thewalrus.symplectic import xxpp_to_xpxp, is_symplectic

np.random.seed = 10

def random_gaussian_state(num_modes):
    S = xxpp_to_xpxp(random_symplectic(num_modes))
    d = np.random.uniform(size=(2*num_modes)) #random displacement
    state = State(num_modes)
    state.apply_gaussian_unitary(S)
    state.apply_displacement(d)
    return state


#Test Gaussian states
@pytest.mark.parametrize("num_modes", [10,5])
def test_initialize(num_modes  : int):
    state = State(num_modes)
    assert state.means.shape == (1, 2*num_modes)
    assert state.covs.shape == (1, 2*num_modes, 2*num_modes)


@pytest.mark.parametrize("num_modes", [10,5])
def test_gaussian(num_modes : int):

    S = random_symplectic(num_modes)
    assert is_symplectic(S)
    S = xxpp_to_xpxp(S)

    d = np.random.uniform(size=(2*num_modes)) #random displacement

    state = State(num_modes)
    state.apply_displacement(d)
    state.apply_gaussian_unitary(S)

    r = S @ d
    V = state.hbar/2 * S @ S.T

    assert np.allclose(state.means, r), "symplectic transform on means failed"
    assert np.allclose(state.covs, V), "symplectic transform on covs failed"

    X = np.random.uniform(size=(2*num_modes,2*num_modes))
    Y = np.random.uniform(size=(2*num_modes,2*num_modes))
    state.apply_gaussian_channel(X, Y)
    assert np.allclose(state.means, X @ r),  "gausian channel on means failed"
    assert np.allclose(state.covs, X @ V @ X.T + Y) ,"gausian channel on covs failed"

    state.post_select_homodyne

@pytest.mark.parametrize("num_modes",[10])
@pytest.mark.parametrize("num_detectors",[1,5,9])
def test_post_select_homodyne(num_modes : int, num_detectors : int):
    state = random_gaussian_state(num_modes)
    phases = np.random.uniform(size = num_detectors) * 2*np.pi
    results = np.random.uniform(size = num_detectors)
    modes = range(num_detectors)
    state.post_select_homodyne(modes, phases, results)

    k = num_modes - num_detectors
    assert state.num_modes == num_modes - num_detectors
    assert state.covs.shape == (1,2*k,2*k)
    assert state.means.shape == (1,2*k)
    assert state.norm < 1.0

@pytest.mark.parametrize("num_modes",[10])
@pytest.mark.parametrize("num_detectors",[1,5,9])
def test_post_select_generaldyne(num_modes : int, num_detectors : int):
    state = random_gaussian_state(num_modes)
    povm_covs = np.eye(2*num_detectors) #heterodyne
    povm_means = np.random.uniform(size = 2*num_detectors)
    modes = range(num_detectors)

    state.post_select_generaldyne(modes, povm_covs, povm_means)
    k = num_modes - num_detectors
    assert state.num_modes == num_modes - num_detectors
    assert state.covs.shape == (1,2*k,2*k)
    assert state.means.shape == (1,2*k)
    assert state.norm < 1.0

@pytest.mark.parametrize("num_modes", [3,5])
@pytest.mark.parametrize("n",[1,5])
def test_post_select_fock_coherent(num_modes : int, n : int):

    state = random_gaussian_state(num_modes)
    i = 0
    while i < num_modes - 1:
        print(i)
        state.post_select_fock(0, n, method_kwargs = {
            'method' : 'coherent_full', 
            'infidelity' : 1e-4, 
            'gradients' : False})
        k = (n+1)**(2*(i+1))
        l = num_modes - (i+1)
        assert state.num_k == k
        assert state.means.shape == (k,2*l)
        assert state.covs.shape == (1,2*l, 2*l)
        assert len(state.log_weights) == k
        i+= 1 

@pytest.mark.parametrize("num_modes", [3,5])
@pytest.mark.parametrize("n",[1,5])
def test_post_select_fock_coherent_red_lcg(num_modes : int, n : int):

    state = random_gaussian_state(num_modes)
    i = 0
    while i < num_modes - 1:
        k = int((n+1)*state.num_k*((n+1)*state.num_k + 1)/2)
        state.post_select_fock(0, n, method_kwargs = {
            'method' : 'coherent', 
            'infidelity' : 1e-4, 
            'gradients' : False})
        
        l = num_modes - (i+1)
        assert state.num_k == int((n+1)**(i+1))
        assert state.means.shape == (k,2*l)
        assert state.covs.shape == (1,2*l, 2*l)
        assert len(state.log_weights) == k
        i+= 1 
        

@pytest.mark.parametrize("num_modes", [3,5])
@pytest.mark.parametrize("n",[1,2])
@pytest.mark.parametrize("M", [4,10])
def test_post_select_ppnrd(num_modes : int, n : int, M : int):

    state = random_gaussian_state(num_modes)
    i = 0
    while i < num_modes - 1:
        k = int((n+1)*state.num_k*((n+1)*state.num_k + 1)/2)
        state.post_select_ppnrd(0,n,M)
        
        l = num_modes - (i+1)
        k = (n+1)**(i+1)
        assert state.means.shape == (k,2*l)
        assert state.covs.shape == (k,2*l, 2*l)
        assert len(state.log_weights) == k
        i+= 1 



