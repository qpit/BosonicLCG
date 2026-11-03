import numpy as np

def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def has_even_dimensions(matrix):
    m, n = matrix.shape
    
    if n != m:
        return False
    if n % 2 != 0:
        return False

def symplectic_form(num_modes):
    Omega = np.array([[0,1],[-1,0]]) #Single mode 
    return np.kron(np.eye(num_modes),Omega)

def is_symplectic(mat, rtol=1e-05, atol=1e-08):
    """Check if mat is symplectic"""
    
    has_even_dimensions(mat)
    num_modes = mat.shape[0] // 2
    Omega = symplectic_form(num_modes)
    return np.allclose(Omega, mat @ Omega @ mat.T, rtol=rtol, atol=atol)

def is_valid_covmat(covmat, hbar = 2, rtol = 1e-05, atol=1e-08):
    """Check if covmat is a valid quantum covariance matrix.
    See https://github.com/XanaduAI/thewalrus/blob/master/thewalrus/quantum/gaussian_checks.py#L26
    """

    #Check dimensions
    has_even_dimensions(covmat)
    #Check if symmetric
    if not np.allclose(covmat, covmat.T, rtol=rtol, atol=atol):
        return False
    
    #Check for violation of uncertainty relations
    num_modes = covmat.shape[0] // 2
    Omega = symplectic_form(num_modes)
    vals = np.linalg.eigvalsh(covmat + 0.5j * hbar * Omega)
    vals[np.abs(vals)< atol] = 0.0
    if np.all(vals >=0):
        return True


def xxpp_to_xpxp(S: np.ndarray, is_vector: bool = None):
    """Permutes the entries of the input from xxpp ordering to xpxp ordering.

    The permutation is applied to the last dimension (if S represents a
    stack of vectors) or to the last two dimensions (if S represents a
    stack of square matrices).

    Args:
        S (array): input even dimensional vector(s) of shape (..., 2k)
            or square matrix/matrices of shape (..., 2k, 2k)
        is_vector (bool, optional): if True, force vector-mode (permute
            only the last axis); if False, force matrix-mode (permute the
            last two axes). If None (default), infer from shape: treated
            as a matrix if the last two dimensions are equal, otherwise
            as a vector.

    Returns:
        (array): permuted array, same shape as input
    """
    S = np.asarray(S)
    shape = S.shape

    if S.ndim < 1:
        raise ValueError("The input array must have at least one dimension")

    if is_vector is None:
        is_matrix = S.ndim >= 2 and shape[-1] == shape[-2]
    else:
        if is_vector and S.ndim < 1:
            raise ValueError("The input array must have at least one dimension")
        if not is_vector and S.ndim < 2:
            raise ValueError("The input array must have at least two dimensions to be treated as a matrix")
        is_matrix = not is_vector

    if is_matrix and shape[-1] != shape[-2]:
        raise ValueError("The input matrix is not square")

    n = shape[-1]
    if n % 2 != 0:
        raise ValueError("The input array is not even-dimensional")

    n = n // 2
    ind = np.arange(2 * n).reshape(2, -1).T.flatten()

    if is_matrix:
        return S[..., ind[:, None], ind]

    return S[..., ind]


def xpxp_to_xxpp(S: np.ndarray, is_vector: bool = None):
    """Permutes the entries of the input from xpxp ordering to xxpp ordering.

    The permutation is applied to the last dimension (if S represents a
    stack of vectors) or to the last two dimensions (if S represents a
    stack of square matrices).

    Args:
        S (array): input even dimensional vector(s) of shape (..., 2k)
            or square matrix/matrices of shape (..., 2k, 2k)
        is_vector (bool, optional): if True, force vector-mode (permute
            only the last axis); if False, force matrix-mode (permute the
            last two axes). If None (default), infer from shape: treated
            as a matrix if the last two dimensions are equal, otherwise
            as a vector.

    Returns:
        (array): permuted array, same shape as input
    """
    S = np.asarray(S)
    shape = S.shape

    if S.ndim < 1:
        raise ValueError("The input array must have at least one dimension")

    if is_vector is None:
        is_matrix = S.ndim >= 2 and shape[-1] == shape[-2]
    else:
        if is_vector and S.ndim < 1:
            raise ValueError("The input array must have at least one dimension")
        if not is_vector and S.ndim < 2:
            raise ValueError("The input array must have at least two dimensions to be treated as a matrix")
        is_matrix = not is_vector

    if is_matrix and shape[-1] != shape[-2]:
        raise ValueError("The input matrix is not square")

    n = shape[-1]
    if n % 2 != 0:
        raise ValueError("The input array is not even-dimensional")

    n = n // 2
    ind = np.arange(2 * n).reshape(-1, 2).T.flatten()

    if is_matrix:
        return S[..., ind[:, None], ind]

    return S[..., ind]


