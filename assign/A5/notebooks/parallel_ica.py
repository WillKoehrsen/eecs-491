import numpy as np
from scipy import signal, linalg

import warnings


def logcosh(x, alpha = 1.0):

    x *= alpha
    
    gx = np.tanh(x, x)
    g_x = np.empty(x.shape[0])
    
    for i, gx_i in enumerate(gx):
        g_x[i] = (alpha * (1 - gx_i ** 2)).mean()
        
    return gx, g_x

# Whiten and pre-process components
def whiten_components(X, n_components):
    
    n, p = X.shape

    X_mean = X.mean(axis=-1)
    
    # Subtract the mean for 0 mean
    X -= X_mean[:, np.newaxis]
    
    # Preprocessing by PCA
    u, d, _ = linalg.svd(X, full_matrices=False)
    
    # Whitening matrix
    whitening = (u / d).T[:n_components]
    
    # Project data onto the principal components using the whitening matrix
    X1 = np.dot(whitening, X)
    X1 *= np.sqrt(p)
    
    # Return whitened components, whitening matrix, and mean of components
    return X1, whitening, X_mean

# Symmetric decorrelation of un_mixing matrix
# https://ieeexplore.ieee.org/document/398721/
# Ensures no vectors are privileged over others
def symmetric_decorrelation(un_mixing):
    
    # Find eigenvalues and eigenvectors of initial weight matrix
    eig_values, eig_vectors = linalg.eigh(np.dot(un_mixing, un_mixing.T))
    # Symmetric decorrelation equation
    sym_un_mixing = np.dot(np.dot(eig_vectors * (1 / np.sqrt(eig_values)), eig_vectors.T), un_mixing)
    
    return sym_un_mixing

def parallel_ica(X, init_un_mixing, alpha = 1.0, max_iter = 1000, tol = 1e-4, print_negentropy=False):
    
    # Symmetric decorrelation of initial un-mixing components 
    un_mixing = symmetric_decorrelation(init_un_mixing)
    
    
    p = float(X.shape[1])
    
    # Iteratively update the un-mixing matrix
    for i in range(max_iter):
        
        # Function and derivative 
        gwtx, g_wtx = logcosh(np.dot(un_mixing, X), alpha)
        
        
        new_un_mixing = symmetric_decorrelation(np.dot(gwtx, X.T) / p - g_wtx[:, np.newaxis] * un_mixing)
        
        # Calculate negentropy based on logcosh
        lim = max(abs(abs(np.diag(np.dot(new_un_mixing, un_mixing.T))) - 1))
        
        if print_negentropy:
            print('Iteration: {} Increase in Negentropy: {:0.4f}.'.format(i, lim))
            
        # Update un-mixing 
        un_mixing = new_un_mixing

        # Check for convergence
        if lim < tol:
            break
            
    else:
        warnings.warn('FastICA algorithm did not converge. Considering increasing '
                      'tolerance or increasing the maximum number of iterations.')
        
    
    return un_mixing
    
    
    
# X = mixing * sources
# sources = un-mixing * whitening * X
def perform_fastica(X, n_components, alpha = 1.0, max_iter = 200, tol = 1e-4,
                   print_negentropy=False):
    
    # Whiten components by subtracting mean
    X1, whitening, X_mean = whiten_components(X.T, n_components)
    
    # initial un_mixing components
    init_un_mixing = np.asarray(np.random.normal(size = (n_components, n_components)))
    
    # Solve ica using the parallel ica algorithm
    un_mixing = parallel_ica(X1, init_un_mixing, alpha, max_iter, tol, print_negentropy)

    # Calculate the sources
    sources = np.dot(np.dot(un_mixing, whitening), X.T).T
    
    # Calculate the mixing matrix
    w = np.dot(un_mixing, whitening)
    mixing = linalg.pinv(w)
    
    # Return mixing matrix, sources, and mean of X
    return mixing, sources, X_mean


def inverse_fastica(mixing, sources, X_mean):
    # Inverse transform
    X = np.dot(sources, mixing.T)
    # Add back in mean
    X += X_mean
    
    return X