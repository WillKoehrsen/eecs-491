import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.size'] = 18

import matplotlib.mlab as mlab

from scipy.stats import multivariate_normal

# Compute the log(sum_i exp(Z_i)) for an array Z
def log_sum_exp(Z):
    return np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))

# Compute the loglikelihood of data for a Gaussian Mixture Model
# with the specified weights, means, and covariances
def loglikelihood(data, weights, means, covs):
    num_clusters = len(means)
    num_dim = len(data[0])
    
    ll = 0
    
    for d in data:
        Z = np.zeros(num_clusters)
        for k in range(num_clusters):
            delta = np.array(d) - means[k]
            exponent_term = np.dot(delta.T, np.dot(np.linalg.inv(covs[k]), delta))

            # Log likelihood contribution for this data point and this cluster
            Z[k] += np.log(weights[k])
            Z[k] -= 1/2 * (num_dim * np.log(2 * np.pi) + np.log(np.linalg.det(covs[k])) + exponent_term)

        # Increase log likelihood by data point across all clusters
        ll += log_sum_exp(Z)
            
    return ll

# Calculate all the responsibilities for the data points
# under the specified Gaussian distributions
# Responsibility is the weight of the cluster times the likelihood
# of the data point under the cluster parameters
def compute_responsibilities(data, weights, means, covs):
    num_data = len(data)
    num_clusters = len(means)
    
    resps = np.zeros((num_data, num_clusters))
    
    # Iterate through each data point
    for i in range(num_data):
        # Iterate through each cluster
        for k in range(num_clusters):
            # Responsibility of cluster k for data point i
            resps[i, k] = weights[k] * multivariate_normal.pdf(data[i], means[k], covs[k])
      
    # Calculate total responsibility for each data point
    row_sums = resps.sum(axis=1)[:, np.newaxis]
    
    # Normalize the responsibilities by the sum
    resps = resps / row_sums
    
    return resps

# Sums the responsibilities for a given cluster
def compute_soft_counts(resps):
    return np.sum(resps, axis=0)


# Calculate cluster weights
# The weight for a cluster is the soft count of the cluster
# divided by the total soft counts
def compute_weights(counts):
    num_clusters = len(counts)
    weights = np.zeros(num_clusters)
    
    # Calculate weight for each cluster
    for k in range(num_clusters):
        weights[k] = counts[k] / np.sum(counts)
        
    return weights


def compute_means(data, resps, counts):
    num_clusters = len(counts)
    num_data = len(data)
    
    means = [np.zeros(len(data[0]))] * num_clusters
    
    # For each cluster, calculate the new location of the mean
    for k in range(num_clusters):
        weighted_sum = 0.
        
        # Iterate over all data points
        for i in range(num_data):
            # Weighted sum is the responsibility times the coordinates of the point
            weighted_sum += resps[i, k] * data[i]
           
        # Multiply the weighted value by 1 / soft count for the cluster
        means[k] = weighted_sum * 1 / counts[k]
        
    return means



# Compute new covariances given the data, the responsibilities
# of each cluster for each data point, the calculated soft counts, and the calculated means
def compute_covariances(data, resp, counts, means):
    num_clusters = len(counts)
    num_dim = len(data[0])
    num_data = len(data)
    covariances = [np.zeros((num_dim, num_dim))] * num_clusters
    
    # Iterate through all clusters
    for k in range(num_clusters):
        weighted_sum = np.zeros((num_dim, num_dim))
        
        # Iterate through all data points
        for i in range(num_data):
            # Compute the contribution to the covariance matrix of the data point
            weighted_sum += resp[i, k] * np.outer((data[i] - means[k]), (data[i] - means[k]).T)
            
        # To normalize the covariances, divide by the soft count of the cluster
        covariances[k] = weighted_sum / counts[k]
        
    return covariances


def EM_algorithm(data, init_means, init_covariances, init_weights, maxiter = 1000, thresh = 1e-4):
    means = init_means[:]
    covariances = init_covariances[:]
    weights = init_weights[:]
    
    num_data = len(data)
    num_dim = len(data[0])
    num_clusters = len(means)
    
    resp = np.zeros((num_data, num_clusters))
    ll = loglikelihood(data, weights, means, covariances)
    ll_trace = [ll]
    
    for it in range(maxiter):
        if it % 5 == 0:
            print('Iteration %d' % it)
        
        # Expectation Step
        # Assign responsibilities to data points for each cluster
        resp = compute_responsibilities(data, weights, means, covariances)
        
        # Maximization Step
        # Calculate new parameters based on responsibilities and data
        counts = compute_soft_counts(resp)
        weights = compute_weights(counts)
        means = compute_means(data, resp, counts)
        covariances = compute_covariances(data, resp, counts, means)
        
        ll_next = loglikelihood(data, weights, means, covariances)
        ll_trace.append(ll_next)
        
        if (ll_next - ll < thresh) and (ll_next > -np.inf):
            break
        ll = ll_next
    
    out = {'weights': weights, 'means': means, 'covs': covariances,
           'loglike': ll_trace, 'resp': resp}
    
    return out