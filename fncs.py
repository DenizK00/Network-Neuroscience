import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import h5py
#import bct
import pandas as pd

def loadHDF5(filename):
    file = h5py.File(filename, 'r')
    dataset = file['estimates/F_dff']
    data = dataset[()]
    
    # thanks deniz <3
    cleared_estimates = pd.DataFrame(data)
    accepted_index = list(file["estimates/idx_components"])

    cleared_df = cleared_estimates.iloc[accepted_index, :]
    
    accepted_data = cleared_df.to_numpy()
        
    return accepted_data

def significantCorrelations(data, flag="pearson"):
    """Make a matrix of significant correlations from an array of traces."""
    
    # Verify the orientation of the data is cell # x activity
    if data.shape[0] > data.shape[1]:
        data = data.T
    
    # Identify the number of cells in the recording
    num_cells = data.shape[0]
    print("The number of cells in the dataset is %s" %num_cells)
    
    # create the holding matrices
    correlation_matrix = np.zeros((num_cells, num_cells))
    p_value_matrix = np.zeros((num_cells, num_cells))
    
    # Calculate correlation and p-values
    # By changing the definition of flag, we can compute correlations via pearson, spearman, or kendall methods.
    if flag == "pearson":
        for i in range(num_cells):
            for j in range(num_cells):
                corr_coef, p_value = pearsonr(data[i], data[j])
                correlation_matrix[i, j] = corr_coef
                p_value_matrix[i, j] = p_value

    elif flag == "spearman":
        for i in range(num_cells):
            for j in range(num_cells):
                corr_coef, p_value = spearmanr(data[i], data[j])
                correlation_matrix[i, j] = corr_coef
                p_value_matrix[i, j] = p_value
                
    elif flag == "kendall":
        for i in range(num_cells):
            for j in range(num_cells):
                corr_coef, p_value = kendalltau(data[i], data[j])
                correlation_matrix[i, j] = corr_coef
                p_value_matrix[i, j] = p_value
    
    # Thresholding based on p-values
    thresholded_matrix = correlation_matrix.copy()
    thresholded_matrix[p_value_matrix >= 0.05] = 0.0
    thresholded_matrix[p_value_matrix == 0] = 0.0
    
    return thresholded_matrix, correlation_matrix, p_value_matrix
    
    
def nodeStrength(c_matrix):
    """Correlation matrix (Numpy array, shape is # of cells x # of cells) should be passed in. It is assumed that the matrix is only those that are manually accepted."""
    
    node_network = np.zeros((len(c_matrix), 1))
    normalized_node_network = np.zeros((len(c_matrix), 1))

    for cellNumber in range(len(c_matrix)):
        node_network[cellNumber] = np.sum(c_matrix[cellNumber])
        normalized_node_network[cellNumber] = node_network[cellNumber] / (len(c_matrix) - 1)

        
    return node_network, normalized_node_network

def nonnegativeMatrix(c_matrix):
    """Removes negative correlations from network."""
    nonnegative_matrix = c_matrix.copy()
    nonnegative_matrix[c_matrix < 0] = 0
    
    return nonnegative_matrix

def network_assortativity(c_matrix, flag="weighted"):
    """
    Must be a nonnegative matrix.
    
    Weighted flag preserves correlation values. If set to binary, the network is transformed to 1s and 0s.
    """
    if flag == "weighted":
        assortativity = assortativity_wei(c_matrix, 0)
    elif flag == "binary":
        assortativity = assortativity_bin(c_matrix, 0)
        
    return assortavivity

def network_modularity(c_matrix, gamma=1):
    """Measure modularity metric and determine community structure.
    
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller module
    """
    
    community_structure, modularity_metric = modularity_und(c_matrix, gamma)
        
    return community_structure, modularity_metric


