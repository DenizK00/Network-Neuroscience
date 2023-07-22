import numpy as np
import networkx as nx

def nodeStrength(tbl, regionName):
    control = np.empty(len(tbl), dtype=object)
    injected = np.empty(len(tbl), dtype=object)
    
    for i in range(len(tbl)):
        network = tbl['functional_connectivity_control'][i]
        node_network = np.zeros((len(network), network[0].shape[0]))
        
        for cellNumber in range(len(network)):
            node_network[cellNumber, :] = np.sum(network[cellNumber], axis=1)
        
        normalized_node_network = np.mean(node_network, axis=0) / (len(network) - 1)
        control[i] = normalized_node_network
      
    return control, injected
