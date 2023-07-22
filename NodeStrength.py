import numpy as np
import networkx as nx
import h5py    
import pandas as pd


def nodeStrength(tbl):
    """Table should contain a column named functional connectivity ?"""
    
    for i in range(len(tbl)):
        network = tbl['functional_connectivity'][i]
        node_network = np.zeros((len(network), network[0].shape[0]))
        
        for cellNumber in range(len(network)):
            node_network[cellNumber, :] = np.sum(network[cellNumber], axis=1)
        
        normalized_node_network = np.mean(node_network, axis=0) / (len(network) - 1)
        tbl.at[i, 'nodeStr'] = node_network
        tbl.at[i, 'nodeStr_norm'] = normalized_node_network
        
    return tbl

def hello():
    print("hey")
    
