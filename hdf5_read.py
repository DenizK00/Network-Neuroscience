#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 18:26:09 2023

@author: deniz
"""

import h5py    
import pandas as pd
import seaborn as sns

f1 = h5py.File("Cleared_Cells_SNc.hdf5",'r+')
f2 = h5py.File("AllCellsWithAG.hdf5",'r+')


cleared_estimates = pd.DataFrame(f2['estimates/F_dff'])
accepted_index = list(f2["estimates/idx_components"])

cleared_df = cleared_estimates.iloc[accepted_index, :]

cleared_df_transpose = cleared_df.corr()
correlation_matrix = cleared_df.T.corr()

sns.heatmap(cleared_df.T.corr(), cmap="jet")


for i in list(f1["estimates/idx_components"]):
    print(i, " ", end="")
    
for i in list(f1["estimates/accepted_list"]):
    print(i, " ", end="")
    
    
