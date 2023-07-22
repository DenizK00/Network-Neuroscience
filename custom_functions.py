import h5py    
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
import tkinter as tk
from tkinter import filedialog
import os
import networkx as nx
import plotly.graph_objects as go


def create_time_ranges(df):
    if df.shape[0] < 3000:
        frame_list = [[0, df.shape[0]]]
    elif df.shape[0] == 3000:
        frame_list = [[0, 1500], [0, 3000]]
    return frame_list


def load_hdf5(filename):
    file = h5py.File(filename, 'r')
    dataset = file['estimates/F_dff']
    data = dataset[()]

    # thanks deniz <3
    cleared_estimates = pd.DataFrame(data)
    accepted_index = list(file["estimates/accepted_list"])

    cleared_df = cleared_estimates.iloc[accepted_index, :]
    accepted_data = cleared_df.to_numpy()
        
    return accepted_data


def plot_by_frame(data, time_range, title):
    cell_num, _ = data.shape
    fig = go.Figure()

    for i in range(cell_num):
        fig.add_trace(go.Scatter(x=time_range, y=data[i, time_range], name=f'Cell {i+1}'))

    fig.update_layout(
        xaxis_title='Frame',
        yaxis_title=r'Ca2+ Activity ($\Delta$F/F$_0$)',
        title=title
    )
    
    fig.show()

def plot_by_frame_old(data, time_range, title):
    cell_num, _ = data.shape
    for i in range(cell_num):
        plt.plot(time_range, data[i, time_range])

    plt.xlabel('Frame')
    plt.ylabel(r'Ca2+ Activity ($\Delta$F/F$_0$)')
    plt.title(title)
    plt.show()

    
def filter_outliers_zscore(dataframe, threshold=3):
    before_drop = dataframe.shape[0]
    z_scores = (dataframe - dataframe.mean()) / dataframe.std()
    outlier_rows = np.abs(z_scores) > threshold    
    filtered_dataframe = dataframe[~outlier_rows.any(axis=1)]
    drop_diff = before_drop - filtered_dataframe.shape[0]
    print("Data Loss due to filtering:", drop_diff)
    return filtered_dataframe


from scipy.stats import spearmanr
def get_corr_figure(data, time_range, title):
    num_cells = data.shape[0]
    corr_matrix = np.zeros((num_cells, num_cells))
    p_value_matrix = np.zeros((num_cells, num_cells))

    # Calculate correlation and p-values
    # Spearman's R
    for i in range(num_cells):
        for j in range(num_cells):
            corr_coef, p_value = spearmanr(data[i, time_range], data[j, time_range])
            corr_matrix[i, j] = corr_coef
            p_value_matrix[i, j] = p_value

    corr_matrix[p_value_matrix >= 0.05] = 0.0
    corr_matrix[p_value_matrix == 0] = 0.0

    fig, ax = plt.subplots()
    plt.imshow(corr_matrix)
    plt.xlabel('Neuron #')
    plt.ylabel("Neuron #")
    cbar = plt.colorbar()
    plt.title(title)
    cbar.ax.set_ylabel("Spearman's R")
    plt.show()
    
    
def get_hist_figure(data, time_range, title, c="C0"):
    num_cells = data.shape[0]
    corr_matrix = np.zeros((num_cells, num_cells))
    p_value_matrix = np.zeros((num_cells, num_cells))

    for i in range(num_cells):
        for j in range(num_cells):
            corr_coef, p_value = pearsonr(data[i, time_range], data[j, time_range])
            corr_matrix[i, j] = corr_coef
            p_value_matrix[i, j] = p_value

    corr_matrix[p_value_matrix >= 0.05] = 0.0
    corr_matrix[p_value_matrix == 0] = 0.0

    significant_correlations = corr_matrix[corr_matrix != 0]
    plt.hist(significant_correlations, bins=10, edgecolor='black', color=c)
    plt.xlabel("Correlation")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()
    
    
def filter_outliers_old(df, threshold=3):
    before_drop = df.shape[0]
    dft = df.T
    columns_to_drop = []
    for c in dft.columns:
        if any((dft[c] - dft[c].mean())/dft[c].std() > threshold):
            columns_to_drop.append(c)
        elif any((dft[c] - dft[c].mean())/dft[c].std() < -threshold):
            columns_to_drop.append(c)
            
    dft.drop(columns=columns_to_drop, inplace=True)
    after_drop = dft.shape[1]
    drop_diff = before_drop - after_drop
    print("Data Loss due to filtering:", str(round(drop_diff/before_drop*100, 2)) + "%")
    return dft.T[1:]


def filter_outliers(df, threshold=3):
    before_drop = df.shape[0]

    # Calculate z-scores for each column
    z_scores = (df - df.mean()) / df.std()
    
    is_outlier = (np.abs(z_scores) > threshold).any(axis=1)
    
    df = df[~is_outlier]

    drop_diff = before_drop - df.shape[0]
    print("Data Loss due to filtering:", str(round(drop_diff/before_drop*100, 2)) + "%")
    return df


def get_assortativity(data):
    num_rows = data.shape[0]
    assortativity_values = []

    for i in range(num_rows):
        correlations = np.corrcoef(data[i], rowvar=False)
        G = nx.from_numpy_array(correlations)
        
        try:
            assortativity = nx.degree_assortativity_coefficient(G)
            assortativity_values.append(assortativity)
        except nx.NetworkXError:
            assortativity_values.append(np.nan)


def pie_chart_above_mean(data, indexes, plot_title):
    above_mean_count = 0
    below_mean_count = 0

    means = []    

    for i in range(len(data)):
        values = data[i][0].flatten()
        mean_value = np.mean(values)

        means.append(mean_value)

    ns_mean = np.mean(means)

    for i in indexes:
        values = data[i][0].flatten()

        above_mean_count += np.sum(values > ns_mean)
        below_mean_count += np.sum(values < ns_mean)

    labels = ['Above Mean', 'Below Mean']
    sizes = [above_mean_count, below_mean_count]

    colors = ['#ff9999', '#66b3ff']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90,
        wedgeprops={'edgecolor': 'white'})

    center_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(center_circle)

    ax.axis('equal')

    # Add a title
    plt.title(plot_title, fontweight='bold')

    # Display the pie chart
    plt.show()

# def small_world(data):
#     small_world_coefs = []
#     for row in data:
#         graph = nx.from_numpy_array(row)
#         small_world_coefficient = nx.algorithms.smallworld.sigma(graph, niter=100, nrand=10)
#         small_world_coefs.append(small_world_coefficient)

#     return small_world_coefs

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit



def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

