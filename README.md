# Network Neuroscience Project

This project focuses on analyzing and visualizing neural network data using network neuroscience techniques. The code provided here processes and analyzes neural activity data from different regions (The Ventral Tegmental Area and Substantia Nigra) and networks within the brain. It utilizes Python with various libraries for data processing, visualization, and network analysis.

## Project Overview

The main objectives of this project are:
1. Load and preprocess neural activity data from HDF5 files. (ventral tegmental area vs substantia nigra).
2. Filter out outliers from the data.
3. Calculate and visualize functional connectivity and dynamic connectivity of neural networks.
4. Create the significant correlations matrix using the correlation matrix, p-value matrix and thresholding.
5. Assess regional functional connectivity and variability (coefficient of variation).
6. Analyze node strength and its distribution in the networks.
7. Calculate and visualize network assortativity, modularity, path lengths, clustering coeffiecient, and
   small-world coefficient.
8. Visualize the Networks.

## Requirements

To run this code, you'll need to have the following installed:

- Python 3.x
- Libraries: `numpy`, `seaborn`, `matplotlib`, `plotly`, `pandas`, `networkx`

## How to Use

1. Ensure all the required libraries are installed.
2. Execute the provided code in your Python environment.
3. A file dialog will appear, prompting you to select the directory containing HDF5 files for analysis.
4. The code will process the data, plot various visualizations, and display the results.

## Code Structure

The code is divided into several sections:

1. Data Loading and Preprocessing:
   - Load and preprocess neural activity data from HDF5 files.
   - Filter out outliers from the data.

2. Functional Connectivity Analysis:
   - Calculate and visualize functional connectivity of neural networks.
   - Assess regional functional connectivity and variability.

3. Dynamic Connectivity Analysis:
   - Calculate and visualize dynamic connectivity over time.

4. Network Analysis:
   - Analyze node strength and its distribution in the networks.
   - Calculate and visualize network assortativity and modularity.
   - Calculate and visualize network path lengths, clustering coeffiecient, and
   small-world coefficient.
   - Demonstrate and compare variability (Coefficient of Variation) for the network measures.

5. Visualization:
   - Visualize the results for network measures through different plots, such as histograms, box plots, and pie charts.
   - Visualize the networks.

## Results

The code will generate various visualizations, providing insights into the neural network's functional and dynamic connectivity, node strength, assortativity, modularity, and more.

Note: This project assumes that the input HDF5 files contain appropriate neural activity data for analysis. The results and interpretations depend on the quality and nature of the data provided.
