import numpy as np
import torch
from torch_geometric.data import Data
from tslearn.datasets import UCR_UEA_datasets

def load_dataset(dataset_name):
    """Load and preprocess dataset from UCR/UEA archive."""
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)
    
    # Combine train and test for full dataset
    X_all = np.concatenate([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    
    # Get unique labels
    unique_labels = np.unique(y_all)
    
    return X_all, y_all, unique_labels, (X_train, y_train, X_test, y_test)

def create_graph_from_timeseries(x, window_size=5):
    """Convert time series to graph structure."""
    num_points = len(x)
    
    # Create node features (value and normalized position)
    node_features = np.column_stack([
        x,
        np.linspace(0, 1, num_points)
    ])
    
    # Create edges between neighboring points within window
    edges = []
    for i in range(num_points):
        for j in range(max(0, i-window_size), min(num_points, i+window_size+1)):
            if i != j:
                edges.append([i, j])
    
    # Convert to torch tensors
    x_tensor = torch.FloatTensor(node_features)
    edge_index = torch.LongTensor(edges).t().contiguous()
    
    # Create PyG Data object
    data = Data(x=x_tensor, edge_index=edge_index)
    
    return data

def create_graph_dataset(X, y, window_size=5):
    """Convert array of time series to list of graph objects."""
    graphs = []
    for x, label in zip(X, y):
        data = create_graph_from_timeseries(x.flatten(), window_size)
        data.y = torch.LongTensor([label])
        graphs.append(data)
    return graphs
