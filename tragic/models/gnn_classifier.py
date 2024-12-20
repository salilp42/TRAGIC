import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import add_self_loops

class GATConvWithAlpha(GATConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, x, edge_index, return_attention_weights=False):
        if return_attention_weights:
            return super().forward(x, edge_index, return_attention_weights=True)
        return super().forward(x, edge_index)

class GNNTimeSeriesClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, heads=8, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        self.gat1 = GATConvWithAlpha(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.gat2 = GATConvWithAlpha(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, data, return_attention=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        
        # First GAT layer
        if return_attention:
            x1, attention1 = self.gat1(x, edge_index, return_attention_weights=True)
        else:
            x1 = self.gat1(x, edge_index)
        x1 = self.layer_norm1(x1)
        x1 = torch.relu(x1)
        
        # Second GAT layer
        if return_attention:
            x2, attention2 = self.gat2(x1, edge_index, return_attention_weights=True)
        else:
            x2 = self.gat2(x1, edge_index)
        x2 = self.layer_norm2(x2)
        x2 = torch.relu(x2)
        
        # Global pooling
        x_graph = global_mean_pool(x2, batch)
        
        # Classification
        logits = self.classifier(x_graph)
        
        if return_attention:
            return logits, (attention1, attention2)
        return logits
    
    def get_attention_maps(self, data):
        """Extract attention maps for visualization."""
        attention_maps = []
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        x = self.input_proj(x)
        
        # Get attention weights from each layer
        for layer in [self.gat1, self.gat2]:
            _, (edge_index, weights) = layer(x, edge_index, return_attention_weights=True)
            
            # Convert sparse attention to dense
            num_nodes = x.size(0)
            layer_maps = []
            
            for head in range(weights.size(1)):
                dense_att = torch.zeros((num_nodes, num_nodes), device=x.device)
                dense_att[edge_index[0], edge_index[1]] = weights[:, head]
                layer_maps.append(dense_att.cpu().numpy())
            
            attention_maps.append(layer_maps)
            x = layer(x, edge_index)
            
        return attention_maps
