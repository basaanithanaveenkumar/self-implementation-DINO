import torch
import torch.nn as nn
import torch.nn.functional as F

class DINO_MLP_HD(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=1256, 
                 n_layers=4, use_layer_norm=True):
        super().__init__()
        
        # Build the MLP layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        
        # Hidden layers
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
        
        # Bottleneck layer
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        
        # Create the MLP
        self.mlp = nn.Sequential(*layers)

        # Last layer (corrected from nn.linear to nn.Linear)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Use Xavier initialization for linear layers
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = self.last_layer(x)
        return x