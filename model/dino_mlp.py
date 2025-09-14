import torch
import torch.nn as nn
import torch.nn.functional as F

class DINO_MLP_HD(nn.Module):
    """
    A high-dimensional Multi-Layer Perceptron (MLP) head for DINO (Distillation 
    with NO labels) framework.
    
    This MLP head projects input features to a lower-dimensional output space
    through multiple hidden layers with non-linear activations, layer normalization,
    and a bottleneck layer. It is specifically designed for self-supervised 
    learning in the DINO framework.
    
    Attributes:
        mlp (nn.Sequential): The main MLP backbone consisting of multiple linear,
                             layer norm, and activation layers.
        last_layer (nn.Linear): Final linear projection layer to output dimension.
    
    Args:
        in_dim (int): Dimension of input features.
        out_dim (int): Dimension of output features.
        hidden_dim (int, optional): Dimension of hidden layers. Default: 2048.
        bottleneck_dim (int, optional): Dimension of the bottleneck layer. 
                                       Default: 1256.
        n_layers (int, optional): Total number of linear layers (including input, 
                                 hidden, and bottleneck layers). Default: 4.
        use_layer_norm (bool, optional): Whether to use LayerNorm after linear 
                                        layers. Default: True.
    """
    
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=1256, 
                 n_layers=4, use_layer_norm=True):
        super().__init__()
        
        # Validate input parameters
        if n_layers < 2:
            raise ValueError("n_layers must be at least 2 (input + bottleneck layers)")
        
        # Build the MLP layers
        layers = []
        
        # Input layer: Project from input dimension to hidden dimension
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))  # Normalize hidden states
        layers.append(nn.GELU())  # GELU activation for non-linearity
        
        # Hidden layers: Multiple layers with same hidden dimension
        # Number of hidden layers = total layers - 2 (input and bottleneck layers)
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))  # Hidden linear layer
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))  # Optional layer normalization
            layers.append(nn.GELU())  # GELU activation function
        
        # Bottleneck layer: Project from hidden dimension to bottleneck dimension
        # This creates a compressed representation before final projection
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        
        # Create the MLP as a sequential container
        self.mlp = nn.Sequential(*layers)

        # Last layer: Final projection from bottleneck to output dimension
        # This layer is kept separate from the main MLP for flexibility
        self.last_layer = nn.Linear(bottleneck_dim, out_dim)
        
        # Initialize weights using custom initialization scheme
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Custom weight initialization for different layer types.
        
        Applies Xavier uniform initialization to linear layers and standard
        initialization to LayerNorm layers.
        
        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Linear):
            # Xavier uniform initialization for better gradient flow
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Initialize biases to zero
        elif isinstance(m, nn.LayerNorm):
            # Standard LayerNorm initialization
            nn.init.constant_(m.weight, 1.0)  # Scale parameter
            nn.init.constant_(m.bias, 0)      # Bias parameter

    def forward(self, x):
        """
        Forward pass through the MLP head.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_dim)
        """
        # Pass through main MLP backbone
        x = self.mlp(x)
        
        # Final projection to output dimension
        x = self.last_layer(x)
        
        return x
    def print_summary(self, input_size=(1, 256)):
        """
        Print model summary similar to torchsummary
        """
        print("=" * 80)
        print("DINO MLP Head Summary")
        print("=" * 80)
        print(f"Input size: {input_size}")
        print(f"Output size: {input_size[0]}, {self.last_layer.out_features}")
        print("-" * 80)
        
        total_params = 0
        trainable_params = 0
        
        print(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<15} {'Trainable':<10}")
        print("=" * 80)
        
        # Create a dummy input
        x = torch.randn(input_size)
        
        # Track layers
        layers_info = []
        
        # Process through MLP
        for i, layer in enumerate(self.mlp):
            if isinstance(layer, nn.Linear):
                layer_type = f"Linear_{i//3+1}"
                param_count = sum(p.numel() for p in layer.parameters())
                total_params += param_count
                trainable_params += param_count
                
                # Get output shape
                x = layer(x)
                output_shape = list(x.shape)
                
                layers_info.append({
                    'name': layer_type,
                    'output_shape': output_shape.copy(),
                    'params': param_count,
                    'trainable': True
                })
            elif isinstance(layer, nn.LayerNorm):
                layer_type = f"LayerNorm_{i//3+1}"
                param_count = sum(p.numel() for p in layer.parameters())
                total_params += param_count
                trainable_params += param_count
                
                # Get output shape
                x = layer(x)
                output_shape = list(x.shape)
                
                layers_info.append({
                    'name': layer_type,
                    'output_shape': output_shape.copy(),
                    'params': param_count,
                    'trainable': True
                })
            elif isinstance(layer, nn.GELU):
                layer_type = f"GELU_{i//3+1}"
                param_count = 0
                
                # Get output shape
                x = layer(x)
                output_shape = list(x.shape)
                
                layers_info.append({
                    'name': layer_type,
                    'output_shape': output_shape.copy(),
                    'params': param_count,
                    'trainable': False
                })
        
        # Process last layer
        layer_type = "Last_Linear"
        param_count = sum(p.numel() for p in self.last_layer.parameters())
        total_params += param_count
        trainable_params += param_count
        
        x = self.last_layer(x)
        output_shape = list(x.shape)
        
        layers_info.append({
            'name': layer_type,
            'output_shape': output_shape.copy(),
            'params': param_count,
            'trainable': True
        })
        
        # Print all layers
        for info in layers_info:
            print(f"{info['name']:<25} {str(info['output_shape']):<20} {info['params']:<15} {info['trainable']:<10}")
        
        print("=" * 80)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")
        print("=" * 80)
        
        return x.shape  # Return final output shape


# Test the model
if __name__ == "__main__":
    # Create model instance
    in_dim = 256
    out_dim = 1024
    model = DINO_MLP_HD(in_dim, out_dim, n_layers=4, use_layer_norm=True)
    
    # Print model summary
    print("Model Architecture:")
    print(model)
    print("\n" + "="*50 + "\n")
    
    # Print detailed summary
    final_shape = model.print_summary(input_size=(1, in_dim))
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    test_input = torch.randn(batch_size, in_dim)
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"Output mean: {output.mean().item():.4f}")
    print(f"Output std: {output.std().item():.4f}")