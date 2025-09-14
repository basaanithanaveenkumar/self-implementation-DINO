# Now let's define the VisionTransformerWrapper with a fix for the parameter freezing
from .dino_mlp import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from bbd_dataset import create_coco_dataloader
from timm.utils import AttentionExtract

timm.layers.set_fused_attn(False)

class VisionTransformerWrapper(nn.Module):
    """
    Wrapper for Vision Transformer backbones
    """
    def __init__(self, model_name: str, img_size: int = 224, pretrained: bool = True, is_teacher=True):
        super().__init__()
        self.raw_backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            img_size=img_size,
            #num_classes=0  # Remove classification head
        )
        
        # Get feature dimension
        if hasattr(self.raw_backbone, 'num_features'):
            self.feature_dim = self.raw_backbone.num_features
        else:
            # For ViT models, use embed_dim
            self.feature_dim = self.raw_backbone.embed_dim
        
        # Get input dimension from the backbone
        input_vec_dim = self.feature_dim
        
        # Remove the classification head
        layers = list(self.raw_backbone.children())[:-1]
        self.backbone = nn.Sequential(*layers)
        
        # Freeze parameters if it's a teacher
        for param in self.backbone.parameters():
            if is_teacher:
                param.requires_grad = False
            else:
                param.requires_grad = True
                
        # Create the DINO head
        self.dino_mlp_head = DINO_MLP_HD(
            in_dim=input_vec_dim,
            out_dim=1024,
            hidden_dim=2048,
            bottleneck_dim=256,
            n_layers=5,
            use_layer_norm=True
        )
        self.extractor = AttentionExtract(self.raw_backbone, method='fx')

    def forward(self, x,return_attention=False):
        vis_features = self.backbone(x)
        attention_outputs = self.extractor(x)
        # Assuming the first token is the class token
        cls_token = vis_features[:, 0]
        print(f"Class token shape: {cls_token.shape}")
        x = self.dino_mlp_head(cls_token)
        if return_attention:
          # Return both output and attention maps
          return F.normalize(x, dim=-1, p=2), attention_outputs
        else:
          return F.normalize(x, dim=-1, p=2)