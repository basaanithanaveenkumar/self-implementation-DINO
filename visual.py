import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import argparse
import os
from torchvision import transforms
import glob
from PIL import Image
from model.vision_transformer import VisionTransformerWrapper

# Add this to your VisionTransformerWrapper class to extract attention maps
def add_attention_extraction(model):
    """
    Monkey patch the VisionTransformerWrapper to extract attention maps
    """
    def hook_fn(module, input, output):
        # Store the attention weights
        model.attention_maps = output[1] if len(output) > 1 else None
    
    # Register hook to the attention layer
    for name, module in model.named_modules():
        if hasattr(module, 'attention') and module.attention is not None:
            module.attention.register_forward_hook(hook_fn)
            break
    
    return model

def visualize_attention_maps(model, image_path, output_path, device='cuda'):
    """
    Visualize attention maps for a given image and create a video
    
    Args:
        model: Trained DINO model with attention extraction
        image_path: Path to input image
        output_path: Path to save output video
        device: Device to run inference on
    """
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    
    # Preprocess image for model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to PIL image for transformation
    pil_image = Image.fromarray(image)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    # Get attention maps from model
    model.eval()
    with torch.no_grad():
        # Forward pass to get attention maps
        _, attentions = model(input_tensor,return_attention=True)
        print(attentions.keys())
        #attentions = model.attention_maps
    
    if attentions is None:
        print("No attention maps found in the model")
        return
    
    # Process attention maps
    # attentions shape: [batch_size, num_heads, num_patches, num_patches]
    # We'll use the CLS token attention to all patches
    attentions = attentions["blocks.11.attn.softmax"][0]  # Remove batch dimension
    nh = attentions.shape[0]  # Number of attention heads
    
    # Average attention across heads for a general view
    avg_attention = torch.mean(attentions, dim=0)
    
    # Get the attention from the CLS token to all other tokens
    cls_attention = avg_attention[0, 1:]  # Skip the CLS token itself
    
    # Reshape to 2D grid (assuming square patches)
    grid_size = int(np.sqrt(cls_attention.shape[0]))
    attention_map = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
    
    # Resize attention map to match original image
    attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
    
    # Normalize attention map
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # Create figure for visualization
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show original image
    axs[0].imshow(original_image)
    axs[0].set_title('Original Image', fontsize=14)
    axs[0].axis('off')
    
    # Show attention map overlaid on image
    axs[1].imshow(original_image)
    im = axs[1].imshow(attention_map, cmap='jet', alpha=0.5)
    axs[1].set_title('Attention Map (CLS Token)', fontsize=14)
    axs[1].axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path.replace('.mp4', '.png'), dpi=150, bbox_inches='tight')
    
    # Create video with attention maps from all heads
    if nh > 1:
        create_attention_video(attentions, original_image, output_path, nh)
    
    print(f"Attention visualization saved to {output_path}")

def create_attention_video(attentions, original_image, output_path, num_heads):
    """
    Create a video showing attention maps from all heads
    
    Args:
        attentions: Attention maps from all heads
        original_image: Original input image
        output_path: Path to save the video
        num_heads: Number of attention heads
    """
    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 2
    height, width = original_image.shape[0], original_image.shape[1]
    video_width = width * 2
    video_height = height
    out = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))
    
    for head_idx in range(num_heads):
        # Get attention for this head
        head_attention = attentions[head_idx]
        cls_attention = head_attention[0, 1:]  # CLS token attention to all other tokens
        
        # Reshape to 2D grid
        grid_size = int(np.sqrt(cls_attention.shape[0]))
        attention_map = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
        
        # Resize attention map to match original image
        attention_map = cv2.resize(attention_map, (width, height))
        
        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # Apply colormap
        attention_colored = cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Overlay attention on original image
        overlay = cv2.addWeighted(original_image, 0.6, attention_colored, 0.4, 0)
        
        # Convert back to BGR for video writing
        original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        
        # Combine images side by side
        combined = np.hstack((original_bgr, overlay_bgr))
        
        # Add text
        cv2.putText(combined, f"Head {head_idx+1}/{num_heads}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame to video
        out.write(combined)
    
    out.release()

def create_attention_visualizations(checkpoint_path, image_dir, output_dir, device='cuda'):
    """
    Create attention visualization for all images in a directory
    
    Args:
        checkpoint_path: Path to model checkpoint
        image_dir: Directory containing input images
        output_dir: Directory to save output videos
        device: Device to run inference on
    """
    # Load model from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model
    model = VisionTransformerWrapper(
        model_name="vit_small_patch16_224",
        img_size=224,
        pretrained=False,
        is_teacher=True
    )
    
    # Load weights
    model.load_state_dict(checkpoint['teacher_state_dict'])
    
    # Add attention extraction capability
    model = add_attention_extraction(model)
    model.to(device)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image in directory
    image_dir = Path(image_dir)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(str(image_dir / ext)))
    
    print(f"Found {len(image_paths)} images to process")
    
    for image_path in image_paths:
        print(f"Processing {image_path}")
        output_path = output_dir / f"{Path(image_path).stem}_attention.mp4"
        visualize_attention_maps(model, image_path, str(output_path), device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DINO Attention Visualization')
    parser.add_argument('--checkpoint', type=str, default= "/content/dino_coco_checkpoints/best_checkpoint.pth",
                       help='Path to model checkpoint')
    parser.add_argument('--image_dir', type=str, default="/content/object-detection-BBD/data/100k/test",
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default="./attention_visualizations",
                       help='Directory to save output videos')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    else:
        device = args.device
    
    # Create attention visualizations
    create_attention_visualizations(
        checkpoint_path=args.checkpoint,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        device=args.device
    )