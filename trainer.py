import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
import logging
from pathlib import Path
from model.vision_transformer import *
from model.dino_loss import *
#vibe codd using deepseek

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DINO_Trainer")

class DINOTrainer:
    def __init__(self, student_model, teacher_model, dataloader, 
                 loss_fn, optimizer, device, out_dir="./dino_checkpoints",
                 warmup_epochs=10, total_epochs=100, save_freq=10,
                 use_amp=True, base_lr=0.0005):  # Added base_lr parameter
        """
        DINO trainer for self-supervised learning
        
        Args:
            student_model: Student model (trainable)
            teacher_model: Teacher model (EMA of student)
            dataloader: DataLoader with multi-crop images
            loss_fn: DINO loss function
            optimizer: Optimizer for student model
            device: Training device (cuda/cpu)
            out_dir: Directory to save checkpoints
            warmup_epochs: Number of warmup epochs for learning rate
            total_epochs: Total training epochs
            save_freq: Frequency of saving checkpoints
            use_amp: Whether to use automatic mixed precision
            base_lr: Base learning rate for scheduling
        """
        self.student = student_model
        self.teacher = teacher_model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.out_dir = Path(out_dir)
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.save_freq = save_freq
        self.use_amp = use_amp
        self.base_lr = base_lr  # Store base learning rate
        
        # Create output directory
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Learning rate scheduler
        self.lr_schedule = self._get_lr_schedule()
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=use_amp)
        
        # Move models to device
        self.student.to(device)
        self.teacher.to(device)
        
        # Set teacher to eval mode
        self.teacher.eval()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Logging
        logger.info(f"DINO Trainer initialized on device: {device}")
        logger.info(f"Using mixed precision: {use_amp}")
        logger.info(f"Base learning rate: {base_lr}")
    
    def _get_lr_schedule(self):
        """Create learning rate schedule with warmup"""
        def lr_schedule(step):
            # Warmup for the first warmup_steps
            warmup_steps = self.warmup_epochs * len(self.dataloader)
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            else:
                # Cosine decay after warmup
                total_steps = self.total_epochs * len(self.dataloader)
                return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        
        return lr_schedule
    
    def update_teacher(self, momentum=0.996):
        """Update teacher model with EMA of student weights"""
        with torch.no_grad():
            for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
                param_t.data.mul_(momentum).add_((1 - momentum) * param_s.data)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.student.train()
        
        epoch_loss = 0
        num_batches = len(self.dataloader)
        
        for batch_idx, crops in enumerate(self.dataloader):
            # Move crops to device
            crops = [crop.to(self.device, non_blocking=True) for crop in crops]
            
            # Update learning rate
            self._adjust_learning_rate(self.global_step)
            lr = self.optimizer.param_groups[0]["lr"]
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                # Student forward pass (all crops)
                student_outputs = []
                for crop in crops:
                    student_outputs.append(self.student(crop))
                
                # Teacher forward pass (only global crops)
                teacher_outputs = []
                with torch.no_grad():
                    for crop in crops[:2]:  # First two are global crops
                        teacher_outputs.append(self.teacher(crop))
                
                # Compute loss
                print(len(student_outputs),"len of student outputs")
                print(len(teacher_outputs), "len of teacher outputs")
                loss = self.loss_fn(student_outputs, teacher_outputs, self.epoch)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Update teacher with EMA
            self.update_teacher()
            
            # Update metrics
            epoch_loss += loss.item()
            self.global_step += 1
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.info(
                    f"Epoch {self.epoch}/{self.total_epochs} | "
                    f"Batch {batch_idx}/{num_batches} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {lr:.6f}"
                )
        
        return epoch_loss / num_batches
    
    def _adjust_learning_rate(self, step):
        """Adjust learning rate based on schedule"""
        # Calculate the multiplier from the schedule
        multiplier = self.lr_schedule(step)
        
        # Set the learning rate for all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.base_lr * multiplier
    
    def train(self):
        """Main training loop"""
        logger.info("Starting DINO training...")
        start_time = time.time()
        
        for epoch in range(self.epoch, self.total_epochs):
            self.epoch = epoch
            
            # Train for one epoch
            epoch_loss = self.train_epoch()
            
            # Log epoch results
            logger.info(
                f"Epoch {epoch}/{self.total_epochs} | "
                f"Avg Loss: {epoch_loss:.4f} | "
                f"Time: {time.time() - start_time:.2f}s"
            )
            
            # Save checkpoint
            if epoch % self.save_freq == 0 or epoch == self.total_epochs - 1:
                self.save_checkpoint(epoch_loss)
            
            # Update best loss
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_checkpoint(epoch_loss, is_best=True)
        
        logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    def save_checkpoint(self, loss, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'student_state_dict': self.student.state_dict(),
            'teacher_state_dict': self.teacher.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'loss': loss,
            'best_loss': self.best_loss,
            'base_lr': self.base_lr,  # Save base learning rate
        }
        
        # Save regular checkpoint
        checkpoint_path = self.out_dir / f"checkpoint_epoch_{self.epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.out_dir / "best_checkpoint.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"New best checkpoint saved with loss: {loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.base_lr = checkpoint.get('base_lr', 0.0005)  # Load base learning rate
        
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.teacher.load_state_dict(checkpoint['teacher_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.use_amp and checkpoint['scaler_state_dict'] is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Loaded checkpoint from epoch {self.epoch} with loss {checkpoint['loss']:.4f}")

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models (using your VisionTransformerWrapper)
    model_name = "vit_small_patch16_224"
    img_size = 224
    std_img_size = 224
    out_dim = 1024
    
    # Student model (trainable)
    student_model = VisionTransformerWrapper(
        model_name=model_name,
        img_size=std_img_size,
        pretrained=True,
        is_teacher=False
    )
    
    # Teacher model (frozen, updated via EMA)
    teacher_model = VisionTransformerWrapper(
        model_name=model_name,
        img_size=img_size,
        pretrained=True,
        is_teacher=True
    )
    # Example configuration (adjust values based on your setup)
    ncrops = 6  # 2 global crops + 4 local crops
    warmup_teacher_temp = 0.04
    teacher_temp = 0.07
    warmup_teacher_temp_epochs = 30
    nepochs = 100

    # Initialize DINOLoss with the required arguments
    loss_fn = DINOLoss(
        #ncrops=ncrops,
        warmup_teacher_temp=warmup_teacher_temp,
        teacher_temp=teacher_temp,
        warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
        nepochs=nepochs,
        out_dim=1024
    )
    
    # Create COCO data loader
    dataDir = "/content/object-detection-BBD/data/100k/test"
    annFile = f"{dataDir}/_annotations.coco.json"
    dataloader = create_coco_dataloader(
        annFile=annFile,
        dataDir=dataDir,
        batch_size=4,
        num_workers=4,
        global_crop_size=224,
        local_crop_size=96,
    )
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=0.0005,
        weight_decay=0.04
    )
    
    # Initialize trainer
    trainer = DINOTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        out_dir="./dino_coco_checkpoints",
        warmup_epochs=10,
        total_epochs=100,
        save_freq=10,
        use_amp=True,
        # use_ddp=False  # Set to True for multi-GPU training
    )
    
    # Start training
    trainer.train()