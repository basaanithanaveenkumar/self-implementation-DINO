import torch
import torch.nn as nn
import torch.nn.functional as F

# vibe coded from deepseek

class DINOLoss(nn.Module):
    """
    DINO loss function implementation.
    
    This loss function implements the self-distillation with no labels approach
    used in the DINO paper. It consists of:
    1. Cross-entropy loss between student and teacher outputs
    2. Centering of teacher outputs to avoid collapse
    3. Sharpening of teacher distributions with temperature
    
    Args:
        out_dim (int): Output dimension of the projection head
        warmup_teacher_temp (float): Initial teacher temperature
        teacher_temp (float): Final teacher temperature (after warmup)
        warmup_teacher_temp_epochs (int): Number of warmup epochs for teacher temperature
        nepochs (int): Total number of epochs
        student_temp (float): Student temperature
        center_momentum (float): Momentum for center update
    """
    
    def __init__(self, out_dim, warmup_teacher_temp=0.04, teacher_temp=0.07,
                 warmup_teacher_temp_epochs=30, nepochs=100, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.nepochs = nepochs
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Register buffer for center
        self.register_buffer("center", torch.zeros(1, out_dim,device=device))
        
        # Temperature scheduling
        self.warmup_teacher_temp = warmup_teacher_temp
        self.teacher_temp = teacher_temp
        
    def forward(self, student_output, teacher_output, epoch):
        """
        Forward pass of the DINO loss.
        
        Args:
            student_output: List of student outputs for different crops
            teacher_output: List of teacher outputs for different crops
            epoch: Current epoch number (for temperature scheduling)
            
        Returns:
            Loss value
        """
        # Get current teacher temperature (with warmup)
        teacher_temp = self.get_teacher_temp(epoch)
        
        # Gather all outputs
        student_out = self.gather_outputs(student_output)
        teacher_out = self.gather_outputs(teacher_output).detach()  # Detach early
        
        # Apply temperature to student outputs
        if isinstance(student_output, list):
            student_output = torch.cat(student_output, dim=0)
            teacher_output = torch.cat(teacher_output, dim=0)
        student_out = student_out / self.student_temp
        
        # Apply temperature and center to teacher outputs
        teacher_out = (teacher_out - self.center) / teacher_temp
        teacher_out = F.softmax(teacher_out, dim=-1)
        
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss
    
    def get_teacher_temp(self, epoch):
        """Get teacher temperature with warmup schedule"""
        if epoch < self.warmup_teacher_temp_epochs:
            # Linear warmup
            return self.warmup_teacher_temp + (self.teacher_temp - self.warmup_teacher_temp) * \
                   epoch / self.warmup_teacher_temp_epochs
        else:
            return self.teacher_temp
    
    def gather_outputs(self, outputs):
        """
        Gather outputs from all crops and concatenate them.
        """
        return torch.cat([output for output in outputs], dim=0)
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output centering.
        """
        # Gather all teacher outputs (raw, before temperature/softmax)
        teacher_out = self.gather_outputs(teacher_output)
        
        # Calculate batch mean
        batch_center = torch.mean(teacher_out, dim=0, keepdim=True)
        
        # Update center with momentum
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)