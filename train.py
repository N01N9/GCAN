import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
import argparse, os, random, numpy as np
from tqdm import tqdm
from datetime import datetime
import math

# Import modules
from model import RefinedMultiStreamGCAN
from loss import GCANLoss, DiarizationMetrics
from dataset import create_dataloaders 


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f"✓ Seed fixed to: {seed}")


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """
    Create a schedule with a learning rate that:
    - Increases linearly from 0 to initial_lr during warmup
    - Decreases following a cosine curve after warmup
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # Cosine annealing
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        # Calculate total training steps
        total_steps = config['epochs'] * config['steps_per_epoch']
        warmup_steps = config.get('warmup_steps', min(1000, total_steps // 10))
        
        # Warm-up + Cosine Annealing Scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=0.01
        )
        
        # Gradient accumulation
        self.grad_accumulation_steps = config.get('grad_accumulation_steps', 1)
        
        # Output directory setup
        self.output_dir = Path(config['output_dir']) / f"GCAN_{datetime.now().strftime('%m%d_%H%M')}"
        if config['resume']:
            resume_path = Path(config['resume']).parent.parent
            if resume_path.exists(): 
                self.output_dir = resume_path

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.output_dir / "logs")
        
        self.best_loss = float('inf')
        self.global_step = 0
        self.start_epoch = 0
        self.patience_counter = 0
        self.early_stop_patience = config.get('early_stop_patience', 15)
        
        self.train_metrics = DiarizationMetrics()
        self.val_metrics = DiarizationMetrics()

        # Resume from checkpoint
        if config['resume']:
            self._load_checkpoint(config['resume'])
            
        # Log configuration
        self._log_config()

    def _log_config(self):
        """Log training configuration"""
        print("\n" + "="*50)
        print("Training Configuration")
        print("="*50)
        print(f"  Output Dir: {self.output_dir}")
        print(f"  Epochs: {self.config['epochs']}")
        print(f"  Steps/Epoch: {self.config['steps_per_epoch']}")
        print(f"  Batch Size: {self.config['batch_size']}")
        print(f"  Grad Accumulation: {self.grad_accumulation_steps}")
        print(f"  Effective Batch: {self.config['batch_size'] * self.grad_accumulation_steps}")
        print(f"  Learning Rate: {self.config['learning_rate']}")
        print(f"  Warmup Steps: {min(1000, self.config['epochs'] * self.config['steps_per_epoch'] // 10)}")
        print(f"  Early Stop Patience: {self.early_stop_patience}")
        print("="*50 + "\n")

    def _save_checkpoint(self, epoch, is_best=False, step=None):
        """Save checkpoint with all training state"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'patience_counter': self.patience_counter,
            'config': self.config
        }
        
        if step is not None:
            save_path = self.output_dir / f"checkpoint_step_{step}.pt"
        else:
            save_path = self.output_dir / "last.pt"
            
        torch.save(checkpoint, save_path)
        
        if is_best:
            torch.save(checkpoint, self.output_dir / "best.pt")

    def _load_checkpoint(self, path):
        """Resume from checkpoint"""
        print(f"➜ Resuming from checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
        print(f"✓ Resume complete: Epoch {self.start_epoch}, Step {self.global_step}")

    def _check_for_nan(self, outputs, loss):
        """NaN detection with detailed logging"""
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️ WARNING: Loss is {loss.item()} at step {self.global_step}")
            self._save_checkpoint(epoch=-1, step=f"nan_{self.global_step}")
            raise RuntimeError(f"Training diverged! Loss became {loss.item()}")
        
        for key, val in outputs.items():
            if isinstance(val, torch.Tensor):
                if torch.isnan(val).any() or torch.isinf(val).any():
                    print(f"\n⚠️ WARNING: NaN/Inf in {key} at step {self.global_step}")
                    self._save_checkpoint(epoch=-1, step=f"nan_{self.global_step}")
                    raise RuntimeError(f"NaN/Inf detected in {key}")

    def train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset() 
        epoch_loss = 0
        epoch_losses = {'assign': 0, 'exist': 0, 'ortho': 0, 'contrastive': 0, 'overlap': 0}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Train", total=self.config['steps_per_epoch'])
        
        step_count = 0
        accumulated_loss = 0
        
        for batch_idx, batch in enumerate(pbar):
            if step_count >= self.config['steps_per_epoch']: 
                break
            
            audio = batch['audio'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            outputs = self.model(audio)
            loss, l_dict = self.criterion(outputs, targets)
            
            # Check for NaN before backward
            self._check_for_nan(outputs, loss)
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / self.grad_accumulation_steps
            scaled_loss.backward()
            
            accumulated_loss += loss.item()
            
            # Update weights after accumulation
            if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Skip update if gradients are too large
                if grad_norm > 10.0:
                    print(f"\n⚠️ Large gradient {grad_norm:.2f} at step {self.global_step}, skipping")
                    self.optimizer.zero_grad()
                    accumulated_loss = 0
                    continue
                
                self.optimizer.step()
                self.scheduler.step()  # Step scheduler every optimization step
                self.optimizer.zero_grad()
                
                epoch_loss += accumulated_loss
                for key in epoch_losses:
                    if key in l_dict:
                        epoch_losses[key] += l_dict[key]
                
                # TensorBoard logging
                if self.global_step % 10 == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('Loss/train_total', loss.item(), self.global_step)
                    self.writer.add_scalar('Loss/train_assign', l_dict['assign'], self.global_step)
                    self.writer.add_scalar('Loss/train_exist', l_dict['exist'], self.global_step)
                    self.writer.add_scalar('Loss/train_ortho', l_dict['ortho'], self.global_step)
                    self.writer.add_scalar('Loss/train_contrastive', l_dict.get('contrastive', 0), self.global_step)
                    self.writer.add_scalar('Loss/train_overlap', l_dict.get('overlap', 0), self.global_step)
                    self.writer.add_scalar('Acc/train_frame', l_dict['frame_acc'], self.global_step)
                    self.writer.add_scalar('Acc/train_spk_num', l_dict['spk_num_acc'], self.global_step)
                    self.writer.add_scalar('Gradient/norm', grad_norm.item(), self.global_step)
                    self.writer.add_scalar('LR/learning_rate', lr, self.global_step)
                    
                    # Log temperature if available
                    if 'temperature' in outputs:
                        self.writer.add_scalar('Model/temperature', outputs['temperature'].item(), self.global_step)
                
                # Checkpoint every 2000 steps
                if self.global_step > 0 and self.global_step % 2000 == 0:
                    self._save_checkpoint(epoch, step=self.global_step)
                    print(f"\n★ Step {self.global_step}: Checkpoint saved.")

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'acc': f"{l_dict['frame_acc']:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                self.global_step += 1
                step_count += 1
                accumulated_loss = 0
            
        return epoch_loss / max(1, step_count)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        self.val_metrics.reset()
        val_loss = 0
        
        for i, batch in enumerate(self.val_loader):
            if i >= self.config['val_steps']: 
                break
                
            audio = batch['audio'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            outputs = self.model(audio)
            loss, l_dict = self.criterion(outputs, targets)
            
            val_loss += loss.item()
            self.val_metrics.update(outputs, targets, l_dict)
            
        avg_val_loss = val_loss / self.config['val_steps']
        val_m = self.val_metrics.compute()
        
        self.writer.add_scalar('Loss/val_total', avg_val_loss, self.global_step)
        self.writer.add_scalar('Acc/val_spk_num', val_m['spk_num_acc'], self.global_step)
        self.writer.add_scalar('Acc/val_frame', val_m['frame_acc'], self.global_step)
        
        return avg_val_loss, val_m

    def run(self):
        print(f"✓ Training started! Logs: {self.output_dir / 'logs'}")
        try:
            for epoch in range(self.start_epoch, self.config['epochs']):
                t_loss = self.train_epoch(epoch)
                v_loss, v_m = self.validate(epoch)
                
                lr = self.optimizer.param_groups[0]['lr']
                print(f"E{epoch} | Train: {t_loss:.4f} | Val: {v_loss:.4f} | "
                      f"SpkAcc: {v_m['spk_num_acc']:.4f} | FrmAcc: {v_m['frame_acc']:.4f} | LR: {lr:.2e}")
                
                # Best model saving
                is_best = v_loss < self.best_loss
                if is_best:
                    self.best_loss = v_loss
                    self.patience_counter = 0
                    print(f"★ New best! Val Loss: {v_loss:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"  No improvement ({self.patience_counter}/{self.early_stop_patience})")
                
                self._save_checkpoint(epoch, is_best=is_best)
                
                # Early stopping
                if self.patience_counter >= self.early_stop_patience:
                    print(f"\n⏹ Early stopping triggered at epoch {epoch}")
                    break
                    
        except RuntimeError as e:
            print(f"\n❌ Training stopped: {e}")
            print(f"Emergency checkpoint saved in {self.output_dir}")
        finally:
            self.writer.close()
            print(f"\n✓ Training complete! Best loss: {self.best_loss:.4f}")


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    tl, vl = create_dataloaders(
        args.repo_id, args.batch_size, args.num_workers, 
        args.val_ratio, 16000, args.audio_length
    )
    
    # Initialize model with new parameters
    model = RefinedMultiStreamGCAN(
        num_speakers=args.num_slots, 
        hidden_dim=args.hidden_dim,
        num_transformer_layers=args.num_transformer_layers,
        dropout=args.dropout
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Enhanced loss function
    criterion = GCANLoss(
        lambda_existence=args.lambda_existence,
        lambda_ortho=args.lambda_ortho,
        lambda_contrastive=args.lambda_contrastive,
        lambda_overlap=args.lambda_overlap,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma
    ).to(device)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Create config dict
    config = vars(args)
    
    # Run training
    Trainer(model, tl, vl, criterion, optimizer, device, config).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCAN Training Script')
    
    # Data arguments
    parser.add_argument('--repo_id', type=str, required=True, help='HuggingFace repo ID')
    parser.add_argument('--output_dir', type=str, default='./exp')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint.pt')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--steps_per_epoch', type=int, default=10000)
    parser.add_argument('--val_steps', type=int, default=1000)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--grad_accumulation_steps', type=int, default=1)
    parser.add_argument('--early_stop_patience', type=int, default=15)
    
    # Model arguments
    parser.add_argument('--num_slots', type=int, default=6)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_transformer_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Loss arguments
    parser.add_argument('--lambda_existence', type=float, default=1.0)
    parser.add_argument('--lambda_ortho', type=float, default=0.1)
    parser.add_argument('--lambda_contrastive', type=float, default=0.1)
    parser.add_argument('--lambda_overlap', type=float, default=0.5)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    
    # Data arguments
    parser.add_argument('--audio_length', type=float, default=20.0)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    main(parser.parse_args())