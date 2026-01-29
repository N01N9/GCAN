import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha balancing
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ContrastiveLoss(nn.Module):
    """Contrastive Loss for speaker embedding separation"""
    def __init__(self, margin=1.0, temperature=0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, attractors, exist_targets):
        """
        Args:
            attractors: (B, K, D) speaker embeddings
            exist_targets: (B, K) binary mask for existing speakers
        """
        B, K, D = attractors.shape
        
        # Normalize embeddings
        attractors_norm = F.normalize(attractors, p=2, dim=-1)
        
        total_loss = 0.0
        count = 0
        
        for b in range(B):
            # Get indices of existing speakers
            exist_mask = exist_targets[b] > 0.5
            n_exist = exist_mask.sum().item()
            
            if n_exist < 2:
                continue
                
            exist_idx = torch.where(exist_mask)[0]
            exist_emb = attractors_norm[b, exist_idx]  # (n, D)
            
            # Compute pairwise similarities
            sim_matrix = torch.mm(exist_emb, exist_emb.t())  # (n, n)
            
            # Push apart different speakers (off-diagonal should be low)
            mask = ~torch.eye(n_exist, device=sim_matrix.device, dtype=torch.bool)
            off_diag_sim = sim_matrix[mask]
            
            # Contrastive loss: push apart
            loss = F.relu(off_diag_sim - (-self.margin)).mean()
            total_loss += loss
            count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=attractors.device)
        return total_loss / count


class OverlapLoss(nn.Module):
    """Loss for detecting overlapping speech regions"""
    def __init__(self, pos_weight=2.0):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, overlap_logits, overlap_targets):
        """
        Args:
            overlap_logits: (B, T) predicted overlap intensity
            overlap_targets: (B, T) ground truth overlap regions
        """
        # Simple MSE loss for overlap intensity
        loss = F.mse_loss(overlap_logits, overlap_targets)
        return loss


class GCANLoss(nn.Module):
    """Enhanced GCAN Loss with Focal Loss, Contrastive Loss, and Label Smoothing"""
    def __init__(
        self, 
        lambda_existence=1.0, 
        lambda_ortho=0.1, 
        lambda_contrastive=0.1,
        lambda_overlap=0.5,
        pos_weight=5.0,
        label_smoothing=0.1,
        focal_gamma=2.0,
        focal_alpha=0.25
    ):
        super().__init__()
        self.lambda_existence = lambda_existence
        self.lambda_ortho = lambda_ortho
        self.lambda_contrastive = lambda_contrastive
        self.lambda_overlap = lambda_overlap
        self.label_smoothing = label_smoothing
        
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        
        # Loss components
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, reduction='none')
        self.contrastive_loss = ContrastiveLoss(margin=0.5, temperature=0.1)
        self.overlap_loss = OverlapLoss(pos_weight=2.0)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction="none")
    
    def _apply_label_smoothing(self, targets):
        """Apply label smoothing: 0 -> eps, 1 -> 1-eps"""
        eps = self.label_smoothing
        return targets * (1 - eps) + (1 - targets) * eps
    
    def forward(self, outputs, targets):
        logits = outputs["assignments"]  
        if torch.isnan(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0)    
        existence = outputs["existence"]    
        attractors = outputs["attractors"]
        gt_labels = targets["speaker_labels"] 
        overlap_targets = targets.get("overlap_regions", None)

        device = logits.device
        B, T, K = logits.shape
        N = gt_labels.shape[2]

        # Align temporal dimension
        if T != gt_labels.size(1):
            logits = F.interpolate(
                logits.transpose(1, 2), 
                size=gt_labels.size(1), 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
            T = gt_labels.size(1)
            
            # Also interpolate overlap_logits if present
            if 'overlap_logits' in outputs:
                outputs['overlap_logits'] = F.interpolate(
                    outputs['overlap_logits'].unsqueeze(1),
                    size=T,
                    mode='linear',
                    align_corners=False
                ).squeeze(1)

        total_assign_loss, total_exist_loss = 0.0, 0.0
        correct_frames, total_frames = 0, 0
        correct_spk = 0
        exist_targets_batch = torch.zeros(B, K, device=device)

        for b in range(B):
            with torch.no_grad():
                cost = torch.zeros(K, N, device=device)
                for k in range(K):
                    for n in range(N):
                        c = self.bce(logits[b, :, k], gt_labels[b, :, n]).mean()
                        cost[k, n] = c

            # Handle NaN/Inf in cost matrix
            if not torch.isfinite(cost).all():
                cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=100.0)
                print(f"⚠️ Warning: Invalid entries in cost matrix at batch {b}")

            # Hungarian Matching
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

            # Apply label smoothing to matched labels
            matched_labels = gt_labels[b, :, col_ind]
            if self.label_smoothing > 0:
                matched_labels_smooth = self._apply_label_smoothing(matched_labels)
            else:
                matched_labels_smooth = matched_labels
            
            matched_logits = logits[b, :, row_ind]
            
            # Use Focal Loss for assignment loss
            focal_loss_val = self.focal_loss(matched_logits, matched_labels_smooth)
            total_assign_loss += focal_loss_val.mean()
            
            with torch.no_grad():
                preds = (torch.sigmoid(matched_logits) > 0.5).float()
                correct_frames += (preds == matched_labels).sum().item()
                total_frames += (T * len(row_ind))

            # Existence Loss
            exist_target = torch.zeros(K, device=device)
            exist_target[row_ind] = 1.0
            exist_targets_batch[b] = exist_target
            
            if self.label_smoothing > 0:
                exist_target_smooth = self._apply_label_smoothing(exist_target)
            else:
                exist_target_smooth = exist_target
                
            total_exist_loss += F.binary_cross_entropy_with_logits(
                existence[b], exist_target_smooth, 
                pos_weight=self.pos_weight
            )
            
            with torch.no_grad():
                pred_spk_num = (torch.sigmoid(existence[b]) > 0.5).sum().item()
                gt_spk_num = targets['num_speakers'][b].item()
                if pred_spk_num == gt_spk_num:
                    correct_spk += 1

        # Orthogonality Loss
        att = F.normalize(attractors, p=2, dim=-1)
        sim = torch.bmm(att, att.transpose(1, 2))
        ortho_loss = ((sim - torch.eye(K, device=device).unsqueeze(0))**2).mean()

        # Contrastive Loss
        contrastive_loss = self.contrastive_loss(attractors, exist_targets_batch)
        
        # Overlap Loss
        overlap_loss = torch.tensor(0.0, device=device)
        if 'overlap_logits' in outputs and overlap_targets is not None:
            overlap_loss = self.overlap_loss(outputs['overlap_logits'], overlap_targets)

        # Total Loss
        total_loss = (
            (total_assign_loss / B) + 
            (self.lambda_existence * total_exist_loss / B) + 
            (self.lambda_ortho * ortho_loss) +
            (self.lambda_contrastive * contrastive_loss) +
            (self.lambda_overlap * overlap_loss)
        )

        return total_loss, {
            "loss": total_loss.item(),
            "assign": (total_assign_loss / B).item(),
            "exist": (total_exist_loss / B).item(),
            "ortho": ortho_loss.item(),
            "contrastive": contrastive_loss.item(),
            "overlap": overlap_loss.item(),
            "frame_acc": correct_frames / max(1, total_frames),
            "spk_num_acc": correct_spk / B
        }
    

class DiarizationMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.correct_spk = 0
        self.total_samples = 0
        self.total_frame_acc = 0.0
        self.steps = 0

    def update(self, outputs, targets, l_dict=None):
        # Speaker count accuracy
        probs = torch.sigmoid(outputs['existence']).detach().cpu()
        pred_num = (probs > self.threshold).sum(dim=1).long()
        gt_num = targets['num_speakers'].detach().cpu().long()
        self.correct_spk += (pred_num == gt_num).sum().item()
        self.total_samples += gt_num.size(0)

        # Frame accuracy accumulation
        if l_dict and 'frame_acc' in l_dict:
            self.total_frame_acc += l_dict['frame_acc']
            self.steps += 1

    def compute(self):
        return {
            'spk_num_acc': self.correct_spk / max(1, self.total_samples),
            'frame_acc': self.total_frame_acc / max(1, self.steps)
        }