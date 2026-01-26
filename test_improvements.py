#!/usr/bin/env python3
"""Test script to verify GCAN model improvements"""

import torch
import torch.nn.functional as F
import sys

def test_model_architecture():
    """Test enhanced model architecture"""
    print("=" * 50)
    print("Testing Model Architecture")
    print("=" * 50)
    
    from model import RefinedMultiStreamGCAN
    
    # Create model with new parameters
    model = RefinedMultiStreamGCAN(
        num_speakers=6, 
        hidden_dim=256,
        num_transformer_layers=6,
        dropout=0.1
    )
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    audio_length = 32000  # 2 seconds
    x = torch.randn(batch_size, audio_length)
    
    with torch.no_grad():
        out = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Assignments shape: {out['assignments'].shape}")
    print(f"Existence shape: {out['existence'].shape}")
    print(f"Attractors shape: {out['attractors'].shape}")
    print(f"Overlap logits shape: {out['overlap_logits'].shape}")
    print(f"Learnable temperature: {out['temperature'].item():.4f}")
    
    # Check logit range
    logits = out['assignments']
    print(f"\nLogits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    
    # Check for NaN
    has_nan = any(torch.isnan(v).any() if isinstance(v, torch.Tensor) else False for v in out.values())
    print(f"Has NaN: {has_nan}")
    
    print("\n✓ Model architecture test PASSED")
    return True


def test_loss_function():
    """Test enhanced loss function"""
    print("\n" + "=" * 50)
    print("Testing Loss Function")
    print("=" * 50)
    
    from model import RefinedMultiStreamGCAN
    from loss import GCANLoss, FocalLoss, ContrastiveLoss
    
    # Create model and loss
    model = RefinedMultiStreamGCAN(num_speakers=6, hidden_dim=256)
    criterion = GCANLoss(
        lambda_existence=1.0,
        lambda_ortho=0.1,
        lambda_contrastive=0.1,
        lambda_overlap=0.5,
        label_smoothing=0.1,
        focal_gamma=2.0
    )
    
    model.eval()
    
    # Create dummy data
    batch_size = 2
    x = torch.randn(batch_size, 32000)
    
    with torch.no_grad():
        outputs = model(x)
    
    # Create dummy targets
    T = outputs['assignments'].shape[1]
    targets = {
        'speaker_labels': torch.randint(0, 2, (batch_size, T, 6)).float(),
        'num_speakers': torch.tensor([3, 4]),
        'overlap_regions': torch.randint(0, 2, (batch_size, T)).float()
    }
    
    # Compute loss
    loss, loss_dict = criterion(outputs, targets)
    
    print(f"Total loss: {loss.item():.4f}")
    print(f"  - Assignment loss: {loss_dict['assign']:.4f}")
    print(f"  - Existence loss: {loss_dict['exist']:.4f}")
    print(f"  - Orthogonality loss: {loss_dict['ortho']:.4f}")
    print(f"  - Contrastive loss: {loss_dict['contrastive']:.4f}")
    print(f"  - Overlap loss: {loss_dict['overlap']:.4f}")
    print(f"  - Frame accuracy: {loss_dict['frame_acc']:.4f}")
    print(f"  - Speaker num accuracy: {loss_dict['spk_num_acc']:.4f}")
    
    # Check for NaN
    has_nan = torch.isnan(loss)
    print(f"\nLoss has NaN: {has_nan}")
    
    print("\n✓ Loss function test PASSED")
    return True


def test_focal_loss():
    """Test focal loss implementation"""
    print("\n" + "=" * 50)
    print("Testing Focal Loss")
    print("=" * 50)
    
    from loss import FocalLoss
    
    focal = FocalLoss(gamma=2.0, alpha=0.25)
    
    # Create test data
    logits = torch.randn(10, 6)
    targets = torch.randint(0, 2, (10, 6)).float()
    
    loss = focal(logits, targets)
    print(f"Focal loss: {loss.item():.4f}")
    
    # Compare with standard BCE
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
    print(f"Standard BCE loss: {bce_loss.item():.4f}")
    
    print("\n✓ Focal loss test PASSED")
    return True


def test_backward_pass():
    """Test gradient flow"""
    print("\n" + "=" * 50)
    print("Testing Backward Pass")
    print("=" * 50)
    
    from model import RefinedMultiStreamGCAN
    from loss import GCANLoss
    
    model = RefinedMultiStreamGCAN(num_speakers=6, hidden_dim=256)
    criterion = GCANLoss()
    
    model.train()
    
    # Forward pass
    x = torch.randn(2, 32000)
    outputs = model(x)
    
    T = outputs['assignments'].shape[1]
    targets = {
        'speaker_labels': torch.randint(0, 2, (2, T, 6)).float(),
        'num_speakers': torch.tensor([3, 4]),
        'overlap_regions': torch.randint(0, 2, (2, T)).float()
    }
    
    # Backward pass
    loss, _ = criterion(outputs, targets)
    loss.backward()
    
    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    
    avg_grad_norm = sum(grad_norms) / len(grad_norms)
    max_grad_norm = max(grad_norms)
    
    print(f"Average gradient norm: {avg_grad_norm:.6f}")
    print(f"Max gradient norm: {max_grad_norm:.6f}")
    print(f"Number of parameters with gradients: {len(grad_norms)}")
    
    # Check for NaN gradients
    has_nan_grad = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    print(f"Has NaN gradients: {has_nan_grad}")
    
    print("\n✓ Backward pass test PASSED")
    return True


def main():
    print("\n" + "=" * 60)
    print("GCAN Model Improvement Verification")
    print("=" * 60 + "\n")
    
    tests = [
        ("Model Architecture", test_model_architecture),
        ("Loss Function", test_loss_function),
        ("Focal Loss", test_focal_loss),
        ("Backward Pass", test_backward_pass),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} test FAILED: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ❌")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
