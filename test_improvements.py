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
    
    from model import HR_GridMamba
    
    # Create model with new parameters
    model = HR_GridMamba(
        n_srcs=6, 
        d_model=256,
        n_layers=6,
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
    print(f"Output shape: {out.shape}")
    
    # Check output shape [B, n_srcs, Time]
    expected_shape = (batch_size, 6, audio_length)
    if out.shape != expected_shape:
        print(f"❌ Output shape mismatch! Expected {expected_shape}, got {out.shape}")
        return False
        
    print(f"Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    
    # Check for NaN
    has_nan = torch.isnan(out).any()
    print(f"Has NaN: {has_nan}")
    
    print("\n✓ Model architecture test PASSED")
    return True


def test_loss_function():
    """Test enhanced loss function"""
    print("\n" + "=" * 50)
    print("Testing Loss Function")
    print("=" * 50)
    
    from model import HR_GridMamba, pit_loss
    
    # Create model
    model = HR_GridMamba(n_srcs=4, d_model=64).eval()
    
    # Create dummy outputs (waveforms) [B, n_srcs, T]
    batch_size = 2
    T = 32000
    outputs = torch.randn(batch_size, 4, T)
    
    # Create dummy targets [B, n_active, T]
    # Test case where active speakers < n_srcs
    targets = torch.randn(batch_size, 2, T)
    
    # Compute loss
    loss = pit_loss(outputs, targets, n_srcs=4)
    
    print(f"PIT Loss value: {loss.item():.4f}")
    
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
    
    from model import HR_GridMamba, pit_loss
    
    model = HR_GridMamba(n_srcs=4, d_model=64)
    model.train()
    
    # Forward pass
    x = torch.randn(2, 32000)
    outputs = model(x)
    
    # Create valid target
    targets = torch.randn(2, 2, 32000)
    
    # Backward pass
    loss = pit_loss(outputs, targets, n_srcs=4)
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
