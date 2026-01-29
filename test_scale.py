
import torch
import torch.nn.functional as F
from model import RefinedMultiStreamGCAN

def test_logit_range():
    print("Testing Logit Range...")
    model = RefinedMultiStreamGCAN(num_speakers=6, hidden_dim=256)
    model.eval()
    
    # Create random audio input (Batch=2, T=32000 for 2 seconds)
    x = torch.randn(2, 32000)
    
    with torch.no_grad():
        out = model(x)
        logits = out['assignments']
        existence = out['existence']
        
        print(f"Logits Range: {logits.min().item():.4f} ~ {logits.max().item():.4f}")
        print(f"Existence Range: {existence.min().item():.4f} ~ {existence.max().item():.4f}")
        
        # Check if they are confined to approx [-1, 1] (since temp=1.0)
        # Cosine similarity is [-1, 1]. divided by 1.0 is [-1, 1].
        # If temp=0.1, it should be [-10, 10].
        
        prob_max = torch.sigmoid(logits.max()).item()
        print(f"Max Prob: {prob_max:.4f}")
        
        if logits.abs().max() < 1.5:
            print("⚠️ ALERT: Logits are too small! The model cannot be confident.")
        else:
            print("✅ Status: Logits have good dynamic range.")

if __name__ == "__main__":
    test_logit_range()
