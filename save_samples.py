import os
import argparse
import torch
import torchaudio
from torch.utils.data import DataLoader, IterableDataset
import io
from datasets import load_dataset, Audio

class HuggingFacePremixedDataset(IterableDataset):
    def __init__(self, repo_id, split='train', sample_rate=16000):
        self.repo_id = repo_id
        self.split = split
        self.sample_rate = sample_rate
        
        print(f"➜ Initializing PRE-MIXED dataset from: {repo_id}")
        
        # Load dataset in streaming mode
        # Assuming the dataset has columns like 'mixture' and 'target' (or 'sources')
        self.hf_dataset = load_dataset(repo_id, split="train", streaming=True)

    def __iter__(self):
        iterator = iter(self.hf_dataset)
        
        for item in iterator:
            # Check structure
            # Case A: 'mixture' and 'target' columns
            if "mixture" in item and "target" in item:
                mix = torch.tensor(item["mixture"]["array"], dtype=torch.float32)
                tgt = torch.tensor(item["target"]["array"], dtype=torch.float32) # [C, T] or [T, C] check needed
                
                # Ensure shapes [1, T] and [C, T]
                if mix.ndim == 1: mix = mix.unsqueeze(0)
                if tgt.ndim == 1: tgt = tgt.unsqueeze(0) # Should be [C, T]
                
                yield mix, tgt

            # Case B: 'audio' (mixture) and 'label' (target) - common in some datasets
            elif "audio" in item and "label" in item:
                 mix = torch.tensor(item["audio"]["array"], dtype=torch.float32)
                 # Label might be target audio or class, assuming audio based on user context
                 # If label is not audio, this is wrong.
                 pass

            # Case C: N02N9/GCAN-voxceleb specific structure?
            # If it's pre-processed, it might be just "audio" which is the Mixture, 
            # and "annotation" or similar?
            # Or maybe "mixture.wav" and "source1.wav"... inside the item?
            
            # Let's try to dump the keys for debugging if we can't find standard keys
            # But since we assume it's pre-mixed:
            
            # Default fallback: Try to access commonly used keys
            try:
                # If keys are like 'mix', 's1', 's2', 's3', 's4'
                if "mix" in item:
                     mix = torch.tensor(item["mix"]["array"], dtype=torch.float32)
                     sources = []
                     for k in ["s1", "s2", "s3", "s4"]:
                         if k in item:
                             sources.append(torch.tensor(item[k]["array"], dtype=torch.float32))
                     
                     if sources:
                         tgt = torch.stack(sources)
                         yield mix.unsqueeze(0) if mix.ndim==1 else mix, tgt
            except:
                pass

def save_samples_simple(repo_id, output_dir="samples", num_samples=3):
    print(f"Loading Pre-mixed Dataset from {repo_id}...")
    
    dataset = load_dataset(repo_id, split="train", streaming=True)
    
    # --- [CRITICAL FIX] ---
    # We use decode=False to:
    # 1. Avoid ImportError (e.g. torchcodec missing)
    # 2. Prevent automatic mono downmixing by 'datasets'
    # 3. Get raw bytes for direct torchaudio loading
    try:
        features = getattr(dataset, "features", None)
        if features:
            for k in features:
                if k in ["mix.wav", "sources.wav", "mixture", "target", "audio", "source", "mix"]:
                    print(f"  - Casting '{k}' to decode=False")
                    dataset = dataset.cast_column(k, Audio(sampling_rate=16000, mono=False))
    except Exception as e:
        print(f"Warning: Could not configure audio features: {e}")
    # ----------------------

    iterator = iter(dataset)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving {num_samples} samples to '{output_dir}'...")
    
    count = 0
    for i, item in enumerate(iterator):
        if count >= num_samples: break
        
        print(f"[Sample {i+1}] Processing...")
        
        # 1. Mixture
        mix_waveform = None
        sr = 16000
        
        # Identify key
        mix_keys = ["mix.wav", "mixture", "mix", "audio"]
        mix_k = next((k for k in mix_keys if k in item), None)
        
        if mix_k:
            data = item[mix_k]
            if isinstance(data, dict) and "array" in data:
                mix_waveform = torch.tensor(data["array"], dtype=torch.float32)
                sr = data.get("sampling_rate", 16000)
            elif isinstance(data, dict) and "bytes" in data:
                 mix_waveform, sr = torchaudio.load(io.BytesIO(data["bytes"]))
            
            if mix_waveform is not None:
                if mix_waveform.ndim == 1: mix_waveform = mix_waveform.unsqueeze(0)
                
                mix_path = os.path.join(output_dir, f"sample_{i+1}_mix.wav")
                torchaudio.save(mix_path, mix_waveform, sr)
                print(f"  - Mixture saved: {mix_path}")

        # 2. Sources (Target)
        src_waveform = None
        src_keys = ["sources.wav", "target", "source", "label"] 
        src_k = next((k for k in src_keys if k in item), None)
        
        if src_k:
            data = item[src_k]
            if isinstance(data, dict) and "array" in data:
                src_waveform = torch.tensor(data["array"], dtype=torch.float32)
                sr = data.get("sampling_rate", 16000)
            elif isinstance(data, dict) and "bytes" in data:
                src_waveform, sr = torchaudio.load(io.BytesIO(data["bytes"]))
                
            if src_waveform is not None:
                if src_waveform.ndim == 2 and src_waveform.shape[0] > src_waveform.shape[1] and src_waveform.shape[1] <= 8:
                     src_waveform = src_waveform.t()
                elif src_waveform.ndim == 1:
                     src_waveform = src_waveform.unsqueeze(0)
                
                print(f"    Target Shape: {src_waveform.shape}")

                for ch in range(src_waveform.shape[0]):
                    src_path = os.path.join(output_dir, f"sample_{i+1}_source_{ch+1}.wav")
                    torchaudio.save(src_path, src_waveform[ch].unsqueeze(0), sr)
                    print(f"  - Source {ch+1} saved: {src_path}")
                
        count += 1
        
    print("\nDone!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="N02N9/GCAN-voxceleb")
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()
    save_samples_simple(args.repo_id, num_samples=args.num_samples)
