import torch
import torch.nn.functional as F
import torchaudio
import os
import random
import numpy as np
from torch.utils.data import IterableDataset
from datasets import load_dataset, Audio
from huggingface_hub import list_repo_files

class HRGridMambaDatasetStreaming(IterableDataset):
    """
    Requested streaming implementation based on the user's provided local logic.
    Instead of loading local files, it streams from HuggingFace and applies the same
    turn-taking and mixing logic on-the-fly.
    """
    def __init__(self, repo_id, split='train', sample_rate=16000, duration=20.0):
        self.repo_id = repo_id
        self.split = split
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        
        # Load HuggingFace Streaming Dataset
        # We need efficient random access-like behavior for mixing multiple speakers.
        # But streaming datasets are sequential.
        # Solution: Use a buffer to shuffle and pick speakers.
        print(f"➜ Initializing streaming dataset from: {repo_id}")
        
        # Load dataset in streaming mode
        self.hf_dataset = load_dataset(repo_id, split="train", streaming=True)
        self.hf_dataset = self.hf_dataset.cast_column("wav", Audio(sampling_rate=sample_rate))
    
    def _apply_turn_taking(self, sources):
        """Same turn-taking logic as provided code"""
        masked_sources = []
        for src in sources:
            # 1. Active length (20% ~ 100%)
            active_ratio = random.uniform(0.2, 1.0)
            active_len = int(self.n_samples * active_ratio)
            
            # 2. Start position
            max_start = self.n_samples - active_len
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + active_len
            
            # 3. Masking
            masked_src = src.clone()
            if start_idx > 0:
                masked_src[:start_idx] = 0
            if end_idx < self.n_samples:
                masked_src[end_idx:] = 0
                
            masked_sources.append(masked_src)
            
        return masked_sources

    def _process_audio(self, audio_tensor):
        """Process raw audio to match target length"""
        # Convert to mono
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
        else:
            audio_tensor = audio_tensor.unsqueeze(0)
            
        # Resample is handled by HF .cast_column, so we just check length
        if audio_tensor.shape[1] > self.n_samples:
            # Crop random segment
            max_offset = audio_tensor.shape[1] - self.n_samples
            offset = random.randint(0, max_offset)
            audio_tensor = audio_tensor[:, offset:offset+self.n_samples]
        elif audio_tensor.shape[1] < self.n_samples:
            # Pad
            audio_tensor = F.pad(audio_tensor, (0, self.n_samples - audio_tensor.shape[1]))
            
        return audio_tensor[0] # Return [T]

    def __iter__(self):
        """
        Streaming mixing logic:
        Since we can't randomly access files by ID in streaming mode (efficiently),
        we act as a 'mixer' that takes a incoming stream of speakers and mixes them dynamically.
        
        Buffer Strategy:
        1. Fill a buffer with N speakers (e.g., 100 speakers).
        2. Randomly select 1-4 speakers from the buffer to create a mixture.
        3. Yield the mixture.
        4. Replace used speakers with new ones from the stream to keep diversity.
        """
        buffer = []
        buffer_size = 50 # Keep 50 speakers in memory
        
        iterator = iter(self.hf_dataset)
        
        # 1. Fill Buffer
        print("  Filling buffer...")
        try:
            while len(buffer) < buffer_size:
                item = next(iterator)
                if "wav" in item and item["wav"] is not None and "array" in item["wav"]:
                    wav = torch.tensor(item["wav"]["array"], dtype=torch.float32)
                    buffer.append(wav)
        except StopIteration:
            pass # Dataset too small
            
        # 2. Generate Mixtures indefinitely (or until stream ends)
        while True:
            # Refill buffer if needed
            try:
                while len(buffer) < buffer_size:
                    item = next(iterator)
                if "wav" in item and item["wav"] is not None and "array" in item["wav"]:
                    wav = torch.tensor(item["wav"]["array"], dtype=torch.float32)
                    buffer.append(wav)
            except StopIteration:
                if not buffer: break # No more data
                # If stream ends, just reuse buffer until empty or loop? 
                # Let's just use what's left.

            # Determine num speakers (1-4)
            n_speakers = random.randint(1, min(4, len(buffer)))
            
            # Select speakers
            active_indices = random.sample(range(len(buffer)), n_speakers)
            selected_audios = []
            
            for idx in active_indices:
                # Process: crop/pad/mono
                raw_audio = buffer[idx]
                proc_audio = self._process_audio(raw_audio)
                
                # Gain Augmentation
                gain = 10 ** (random.uniform(-5, 0) / 20)
                selected_audios.append(proc_audio * gain)
            
            # Apply Turn-Taking (50% prob)
            if random.random() < 0.5:
                final_sources = self._apply_turn_taking(selected_audios)
            else:
                final_sources = selected_audios
                
            # Create Target Tensor
            target = torch.zeros((4, self.n_samples))
            for i, src in enumerate(final_sources):
                if i < 4:
                    target[i] = src
                    
            # Create Mixture
            mixture = target.sum(0, keepdim=True)
            
            # Clipping
            mixture = torch.clamp(mixture, -1.0, 1.0)
            target = torch.clamp(target, -1.0, 1.0)
            
            yield mixture, target
            
            # Optional: Rotate buffer? 
            # In a true streaming setting, we should pop used items or replace them.
            # Let's replace one random speaker from the selection to keep it fresh.
            if len(active_indices) > 0:
                replace_idx = active_indices[0]
                try:
                    new_item = next(iterator)
                    if "wav" in new_item:
                         buffer[replace_idx] = torch.tensor(new_item["wav"]["array"], dtype=torch.float32)
                except StopIteration:
                    pass # Keep reusing if stream empty

