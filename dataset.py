import torch
import torch.nn.functional as F
import torchaudio
import os
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset, Audio
from huggingface_hub import list_repo_files
from typing import Dict, Optional, Tuple
import numpy as np
import random


class SpecAugment:
    """SpecAugment for mel spectrogram augmentation"""
    def __init__(
        self, 
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to mel spectrogram
        Args:
            mel: (n_mels, time) mel spectrogram
        Returns:
            Augmented mel spectrogram
        """
        mel = mel.clone()
        n_mels, time_steps = mel.shape
        
        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(0, n_mels - f))
            mel[f0:f0 + f, :] = 0
        
        # Time masking
        for _ in range(self.num_time_masks):
            t = random.randint(0, min(self.time_mask_param, time_steps // 4))
            t0 = random.randint(0, max(0, time_steps - t))
            mel[:, t0:t0 + t] = 0
            
        return mel


class MixupAugmentation:
    """Mixup augmentation for audio and labels"""
    def __init__(self, alpha: float = 0.4, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob
        
    def __call__(
        self, 
        audio1: torch.Tensor, 
        labels1: torch.Tensor,
        audio2: torch.Tensor,
        labels2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply Mixup augmentation
        Returns:
            mixed_audio, mixed_labels, lambda
        """
        if random.random() > self.prob:
            return audio1, labels1, 1.0
            
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)  # Ensure lam >= 0.5 for stability
        
        # Mix audio
        mixed_audio = lam * audio1 + (1 - lam) * audio2
        
        # Mix labels (soft labels for mixup)
        mixed_labels = lam * labels1 + (1 - lam) * labels2
        
        return mixed_audio, mixed_labels, lam


class HuggingFaceDiarizationDataset(IterableDataset):
    """
    Enhanced HuggingFace Streaming Dataset for Speaker Diarization
    - 20초 오디오에 2-6명 화자
    - Overlap 포함
    - SpecAugment, Mixup 지원
    """
    def __init__(
        self, 
        repo_id: str,
        split: str = "train",
        val_ratio: float = 0.1,
        sample_rate: int = 16000, 
        duration: float = 20.0,
        n_mels: int = 80,
        augmentation: bool = True,
        spec_augment: bool = True,
        mixup: bool = True,
        seed: int = 42
    ):
        # Get all tar shard files
        all_files = list_repo_files(repo_id, repo_type="dataset")
        all_tar_shards = sorted([f for f in all_files if f.endswith(".tar") and "train" in f])
        
        if not all_tar_shards:
            raise ValueError(f"No tar files found in repo '{repo_id}'")
        
        # Shard-based Train/Validation split
        np.random.seed(seed)
        np.random.shuffle(all_tar_shards)
        
        val_size = max(1, int(len(all_tar_shards) * val_ratio))
        
        if split == "validation":
            selected_shards = all_tar_shards[:val_size]
        else:
            selected_shards = all_tar_shards[val_size:]
            
        # Load streaming dataset
        self.audio_ds = load_dataset(
            repo_id, 
            data_files=selected_shards, 
            split="train",
            streaming=True
        ).cast_column("wav", Audio(sampling_rate=sample_rate))
        
        # Load metadata
        meta_stream = load_dataset(
            repo_id, 
            data_files="metadata.jsonl", 
            split="train", 
            streaming=True
        )
        
        self.meta = {}
        for item in meta_stream:
            key = os.path.basename(item["file_name"]).replace(".wav", "")
            self.meta[key] = item["utterances"]
        
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * duration)
        self.n_mels = n_mels
        self.hop_length = 160
        self.num_frames = self.max_samples // self.hop_length
        self.max_speakers = 6
        self.augmentation = augmentation
        self.split = split
        
        # Augmentation modules
        self.spec_augment = SpecAugment() if spec_augment and split == "train" else None
        self.mixup = MixupAugmentation(alpha=0.4, prob=0.3) if mixup and split == "train" else None
        
        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, 
            n_fft=512, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
        )
        
        # Buffer for Mixup (store previous sample)
        self._prev_sample = None
        
        print(f"Initialized dataset: {repo_id} ({split})")
        print(f"  Selected {len(selected_shards)} / {len(all_tar_shards)} shards")
        print(f"  Loaded metadata for {len(self.meta)} files")
        print(f"  Augmentation: {augmentation}, SpecAugment: {spec_augment}, Mixup: {mixup}")
    
    def __iter__(self):
        """Iterate over dataset samples"""
        for item in self.audio_ds:
            audio_path = item.get("file_name", "") or item["wav"].get("path", "")
            key = os.path.basename(audio_path).replace(".wav", "")
            
            if key not in self.meta:
                continue
            
            # Load audio
            audio = torch.tensor(item["wav"]["array"], dtype=torch.float32)
            
            # Convert to mono
            if audio.ndim > 1:
                audio = audio.mean(dim=0 if audio.shape[0] < audio.shape[1] else 1)
            
            # Pad or trim
            if audio.numel() > self.max_samples:
                audio = audio[:self.max_samples]
            else:
                audio = F.pad(audio, (0, self.max_samples - audio.numel()))
            
            # Create targets
            targets = self._create_targets(self.meta[key])
            
            # Apply basic augmentation
            if self.augmentation and self.split == "train":
                audio = self._augment(audio)
            
            # Apply Mixup if enabled and we have a previous sample
            mixup_lambda = 1.0
            if self.mixup and self._prev_sample is not None:
                prev_audio, prev_labels = self._prev_sample
                audio, mixed_labels, mixup_lambda = self.mixup(
                    audio, targets['target_mask'],
                    prev_audio, prev_labels
                )
                if mixup_lambda < 1.0:
                    targets['target_mask'] = mixed_labels
                    # Also mix exist_target
                    targets['exist_target'] = (targets['target_mask'].sum(dim=0) > 0.5).float()
            
            # Store current sample for next mixup
            if self.mixup:
                self._prev_sample = (audio.clone(), targets['target_mask'].clone())
            
            yield {
                'audio': audio,
                'target_mask': targets['target_mask'],
                'exist_target': targets['exist_target'],
                'overlap_regions': targets['overlap_regions'],
                'mixup_lambda': mixup_lambda
            }
    
    def _create_targets(self, utterances):
        """Create targets from metadata"""
        target_mask = torch.zeros(self.num_frames, self.max_speakers)
        exist_target = torch.zeros(self.max_speakers)
        
        speakers = sorted(set(u["speaker"] for u in utterances))[:self.max_speakers]
        
        for i in range(len(speakers)):
            exist_target[i] = 1.0
        
        for i, spk in enumerate(speakers):
            for u in utterances:
                if u["speaker"] == spk:
                    start_frame = int((u["start"] * self.sample_rate) / self.hop_length)
                    end_frame = int((u["end"] * self.sample_rate) / self.hop_length)
                    
                    start_frame = max(0, min(start_frame, self.num_frames - 1))
                    end_frame = max(0, min(end_frame, self.num_frames))
                    
                    target_mask[start_frame:end_frame, i] = 1.0
        
        overlap_regions = (target_mask.sum(dim=1) > 1).float()
        
        return {
            'target_mask': target_mask,
            'exist_target': exist_target,
            'overlap_regions': overlap_regions
        }
    
    def _augment(self, audio):
        """Apply audio augmentation"""
        # Random gain (±6dB)
        if np.random.rand() < 0.5:
            gain_db = np.random.uniform(-6, 6)
            gain = 10 ** (gain_db / 20)
            audio = audio * gain
        
        # Add background noise
        if np.random.rand() < 0.3:
            noise_level = np.random.uniform(0.001, 0.01)
            noise = torch.randn_like(audio) * noise_level
            audio = audio + noise
        
        # Random speed perturbation (simplified via resampling)
        if np.random.rand() < 0.2:
            speed_factor = np.random.uniform(0.9, 1.1)
            new_length = int(len(audio) / speed_factor)
            audio = F.interpolate(
                audio.unsqueeze(0).unsqueeze(0), 
                size=new_length, 
                mode='linear', 
                align_corners=False
            ).squeeze()
            # Pad or trim back to original length
            if len(audio) > self.max_samples:
                audio = audio[:self.max_samples]
            else:
                audio = F.pad(audio, (0, self.max_samples - len(audio)))
        
        # Clip to prevent overflow
        audio = torch.clamp(audio, -1.0, 1.0)
        
        return audio


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    audios = []
    target_masks = []
    exist_targets = []
    overlap_regions = []
    
    for item in batch:
        audios.append(item['audio'])
        target_masks.append(item['target_mask'])
        exist_targets.append(item['exist_target'])
        overlap_regions.append(item['overlap_regions'])
    
    audios = torch.stack(audios)
    target_masks = torch.stack(target_masks)
    exist_targets = torch.stack(exist_targets)
    overlap_regions = torch.stack(overlap_regions)
    
    num_speakers = exist_targets.sum(dim=1).long()
    
    return {
        'audio': audios,
        'targets': {
            'speaker_labels': target_masks,
            'num_speakers': num_speakers,
            'overlap_regions': overlap_regions
        }
    }


def create_dataloaders(
    repo_id: str,
    batch_size: int = 8,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    sample_rate: int = 16000,
    duration: float = 20.0,
    n_mels: int = 80,
    spec_augment: bool = True,
    mixup: bool = True
):
    """Create train and validation dataloaders"""
    train_dataset = HuggingFaceDiarizationDataset(
        repo_id=repo_id,
        split="train",
        val_ratio=val_ratio,
        sample_rate=sample_rate,
        duration=duration,
        n_mels=n_mels,
        augmentation=True,
        spec_augment=spec_augment,
        mixup=mixup
    )
    
    val_dataset = HuggingFaceDiarizationDataset(
        repo_id=repo_id,
        split="validation",
        val_ratio=val_ratio,
        sample_rate=sample_rate,
        duration=duration,
        n_mels=n_mels,
        augmentation=False,
        spec_augment=False,
        mixup=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed fixed to: {seed}")


# Test script
if __name__ == "__main__":
    import argparse

    set_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_id', type=str, required=True, help='HuggingFace repo ID')
    args = parser.parse_args()
    
    print("Testing Enhanced Dataset with Mixup and SpecAugment...")
    
    train_loader, val_loader = create_dataloaders(
        repo_id=args.repo_id,
        batch_size=2,
        num_workers=0,
        val_ratio=0.1,
        spec_augment=True,
        mixup=True
    )
    
    print("\nTesting Train loader...")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}: Audio {batch['audio'].shape}, Speakers {batch['targets']['num_speakers']}")
        print(f"  Labels shape: {batch['targets']['speaker_labels'].shape}")
        print(f"  Overlap shape: {batch['targets']['overlap_regions'].shape}")
        if i >= 1: 
            break
            
    print("\nTesting Validation loader...")
    for i, batch in enumerate(val_loader):
        print(f"Batch {i}: Audio {batch['audio'].shape}, Speakers {batch['targets']['num_speakers']}")
        if i >= 1: 
            break
    
    print("\n✓ Dataset test completed!")