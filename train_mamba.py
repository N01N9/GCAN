import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torch.cuda.amp import autocast, GradScaler  # [추가] AMP 모듈
from datasets import load_dataset, Audio
import io
import time
import torchaudio
from tqdm import tqdm

# Import model and loss
# (model.py 파일은 같은 폴더에 있어야 합니다)
from model import HR_GridMamba, pit_loss

# ==============================================================================
# Dataset Definition: Smart Cropping Applied
# ==============================================================================
class HuggingFacePremixedDataset(IterableDataset):
    def __init__(self, repo_id, split='train', sample_rate=16000, max_samples=32000):
        self.repo_id = repo_id
        self.split = split
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        
        print(f"➜ Initializing PRE-MIXED dataset from: {repo_id}")
        print(f"➜ Target Segment Length: {max_samples} samples ({max_samples/sample_rate:.1f} sec)")
        
        # Load dataset in streaming mode
        self.hf_dataset = load_dataset(repo_id, split=split, streaming=True)
        
        # Audio feature casting (optional/fallback)
        try:
            features = getattr(self.hf_dataset, "features", None)
            if features:
                for k in features:
                    if k in ["mix.wav", "sources.wav", "mixture", "target", "audio", "source", "mix"]:
                        pass
        except Exception as e:
            print(f"Warning: Could not configure audio features: {e}")

    def __iter__(self):
        iterator = iter(self.hf_dataset)
        
        # [설정] 묵음 필터링 설정
        # threshold: 이 값보다 평균 볼륨이 작으면 '묵음'으로 간주하고 다시 자름
        threshold = 1e-3 
        max_retries = 10  # 유효한 구간을 찾기 위해 최대 시도할 횟수

        for item in iterator:
            mix = None
            tgt = None
            
            # ------------------------------------------------------------------
            # 1. Try to find Mixture
            # ------------------------------------------------------------------
            mix_keys = ["mix.wav", "mixture", "mix", "audio"]
            mix_k = next((k for k in mix_keys if k in item), None)
            
            if mix_k:
                data = item[mix_k]
                if isinstance(data, dict) and "array" in data:
                    mix = torch.tensor(data["array"], dtype=torch.float32)
                elif isinstance(data, dict) and "bytes" in data:
                    mix, sr = torchaudio.load(io.BytesIO(data["bytes"]))
                    if sr != self.sample_rate:
                        mix = torchaudio.functional.resample(mix, sr, self.sample_rate)

            # ------------------------------------------------------------------
            # 2. Try to find Targets (Sources)
            # ------------------------------------------------------------------
            # Case A: Combined sources tensor
            src_keys = ["sources.wav", "target", "source", "label"] 
            src_k = next((k for k in src_keys if k in item), None)
            
            if src_k:
                data = item[src_k]
                if isinstance(data, dict) and "array" in data:
                    tgt = torch.tensor(data["array"], dtype=torch.float32)
                elif isinstance(data, dict) and "bytes" in data:
                    tgt, sr = torchaudio.load(io.BytesIO(data["bytes"]))
                    if sr != self.sample_rate:
                        tgt = torchaudio.functional.resample(tgt, sr, self.sample_rate)

            # Case B: Separate source keys (s1, s2, ...)
            if tgt is None:
                sources = []
                for k in ["s1", "s2", "s3", "s4", "source1", "source2"]:
                    if k in item:
                        data = item[k]
                        if isinstance(data, dict) and "array" in data:
                            s = torch.tensor(data["array"], dtype=torch.float32)
                        elif isinstance(data, dict) and "bytes" in data:
                            s, sr = torchaudio.load(io.BytesIO(data["bytes"]))
                            if sr != self.sample_rate:
                                s = torchaudio.functional.resample(s, sr, self.sample_rate)
                        
                        if s.ndim == 2: s = s.mean(0) # Mono
                        sources.append(s)
                
                if sources:
                    tgt = torch.stack(sources)

            # ------------------------------------------------------------------
            # 3. Validation & Smart Cropping
            # ------------------------------------------------------------------
            if mix is not None and tgt is not None:
                # Shape Validation
                if mix.ndim > 1 and mix.shape[0] > 1:
                    mix = mix.mean(dim=0) # Downmix to mono
                if mix.ndim == 2: mix = mix.squeeze(0)
                
                if tgt.ndim == 1:
                    tgt = tgt.unsqueeze(0)
                elif tgt.ndim == 2 and tgt.shape[0] > tgt.shape[1]: 
                     tgt = tgt.t()
                
                current_len = mix.shape[-1]
                
                # Case A: Audio is longer than duration -> Smart Crop
                if current_len > self.max_samples:
                    found_segment = False
                    for _ in range(max_retries):
                        start = torch.randint(0, current_len - self.max_samples, (1,)).item()
                        
                        crop_mix = mix[..., start : start + self.max_samples]
                        crop_tgt = tgt[..., start : start + self.max_samples]
                        
                        # [핵심] 에너지가 충분한지 확인 (묵음 회피)
                        if crop_mix.abs().mean() > threshold:
                            yield crop_mix, crop_tgt
                            found_segment = True
                            break
                    
                    # 10번 시도해도 실패하면? (여기서는 그냥 마지막 시도 결과를 보냄. 필요시 pass로 변경)
                    if not found_segment:
                        # yield crop_mix, crop_tgt # 너무 조용한 파일도 학습하려면 주석 해제
                        pass 

                # Case B: Audio is shorter -> Pad (if not silence)
                elif current_len < self.max_samples:
                    if mix.abs().mean() > threshold:
                        pad_len = self.max_samples - current_len
                        mix = torch.nn.functional.pad(mix, (0, pad_len))
                        tgt = torch.nn.functional.pad(tgt, (0, pad_len))
                        yield mix, tgt

                # Case C: Exact match
                else:
                    if mix.abs().mean() > threshold:
                        yield mix, tgt

def collate_fn(batch):
    # Filter out None if any (though dataset yields valid items)
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None, None

    mix_list = []
    tgt_list = []
    
    max_srcs = 0
    for m, t in batch:
        mix_list.append(m)
        tgt_list.append(t)
        max_srcs = max(max_srcs, t.shape[0])
        
    mix_batch = torch.stack(mix_list) # [B, T]
    
    tgt_batch_list = []
    for t in tgt_list:
        if t.shape[0] < max_srcs:
            padding = torch.zeros(max_srcs - t.shape[0], t.shape[1])
            t = torch.cat([t, padding], dim=0)
        tgt_batch_list.append(t)
    
    tgt_batch = torch.stack(tgt_batch_list) # [B, n_srcs, T]
    
    return mix_batch, tgt_batch

# ==============================================================================
# Training Logic with AMP
# ==============================================================================
def main(args):
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 2. Dataset
    dataset = HuggingFacePremixedDataset(
        repo_id=args.repo_id, 
        split="train", 
        sample_rate=args.sample_rate,
        max_samples=int(args.sample_rate * args.duration)
    )
    
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    # 3. Model
    model = HR_GridMamba(
        n_srcs=args.num_sources,
        n_fft=args.n_fft,
        stride=args.stride,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # 4. Optimizer & Scaler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler() # [추가] for Mixed Precision

    # 5. Training Loop
    model.train()
    step = 0
    epoch = 0
    
    print(f"Starting training with Duration={args.duration}s, AMP Enabled...")
    
    while epoch < args.epochs:
        epoch += 1
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        epoch_loss = 0.0
        count = 0
        
        for i, batch in enumerate(pbar):
            # Handle empty batch from collate_fn
            if batch[0] is None:
                continue

            mix, tgt = batch
            mix = mix.to(device)
            tgt = tgt.to(device)
            
            optimizer.zero_grad() # Initialize grads
            
            # [핵심] Autocast Context
            with autocast():
                est_sources = model(mix)
                loss = pit_loss(est_sources, tgt, n_srcs=args.num_sources)
            
            # [핵심] Scaled Backward
            scaler.scale(loss).backward()
            
            # Unscale before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
            # Step & Update Scaler
            scaler.step(optimizer)
            scaler.update()
            
            step += 1
            epoch_loss += loss.item()
            count += 1
            
            pbar.set_postfix(loss=loss.item())
            
            if step % args.checkpoint_interval == 0:
                ckpt_path = os.path.join(args.output_dir, f"ckpt_step_{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                # print(f" [Saved checkpoint: {ckpt_path}]") # Optional reduce spam
            
            if args.max_steps and step >= args.max_steps:
                print("Max steps reached.")
                return

            # [중요] 20초 긴 오디오 학습 시 메모리 누적 방지
            del mix, tgt, est_sources, loss
            torch.cuda.empty_cache()

        print(f"Epoch {epoch} finished. Avg Loss: {epoch_loss / max(1, count)}")
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"ckpt_epoch_{epoch}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="N02N9/GCAN-voxceleb", help="HuggingFace Dataset ID")
    parser.add_argument("--output_dir", type=str, default="exp_mamba", help="Output directory")
    
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--clip_grad", type=float, default=5.0)
    
    # Model Args
    parser.add_argument("--num_sources", type=int, default=4)
    parser.add_argument("--n_fft", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Data Args
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=4.0, help="Audio segment duration in seconds")

    args = parser.parse_args()
    
    main(args)