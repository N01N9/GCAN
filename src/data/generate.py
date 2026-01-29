import os
import yaml
import argparse
import numpy as np
import webdataset as wds
import soundfile as sf
import io
import multiprocessing as mp
from tqdm import tqdm
import sys
import random
import json

# 경로 설정
sys.path.append(os.getcwd())
try:
    from src.data.dataset import HRGridMambaDataset
except ImportError:
    # Fallback import
    sys.path.append(os.path.join(os.getcwd(), '../..'))
    from src.data.dataset import HRGridMambaDataset

# ---------------------------------------------------------
# Global Variables for Worker Processes
# ---------------------------------------------------------
worker_dataset = None
SHARED_EMBEDDINGS = None  # 메인 프로세스에서 로드된 데이터를 공유받을 변수

def worker_init(config):
    """
    워커 프로세스 초기화 함수.
    메인 프로세스의 메모리 공간을 공유받으므로 SHARED_EMBEDDINGS 접근 가능.
    """
    global worker_dataset, SHARED_EMBEDDINGS
    
    pid = os.getpid()
    # print(f"[Worker {pid}] Initializing Dataset...")
    
    # SHARED_EMBEDDINGS가 None이 아니면 메인에서 로드된 것임
    worker_dataset = HRGridMambaDataset(
        config, 
        split='train', 
        embeddings_data=SHARED_EMBEDDINGS 
    )

def encode_audio(audio_array, sr, format='WAV'):
    if audio_array.ndim == 1:
        pass
    elif audio_array.ndim == 2:
        if audio_array.shape[0] < audio_array.shape[1]:
            audio_array = audio_array.T
            
    buf = io.BytesIO()
    # WAV 포맷 사용 (속도 최적화)
    sf.write(buf, audio_array, sr, format=format)
    return buf.getvalue()

def process_shard(shard_idx, samples_per_shard, config, output_dir):
    global worker_dataset
    
    # 랜덤 시드 재설정
    seed = shard_idx * 1000 + os.getpid()
    random.seed(seed)
    np.random.seed(seed)
    
    shard_pattern = os.path.join(output_dir, f"shard-{shard_idx:06d}.tar")
    sink = wds.TarWriter(shard_pattern)
    
    sr = config['data']['sample_rate']
    
    cnt = 0
    for i in range(samples_per_shard):
        try:
            # 전역 데이터셋 사용
            mixture, target = worker_dataset[0] 
            
            mix_np = mixture.numpy().squeeze()
            target_np = target.numpy()
            
            key = f"{shard_idx:05d}_{i:05d}"
            
            sample = {
                "__key__": key,
                "mix.wav": encode_audio(mix_np, sr, format='WAV'),
                "sources.wav": encode_audio(target_np, sr, format='WAV'),
                "json": {
                    "sample_rate": sr,
                    "duration": config['data']['duration'],
                    "shard": shard_idx,
                    "index_in_shard": i
                }
            }
            sink.write(sample)
            cnt += 1
        except Exception as e:
            # print(f"[Error] Shard {shard_idx} sample {i}: {e}")
            continue
            
    sink.close()
    return cnt

def partial_wrapper(args):
    return process_shard(*args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output_dir", type=str, default="data/shards")
    parser.add_argument("--total_samples", type=int, default=300000)
    parser.add_argument("--shard_size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    
    args = parser.parse_args()
    
    # Config 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.limit:
        args.total_samples = args.limit
        if args.total_samples < args.shard_size:
            args.shard_size = args.total_samples

    total_shards = (args.total_samples + args.shard_size - 1) // args.shard_size
    
    # ---------------------------------------------------------
    # [핵심] Embeddings 미리 로드 (메인 프로세스)
    # ---------------------------------------------------------
    global SHARED_EMBEDDINGS
    emb_path = os.path.join(config['preprocess']['output_path'], 'speaker_embeddings.json')
    
    if os.path.exists(emb_path):
        print(f"[Main] Loading embeddings from {emb_path} into memory...")
        with open(emb_path, 'r') as f:
            SHARED_EMBEDDINGS = json.load(f)
        print(f"[Main] Loaded embeddings for {len(SHARED_EMBEDDINGS)} speakers.")
    else:
        print("[Main] Warning: No embeddings file found. Random mixing will be used.")
        SHARED_EMBEDDINGS = None
        
    print(f"Generating {args.total_samples} samples with {args.workers} workers.")
    
    tasks = []
    samples_remaining = args.total_samples
    for i in range(total_shards):
        n = min(samples_remaining, args.shard_size)
        tasks.append((i, n, config, args.output_dir))
        samples_remaining -= n
        
    if args.workers > 1:
        # Initializer를 통해 worker_init 실행
        # 리눅스(Fork) 환경에서는 SHARED_EMBEDDINGS가 자식 프로세스에 자동으로 복사(공유)됨
        with mp.Pool(processes=args.workers, initializer=worker_init, initargs=(config,)) as pool:
            results = list(tqdm(pool.imap_unordered(partial_wrapper, tasks), total=total_shards))
    else:
        # 단일 프로세스 디버깅용
        worker_init(config)
        results = []
        for task in tqdm(tasks):
            results.append(process_shard(*task))
            
    print(f"Done. Generated {sum(results)} samples.")

if __name__ == "__main__":
    main()