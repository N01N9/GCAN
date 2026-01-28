# src/data/preprocess_neighbors.py
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_dir", type=str, default="data/meta")
    parser.add_argument("--top_k", type=int, default=50, help="저장할 유사 화자 후보 수")
    args = parser.parse_args()

    emb_path = os.path.join(args.meta_dir, "speaker_embeddings.json")
    out_path = os.path.join(args.meta_dir, "speaker_neighbors.npy")
    id_map_path = os.path.join(args.meta_dir, "speaker_id_map.json")

    print(f"Loading embeddings from {emb_path}...")
    with open(emb_path, 'r') as f:
        data = json.load(f)

    # 1. 데이터 정리 (ID 리스트 고정)
    speaker_ids = sorted(list(data.keys()))
    embeddings = np.array([data[sid] for sid in speaker_ids])
    
    # 2. Tensor 변환 및 GPU 이동
    print("Moving to GPU for calculation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # (N, D)
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    
    # 3. 코사인 유사도 계산을 위한 정규화 (L2 Norm)
    # 정규화된 벡터끼리의 내적(Dot Product)은 코사인 유사도와 같음
    emb_tensor = torch.nn.functional.normalize(emb_tensor, p=2, dim=1)
    
    # 4. 전체 유사도 행렬 계산 (N x N)
    # 6000명이면 6000x6000 matrix -> 약 144MB (GPU 메모리 충분)
    print("Calculating Similarity Matrix...")
    sim_matrix = torch.mm(emb_tensor, emb_tensor.t())
    
    # 자기 자신은 제외 (유사도 -1로 설정)
    n_spk = len(speaker_ids)
    sim_matrix.fill_diagonal_(-1.0)
    
    # 5. 각 화자별 가장 비슷한 Top-K 인덱스 추출
    print(f"Extracting Top-{args.top_k} neighbors...")
    # values, indices = torch.topk(sim_matrix, k=args.top_k, dim=1)
    # 우리는 인덱스만 필요함
    _, top_indices = torch.topk(sim_matrix, k=args.top_k, dim=1)
    
    # 6. 저장 (CPU로 이동)
    neighbors_np = top_indices.cpu().numpy().astype(np.int32)
    
    print(f"Saving neighbors to {out_path}...")
    np.save(out_path, neighbors_np)
    
    # 인덱스가 어떤 화자 ID인지 매핑 정보도 저장
    print(f"Saving ID map to {id_map_path}...")
    with open(id_map_path, 'w') as f:
        json.dump(speaker_ids, f)
        
    print("Done! Pre-calculation complete.")

if __name__ == "__main__":
    main()