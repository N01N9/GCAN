import os
import json
import random
import torch
import torchaudio
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial.distance import cdist

class HRGridMambaDataset(Dataset):
    def __init__(self, config, split='train', embeddings_data=None):
        self.config = config
        self.split = split
        self.sample_rate = config['data']['sample_rate']
        self.duration = config['data']['duration']
        self.n_samples = config['data'].get('n_samples', int(self.sample_rate * self.duration))
        
        meta_path = config['preprocess']['output_path']
        list_file = f"file_list_{split}.json"
        
        # 1. 파일 리스트 로드
        with open(os.path.join(meta_path, list_file), 'r') as f:
            self.speakers = json.load(f)
            self.file_list_ids = set(self.speakers.keys())

        # 2. 유사도 족보(Neighbors) 로드
        self.neighbors = None
        self.speaker_ids_map = []
        self.id_to_idx = {}
        self.valid_train_ids = []

        if split == 'train':
            neighbor_path = os.path.join(meta_path, "speaker_neighbors.npy")
            map_path = os.path.join(meta_path, "speaker_id_map.json")
            
            if os.path.exists(neighbor_path) and os.path.exists(map_path):
                # print(f"Loading pre-calculated neighbors from {neighbor_path}...")
                self.neighbors = np.load(neighbor_path)
                with open(map_path, 'r') as f:
                    self.speaker_ids_map = json.load(f)
                
                self.id_to_idx = {sid: i for i, sid in enumerate(self.speaker_ids_map)}
                self.valid_train_ids = [sid for sid in self.speaker_ids_map if sid in self.file_list_ids]
            else:
                self.valid_train_ids = list(self.file_list_ids)
        else:
            self.valid_train_ids = list(self.file_list_ids)

    def _load_audio(self, path):
        try:
            if not os.path.exists(path): return torch.zeros(self.n_samples)
            info = torchaudio.info(path)
            sr = info.sample_rate
            
            # [수정] 턴테이킹을 위해 원본을 좀 더 길게 읽을 수도 있지만, 
            # 여기서는 편의상 duration만큼 읽고 마스킹하는 방식을 사용
            target_frames = int(self.duration * sr)
            
            if info.num_frames > target_frames:
                frame_offset = random.randint(0, info.num_frames - target_frames)
                waveform, _ = torchaudio.load(path, frame_offset=frame_offset, num_frames=target_frames)
            else:
                waveform, _ = torchaudio.load(path)
            
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
            if waveform.shape[0] > 1: waveform = waveform.mean(0, keepdim=True)
            if waveform.shape[1] < self.n_samples:
                waveform = F.pad(waveform, (0, self.n_samples - waveform.shape[1]))
            else:
                waveform = waveform[:, :self.n_samples]
                
            return waveform[0]
        except:
            return torch.zeros(self.n_samples)

    def _select_speakers(self, n_target):
        if not self.valid_train_ids: return []
        
        first_spk = random.choice(self.valid_train_ids)
        selected_ids = [first_spk]
        
        while len(selected_ids) < n_target:
            candidate = None
            if self.neighbors is not None and random.random() < 0.3:
                last_spk = selected_ids[-1]
                if last_spk in self.id_to_idx:
                    last_idx = self.id_to_idx[last_spk]
                    neighbor_indices = self.neighbors[last_idx]
                    cand_idx = np.random.choice(neighbor_indices)
                    cand_id = self.speaker_ids_map[cand_idx]
                    if cand_id not in selected_ids and cand_id in self.file_list_ids:
                        candidate = cand_id
            
            if candidate is None:
                cand = random.choice(self.valid_train_ids)
                if cand not in selected_ids:
                    candidate = cand
            
            selected_ids.append(candidate)
            
        return selected_ids

    # --- [추가됨] 턴테이킹(활성 구간) 생성 로직 ---
    def _apply_turn_taking(self, sources):
        """
        각 소스(화자)마다 랜덤한 시작점(Start)과 끝점(End)을 부여하여
        자연스러운 대화 흐름(오버랩, 턴테이킹, 묵음)을 만듭니다.
        """
        masked_sources = []
        for src in sources:
            # 1. 활성 길이 결정 (전체 길이의 20% ~ 100%)
            active_ratio = random.uniform(0.2, 1.0)
            active_len = int(self.n_samples * active_ratio)
            
            # 2. 시작 지점 결정 (0 ~ 남은 공간)
            # 만약 active_len이 전체보다 작으면, 시작 지점을 밀어서 앞부분 묵음 생성 가능
            max_start = self.n_samples - active_len
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + active_len
            
            # 3. 마스킹 (0으로 채우기)
            # 텐서 복사 (원본 보존)
            masked_src = src.clone()
            
            # 앞부분 묵음
            if start_idx > 0:
                masked_src[:start_idx] = 0
            # 뒷부분 묵음
            if end_idx < self.n_samples:
                masked_src[end_idx:] = 0
                
            masked_sources.append(masked_src)
            
        return masked_sources

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        # 1. 화자 수 결정 (1~4명)
        n_speakers = random.randint(1, 4)
        
        # 2. 화자 ID 선택
        spk_ids = self._select_speakers(n_speakers)
        
        # 3. 오디오 로드 (일단 Full Length로 로드)
        sources_raw = []
        for spk_id in spk_ids:
            path = random.choice(self.speakers[spk_id])
            audio = self._load_audio(path)
            # Gain Augmentation
            audio = audio * (10 ** (random.uniform(-5, 0) / 20))
            sources_raw.append(audio)

        # 4. [핵심] 턴테이킹 적용 여부 결정 (50% 확률)
        # - Mode A (Separation Focus): 꽉 채워서 학습 (기존 방식)
        # - Mode B (Diarization Focus): 턴테이킹 및 묵음 적용
        if random.random() < 0.5:
            final_sources = self._apply_turn_taking(sources_raw)
        else:
            final_sources = sources_raw

        # 5. Target Tensor 생성 (4채널)
        target = torch.zeros((4, self.n_samples))
        for i, src in enumerate(final_sources):
            if i < 4: 
                target[i] = src

        # 6. 믹스처 생성
        mixture = target.sum(0, keepdim=True)
        
        # 클리핑
        mixture = torch.clamp(mixture, -1.0, 1.0)
        target = torch.clamp(target, -1.0, 1.0)

        return mixture, target