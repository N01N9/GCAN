# HR-GridMamba Dataset Pipeline

이 프로젝트는 HR-GridMamba 모델 학습을 위한 동적 데이터 믹싱 및 전처리 파이프라인을 포함합니다.

## 1. 환경 설정 (Setup)

필요한 라이브러리를 설치합니다.
```bash
pip install -r requirements.txt
```

## 2. 데이터 준비 (Data Preparation)

VoxCeleb 데이터셋이 `config.yaml`에 지정된 경로에 압축 해제되어 있어야 합니다. (현재 `/workspace` 내에 자동 압축 해제 중)

데이터 스캔 및 화자 임베딩(Speaker Embedding)을 추출합니다. 이 과정은 "Hard Negative" 마이닝을 위해 필수적입니다.
```bash
python src/data/preprocess.py
```
* **출력:** `data/meta/file_list.json`, `data/meta/speaker_embeddings.json`

## 3. 데이터 로더 테스트 (Testing DataLoader)

데이터 로더가 정상적으로 작동하는지 확인합니다.
```bash
python src/data/dataset.py
```

## 4. 학습 코드 연동 (Integration)

학습 스크립트에서 다음과 같이 `HRGridMambaDataset`을 사용하세요.

```python
import yaml
from torch.utils.data import DataLoader
from src.data.dataset import HRGridMambaDataset

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset = HRGridMambaDataset(config)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for mixture, targets in loader:
    # mixture: [32, 1, 64000]
    # targets: [32, 4, 64000]
    pass
```
