<h1 style="text-align: center;">🔎 이미지 분류 해커톤: 데이터 속 아이콘의 종류를 맞혀라!</h1>

## 🔍 프로젝트 개요
- **목적** : 아이콘 이미지의 픽셀 데이터를 기반으로 아이콘의 종류를 분류하는 AI 알고리즘 개발
- **주제** : 아이콘 이미지 분류 AI 알고리즘 개발
- **기간** : 2025.03.04 ~ 2025.04.30 (약 2달간)
- **팀 구성** : 1인
- **성과** : Private 18th (상위 4%)
- [**데이콘 결과 제출 게시글**](https://dacon.io/competitions/official/236459/codeshare/12339?page=1&dtype=recent)

## ⚙️ 주요 수행 과정
- 본 프로젝트는 **Dacon 이미지 분류 경진대회**에서 높은 정확도를 달성하기 위해 설계된 PyTorch 기반 분류 모델입니다.
- DenseNet201을 백본으로 사용하고 테스트 데이터에 대한 예측은 **10개의 교차 검증 모델의 평균 앙상블**로 생성합니다.
- 데이터는 **32x32** 흑백 이미지로 주어지며, 학습을 위해 **224x224로 리사이즈** 후 모델에 입력합니다.

## 🗂 데이터 구성
- train.csv
  - ID : 샘플별 고유 ID
  - label : 아이콘의 종류(airplane, apple, ball, bird, building, cat, emotion_face, police_car, rabbit, truck)
  - 0~1023 : 각 이미지의 픽셀 값(Grayscale, 32x32 해상도)을 1D 배열로 펼쳐 저장한 형태
- test.csv
  - ID : 샘플별 고유 ID
  - 0~1023 : 각 이미지의 픽셀 값(Grayscale, 32x32 해상도)을 1D 배열로 펼쳐 저장한 형태

## ⚙️ 모델 및 학습 전략
- **모델**: DenseNet201
  - 사전학습 없이 아키텍처만 사용
- **입력 크기**: `1x32x32` → `3x224x224` (ToPILImage → Resize)
- **손실 함수**: `CrossEntropyLoss`
- **최적화 함수**: `AdamW`, 학습률 `5e-5`
- **LR Scheduler**: `CosineAnnealingLR` with `T_max=100`

## 🔄 데이터 전처리 및 증강

- **Train Transform**
  - 수평 뒤집기 : `RandomHorizontalFlip(p=0.5)`
  - 수직 뒤집기 : `RandomVerticalFlip(p=0.5)`
  - Resize(224x224) 후 정규화 (`mean=0.5`, `std=0.5`)
- **Validation/Test Transform**
  - 증강 없이 Resize 및 정규화만 적용
 
## 🔁 교차 검증 (10-Fold CV)

- **StratifiedKFold(n_splits=10, shuffle=True, random_state=42)** 사용
- 각 Fold마다:
  - 학습: 모델 초기화 → 학습 진행 → `val_loss` 기준 최적 모델 저장
  - 테스트: 해당 Fold의 최적 모델로 전체 테스트 데이터 예측
- **최종 예측**: 10개의 Fold 예측 결과 평균 → `argmax`로 클래스 결정

## 📈 성능 지표

각 Fold에서 다음과 같은 지표를 출력합니다:
- Train Loss / Val Loss
- Validation Accuracy

> 최적 모델은 **Validation Loss가 최소일 때** 저장됩니다.

## 🧑🏻‍💻환경
- Python
- VScode
- torch, torchvision
- densenet201

