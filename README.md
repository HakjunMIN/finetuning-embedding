# BGE-M3 Korean Embedding Fine-tuning (Azure ML Studio)

이 프로젝트는 Azure ML Studio에서 노트북으로 실험(Experiment)과 실행(Run)을 수행하고, 이후 MLOps 배포 워크플로우까지 확장할 수 있는 템플릿입니다. 데이터셋은 전기/전자 용어 기반으로 합성 생성됩니다.

## 1) 로컬(macOS) 설정 (uv)

### 사전 준비
- Python 3.10+
- uv 설치: https://docs.astral.sh/uv/

### 설치
```bash
cd /Users/andy/works/ai/finetuning-embedding
uv venv
uv sync
```

### 합성 데이터 생성
약자-전체/정의(positive) 및 비관련 용어(negative) 쌍을 포함한 JSONL 데이터를 생성합니다.
train/test split (80/20)으로 분리됩니다.

```bash
uv run python -m src.synth_data --pairs-per-term 6 --pairs-per-acronym 8
```

### 로컬 훈련(샘플)
macOS에서는 메모리 제약으로 배치 크기와 시퀀스 길이를 줄여 실행:
```bash
uv run python -m src.train --train_data data/train.jsonl --epochs 1 --batch_size 2 --max_seq_length 128 --use_cpu
```

### 모델 평가
훈련된 모델을 test 데이터셋으로 평가하여 정확도와 임베딩 품질을 확인:
```bash
uv run python -m src.evaluate --model_path outputs/bge-m3-kr --sample_size 500
```

## 2) Azure ML Studio 노트북

- 노트북: [notebooks/01_finetune_bge_m3_kr.ipynb](notebooks/01_finetune_bge_m3_kr.ipynb)
- 요구 환경 변수
  - AZURE_SUBSCRIPTION_ID
  - AZURE_RESOURCE_GROUP
  - AZUREML_WORKSPACE_NAME
  - AZUREML_COMPUTE_NAME (예: cpu-cluster)

노트북에서 합성 데이터 생성 → 데이터 자산 등록 → 실험/실행 제출 → 모델 등록 → 엔드포인트 배포까지 수행합니다.

## 3) Azure ML 커맨드 작업 (옵션)

`azureml/job.yml` 을 사용해 커맨드 잡을 제출할 수 있습니다.

```bash
az ml job create --file azureml/job.yml
```

## 4) MLOps 배포 워크플로우

GitHub Actions 기반 배포 워크플로우는 구독 ID가 필요하여, 구독 정보를 확인한 뒤 생성할 수 있습니다.

필요한 작업:
- OIDC 기반 Azure 로그인 설정
- 모델 등록 및 온라인 엔드포인트 생성/갱신

---

### 파일 구조
- [src/synth_data.py](src/synth_data.py): 합성 데이터 생성
- [src/train.py](src/train.py): BGE-M3 파인튜닝
- [src/evaluate.py](src/evaluate.py): 모델 평가
- [src/score.py](src/score.py): 배포용 스코어링 엔트리
- [data/terms_electronics_ko.txt](data/terms_electronics_ko.txt): 전기/전자 용어 목록
- [data/acronyms_ko.jsonl](data/acronyms_ko.jsonl): 약자-전체/정의 매핑(JSONL)
- [azureml/job.yml](azureml/job.yml): Azure ML 커맨드 작업
- [azureml/endpoint.yml](azureml/endpoint.yml): 엔드포인트 정의
- [azureml/deployment.yml](azureml/deployment.yml): 디플로이 정의
