# BGE-M3 Korean Embedding Fine-tuning (Azure ML Studio)

이 프로젝트는 Azure ML Studio에서 노트북으로 실험(Experiment)과 실행(Run)을 수행하고, 이후 MLOps 배포 워크플로우까지 확장할 수 있는 템플릿입니다. 데이터셋은 전기/전자 용어 기반으로 합성 생성됩니다.

## 프로젝트 목적 및 인사이트

### 1. 도메인 특화 임베딩의 필요성
범용 임베딩 모델(GPT, BERT 등)은 일반 텍스트에서 좋은 성능을 보이지만, **전기/전자 공학과 같은 전문 도메인**에서는 한계가 있습니다:
- 약자(PWM, ADC, MOSFET)와 전체 용어(펄스 폭 변조) 간 연결 부족
- 도메인 특화 용어(역률보정, 스위칭전원) 간 의미적 관계 파악 미흡
- 기술 문서 검색/매칭 시 정확도 저하

### 2. Contrastive Learning을 통한 임베딩 공간 최적화
본 프로젝트는 **triplet 구조**(query, positive, negative)로 학습하여:
- **Positive 쌍**: 약자와 정의, 용어와 설명을 임베딩 공간에서 밀집(pull)
- **Negative 쌍**: 비관련 용어는 멀리 배치(push)
- 결과: 의미적으로 유사한 개념은 가깝게, 무관한 개념은 멀게 배치되는 효과적인 임베딩 공간 구축

### 3. 합성 데이터의 장점
실제 레이블링 없이 템플릿 기반 합성 데이터로 학습:
- **비용 효율**: 수작업 레이블링 불필요
- **확장성**: 용어 추가만으로 데이터셋 확장 가능
- **일관성**: 템플릿으로 균일한 품질 보장
- **빠른 프로토타이핑**: 도메인 변경(의료, 금융 등) 시 템플릿만 수정

### 4. 활용 시나리오
- **기술 문서 검색**: "PWM이 뭐야?" → 관련 매뉴얼/논문 검색
- **FAQ 매칭**: 사용자 질문을 기존 FAQ와 의미적으로 매칭
- **용어 추천**: 입력 용어와 관련된 개념 자동 제안
- **지식 그래프**: 임베딩 유사도 기반 용어 간 관계 시각화

### 5. MLOps 워크플로우
Azure ML Studio 기반 전체 파이프라인:
- 실험 추적 및 재현성 확보
- 모델 버전 관리 및 배포 자동화
- 엔드포인트를 통한 실시간 임베딩 서비스

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
