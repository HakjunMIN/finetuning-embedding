# Local Serving Guide for BGE-M3 Korean Embedding Model

이 문서는 파인튜닝된 BGE-M3 Korean embedding 모델을 로컬에서 서빙할 때 사용할 수 있는 선택지와 권장 아키텍처를 정리합니다.

## 모델 구조 요약

이 프로젝트의 모델은 `BAAI/bge-m3` 기반의 SentenceTransformers 모델입니다.

학습 시 LoRA를 적용한 뒤 최종 저장 단계에서 LoRA weight를 base model에 병합합니다.

- 모델 계열: BGE-M3 / XLM-RoBERTa encoder
- 작업 유형: sentence embedding
- 출력 차원: 1024
- 최대 시퀀스 길이: 128
- pooling 방식: CLS token pooling
- normalize: L2 normalization
- 현재 서빙 방식: `SentenceTransformer.encode(..., normalize_embeddings=True)`

저장된 SentenceTransformers pipeline은 다음 구조입니다.

```text
Transformer
  -> Pooling
  -> Normalize
```

따라서 ONNX 등으로 변환할 때 transformer forward만 맞추는 것으로는 충분하지 않고, 반드시 CLS pooling과 L2 normalization까지 동일하게 구현해야 합니다.

```python
embedding = last_hidden_state[:, 0, :]
embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
```

## 결론

이 모델에는 다음 순서를 추천합니다.

```text
1. FastAPI + SentenceTransformer
2. ONNX Runtime + FastAPI
3. ONNX Runtime int8 quantization
4. GPU 운영이 필요하면 Triton + ONNX
5. Apple Silicon 전용이면 MLX/Core ML 별도 벤치마크
```

vLLM은 이 모델의 1순위 선택지는 아닙니다.

## 1. FastAPI + SentenceTransformer

가장 단순한 로컬 서빙 방식입니다.

현재 Azure ML scoring 코드와 거의 같은 방식으로 구현할 수 있습니다.

장점:

- 구현이 가장 쉬움
- 모델 품질 차이가 없음
- 디버깅이 편함
- 현재 `SentenceTransformer.encode()` 결과와 동일

단점:

- PyTorch eager mode라 CPU 성능이 최적은 아님
- cold start가 무거울 수 있음
- 고QPS 서빙에는 한계가 있음

추천 상황:

- 로컬 개발
- 데모
- 낮은 요청량
- 정확도 기준선 측정

## 2. ONNX Runtime + FastAPI

이 프로젝트의 기본 권장 운영안입니다.

BGE-M3는 encoder-only embedding model이므로 ONNX Runtime과 잘 맞습니다.

권장 구성:

```text
Client
  -> FastAPI
  -> Hugging Face tokenizer
  -> ONNX Runtime inference
  -> CLS pooling
  -> L2 normalization
  -> embeddings response
```

장점:

- PyTorch보다 CPU latency/throughput이 좋아질 가능성이 큼
- Docker, Linux, macOS 등으로 이식하기 쉬움
- 운영 구조가 단순함
- int8 dynamic quantization 적용 가능

주의사항:

- tokenizer는 별도로 실행해야 함
- ONNX 출력에 pooling/normalize가 포함되지 않으면 직접 구현해야 함
- 변환 후 반드시 SentenceTransformer 결과와 cosine similarity를 비교해야 함

품질 검증 기준:

```text
SentenceTransformer embedding vs ONNX embedding cosine similarity
```

동일 입력에 대해 cosine similarity가 거의 1에 가까워야 합니다.

## 3. ONNX Runtime int8 quantization

CPU 서빙 성능을 더 높이고 싶다면 ONNX 변환 후 dynamic int8 quantization을 검토합니다.

장점:

- CPU inference 속도 개선 가능
- 메모리 사용량 감소 가능
- embedding 모델에서는 품질 저하가 작을 수 있음

주의사항:

- 임베딩 분포가 바뀔 수 있음
- 검색 품질이 떨어질 수 있음
- 반드시 평가 스크립트로 품질 비교 필요

권장 검증:

```text
fp32 SentenceTransformer
fp32 ONNX Runtime
int8 ONNX Runtime
```

세 결과를 latency, throughput, 검색 정확도 기준으로 비교합니다.

## 4. Triton Inference Server

GPU 환경에서 여러 클라이언트 요청을 안정적으로 처리해야 한다면 Triton을 검토할 수 있습니다.

추천 구성:

```text
Client
  -> FastAPI tokenizer/batching layer
  -> Triton ONNX backend
  -> CLS pooling + normalize
  -> embeddings
```

장점:

- GPU 운영에 적합
- dynamic batching 지원
- 모델 버전 관리 가능
- 여러 모델을 함께 운영하기 좋음

단점:

- 로컬 단일 모델 서빙에는 과할 수 있음
- tokenizer 처리를 별도로 설계해야 함
- 설정 복잡도가 올라감

추천 상황:

- NVIDIA GPU 서버
- 높은 QPS
- 여러 모델 운영
- 운영 환경에서 batching이 중요한 경우

## 5. TensorRT

TensorRT는 NVIDIA GPU에서 latency를 극한으로 줄이고 싶을 때 후보입니다.

다만 이 프로젝트의 첫 번째 선택지로는 추천하지 않습니다.

이유:

- 변환과 검증 비용이 큼
- dynamic shape, tokenizer, pooling 처리까지 신경써야 함
- ONNX Runtime 또는 Triton ONNX backend로도 충분한 경우가 많음

추천 순서:

```text
ONNX Runtime GPU
Triton + ONNX
TensorRT
```

TensorRT는 위 단계로도 성능이 부족할 때 고려합니다.

## 6. MLX / Apple Silicon

Apple Silicon Mac에서만 로컬 서빙할 계획이라면 MLX 또는 Core ML도 검토할 수 있습니다.

다만 ONNX Runtime이 항상 Apple Silicon에서 최고의 선택이라고 보기는 어렵습니다.

선택 기준:

```text
이식성/운영 안정성: ONNX Runtime
Apple Silicon 전용 최적화: MLX 또는 Core ML
빠른 개발/검증: SentenceTransformer
```

MLX 장점:

- Apple Silicon unified memory와 Metal 경로에 최적화 가능
- Mac 전용 배포에서는 성능상 유리할 수 있음

MLX 주의사항:

- BGE-M3 / XLM-RoBERTa encoder 지원을 확인해야 함
- SentenceTransformers의 pooling/normalize 로직을 직접 맞춰야 할 수 있음
- 변환 후 embedding 품질 검증이 필요함

Apple Silicon에서 최종 선택은 벤치마크로 결정하는 것이 좋습니다.

```text
SentenceTransformer PyTorch
ONNX Runtime fp32
ONNX Runtime int8
MLX 또는 Core ML
```

## 7. vLLM

vLLM은 이 모델에는 추천 우선순위가 낮습니다.

vLLM이 강한 영역은 Llama, Qwen, Mistral 같은 decoder-only 생성 모델 서빙입니다.

vLLM의 핵심 최적화:

- KV cache
- continuous batching
- paged attention
- token generation throughput 최적화

하지만 이 프로젝트의 모델은 생성형 LLM이 아니라 encoder-only embedding model입니다.

즉, 입력 문장을 한 번 forward 해서 embedding을 뽑는 구조이므로 vLLM의 장점 대부분을 활용하기 어렵습니다.

vLLM을 고려할 수 있는 경우:

- 이미 vLLM 기반 인프라가 있음
- 생성형 LLM과 embedding API를 같은 방식으로 운영하고 싶음
- OpenAI-compatible API 통일이 중요함

그 외에는 ONNX Runtime, TEI, Triton 쪽이 더 자연스럽습니다.

## 8. Hugging Face Text Embeddings Inference

전용 embedding serving solution이 필요하다면 Hugging Face Text Embeddings Inference도 후보입니다.

장점:

- embedding model serving에 특화
- batching 및 production serving 기능 제공
- OpenAI-compatible embedding API 구성이 쉬움

주의사항:

- 현재 파인튜닝된 SentenceTransformers 모델과의 호환성 확인 필요
- 커스텀 pooling/normalize 결과가 기존 `SentenceTransformer.encode()`와 같은지 검증 필요

## 권장 로드맵

### Phase 1: 기준선 만들기

FastAPI + SentenceTransformer로 로컬 API를 먼저 만듭니다.

목표:

- 현재 모델이 정상 동작하는지 확인
- latency와 throughput 기준선 측정
- embedding 품질 기준선 저장

### Phase 2: ONNX Runtime 변환

모델을 ONNX로 변환하고 FastAPI 서버에서 ONNX Runtime으로 inference합니다.

검증 항목:

- 같은 입력에 대한 embedding cosine similarity
- latency
- throughput
- 메모리 사용량
- batch size별 성능

### Phase 3: int8 quantization

ONNX 모델에 dynamic int8 quantization을 적용합니다.

검증 항목:

- 검색 품질 하락 여부
- cosine similarity 분포 변화
- latency 개선 폭
- 메모리 절감 폭

### Phase 4: 운영 환경별 확장

환경에 따라 다음 중 선택합니다.

```text
CPU 서버: ONNX Runtime int8
Intel CPU: OpenVINO 검토
Apple Silicon: MLX/Core ML 벤치마크
NVIDIA GPU: Triton + ONNX
고성능 GPU 최적화: TensorRT
```

## 최종 추천

현재 프로젝트 기준 최종 추천은 다음과 같습니다.

```text
개발/데모:
FastAPI + SentenceTransformer

로컬 CPU 운영:
FastAPI + ONNX Runtime fp32 또는 int8

Apple Silicon 전용:
ONNX Runtime과 MLX/Core ML 비교 후 선택

GPU 운영:
Triton + ONNX

vLLM:
비추천. 생성형 LLM을 함께 운영하는 특별한 경우에만 검토
```

이 모델은 embedding encoder이므로, 생성형 LLM serving stack보다는 encoder inference에 최적화된 stack을 선택하는 것이 좋습니다.
