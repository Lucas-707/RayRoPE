# RayRoPE Optimization (CUDA Enhanced)

이 저장소는 연세대학교 V-Lab에서 진행된 연구의 일환으로, 기존 PyTorch로 구현된 RayRoPE(Ray-based Rotary Positional Encoding) 라이브러리의 연산 병목을 제거하기 위해 커스텀 CUDA 커널을 직접 작성하여 성능과 메모리 효율을 극대화한 최적화 버전입니다.

## 🚀 주요 최적화 특징 (Key Optimizations)

기존 파이토치 환경에서 발생하던 막대한 중간 텐서 생성과 역전파(Backward) 시의 병목 현상을 해결하기 위해 커스텀 CUDA 커널 통합(Kernel Fusion)을 진행했습니다.

### 1. CUDA Kernel Fusion (커널 병합)
* **내용**: 복잡한 3D 기하학적 좌표 변환(`_prepare_apply_fns`)과 주파수 대역 계산을 `FusedGeometry_KV`와 `FusedRayRoPEFunction` 커널 내부로 통합했습니다.
* **이점**: 불필요한 VRAM 읽기/쓰기(Memory Trip)를 최소화하고, 중간 텐서 생성을 방지하여 GPU 메모리를 대폭 절약합니다.

### 2. Gather Pattern & Memory Coalescing (메모리 최적화)
* **내용**: 역전파 시 수십만 개의 스레드가 동일한 메모리에 접근하면서 발생하던 `atomicAdd` 충돌 병목을 원천 차단하기 위해, 대표 스레드가 값을 긁어모으는(Gather) 패턴으로 커널을 분리 재설계했습니다.
* **결과**: 연속된 메모리 접근(Coalescing)을 보장하며 순수 연산 처리 속도를 극적으로 끌어올렸습니다.

---

## 📊 Benchmark Results

다음은 벤치마크 테스트를 통해 측정된 성능 향상 지표입니다.

**1. 순수 연산 (`_prepare_apply_fns`) 성능 비교**
| Metrics | Official (PyTorch) | Optimal (CUDA) | Improvement |
| :--- | :---: | :---: | :---: |
| **Peak Memory** | 0.081 | **0.020** | **~75% Decrease** |
| **End Memory** | 0.080 | **0.019** | **~76% Decrease** |
| **Avg. Time (ms)** | 9728.62 ms | **61.66 ms** | **~157x Faster** |

**2. 전체 Attention (Forward + Backward) 속도 비교**
| Metrics | Official (PyTorch) | Optimal (CUDA) | Improvement |
| :--- | :---: | :---: | :---: |
| **Forward Time** | 8.687 ms | **5.768 ms** | **~33% Faster** |
| **Backward Time** | 17.082 ms | **17.082 ms** | **Same (Maintained)** |
| **Total Time** | 25.769 ms | **22.850 ms** | **~11% Faster** |

> **Verification**: 극한의 델타(Delta) 조건에서도 수학적으로 매끄러운(Analytical) 극한값을 적용하여 텐서 코어 계산 결과가 PyTorch 원본과 수치적으로 완벽하게 일치함을 검증했습니다.

---

## 🛠 Modified & Added Key Files
* `cuda/csrc/rayrope_coeffs.cu`: RoPE 계수 적용 및 역전파 Gather 패턴 최적화 커널 구현.
* `cuda/csrc/rayrope_geometry_d_pj__0_3d.cu`: 3D 기하학적 좌표 변환 커널.
* `cuda/_wrapper.py`: 파이토치 `autograd.Function`을 상속한 메모리 연속성(Contiguous) 보장 커널 인터페이스.
* `rayrope_cuda.py`: 기존 모델 구조를 대체하여 최적화된 커널을 매끄럽게 연결하는 래퍼(Wrapper) 모듈.

<br>
<br>

---

# RayRoPE Optimization (CUDA Enhanced)

As part of the research conducted at Yonsei University V-Lab, this repository provides a highly optimized version of the Ray-based Rotary Positional Encoding (RayRoPE). We replaced the original PyTorch implementation with custom CUDA kernels to maximize computational performance and memory efficiency.

## 🚀 Key Optimizations

To address the massive intermediate tensor generation and backward pass bottlenecks present in the native PyTorch environment, we integrated custom CUDA Kernel Fusion.

### 1. CUDA Kernel Fusion
* **Details**: The complex 3D geometric coordinate transformations (`_prepare_apply_fns`) and frequency band calculations were fully fused into the `FusedGeometry_KV` and `FusedRayRoPEFunction` custom kernels.
* **Benefits**: Minimizes unnecessary VRAM read/write overhead (Memory Trip) and significantly saves GPU memory by preventing intermediate tensor allocations.

### 2. Gather Pattern & Memory Coalescing
* **Details**: To completely eliminate the `atomicAdd` collision bottlenecks caused by hundreds of thousands of threads accessing the same memory during the backward pass, we redesigned and split the kernel using a Gather pattern where a single representative thread accumulates the gradients.
* **Results**: Ensures memory coalescing and dramatically boosts pure computational throughput.

---

## 📊 Benchmark Results

The following metrics demonstrate the performance improvements measured during benchmark testing.

**1. Pure Computation (`_prepare_apply_fns`) Benchmark**
| Metrics | Official (PyTorch) | Optimal (CUDA) | Improvement |
| :--- | :---: | :---: | :---: |
| **Peak Memory** | 0.081 | **0.020** | **~75% Decrease** |
| **End Memory** | 0.080 | **0.019** | **~76% Decrease** |
| **Avg. Time (ms)** | 9728.62 ms | **61.66 ms** | **~157x Faster** |

**2. Overall Attention (Forward + Backward) Benchmark**
| Metrics | Official (PyTorch) | Optimal (CUDA) | Improvement |
| :--- | :---: | :---: | :---: |
| **Forward Time** | 8.687 ms | **5.768 ms** | **~33% Faster** |
| **Backward Time** | 17.082 ms | **17.082 ms** | **Same (Maintained)** |
| **Total Time** | 25.769 ms | **22.850 ms** | **~11% Faster** |

> **Verification**: We verified that the results perfectly match the original PyTorch numerical outputs by applying mathematically smooth (analytical) limits even under extreme delta conditions.

---

## 🛠 Modified & Added Key Files
* `cuda/csrc/rayrope_coeffs.cu`: Optimized kernels for RoPE coefficient application and backward Gather pattern.
* `cuda/csrc/rayrope_geometry_d_pj__0_3d.cu`: 3D geometric coordinate transformation kernel.
* `cuda/_wrapper.py`: PyTorch `autograd.Function` wrapper ensuring memory contiguity.
* `rayrope_cuda.py`: Python wrapper module cleanly replacing the original model structure with the optimized kernels.