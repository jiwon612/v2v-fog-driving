# 안개 환경 자율주행: YOLOv5 + V2V 통신 기반 CARLA 시뮬레이션

> 경희대학교 전자공학과 캡스톤디자인 (2025년 1학기)  
> 팀 V2eXpert | 이지원 · 정연수 | 지도교수: 최민석 교수님

---

<div align="center">
  <img src="https://github.com/user-attachments/assets/530a0a06-22e8-43ce-bbc8-0c0764096772" width="600"/>
</div>

## 개요

본 프로젝트는 **고가의 LiDAR·Radar 센서 없이** 안개 환경에서도 안전하게 주행할 수 있는 경량 자율주행 시스템을 제안합니다. **YOLOv5 기반 객체 탐지**와 **V2V(차량 간) 통신**을 결합하여, 후행 차량이 선행 차량과의 거리를 추정하고 위험 인지 주행 판단을 수행합니다.

핵심 기술은 **씬 매칭(Scene Matching)** 입니다. YOLOv5 중간 계층에서 ROI Align으로 추출한 특징 벡터를 코사인 유사도로 비교하여, V2V로 수신한 정보를 실제로 활용하기 전에 두 차량이 동일한 도로 장면을 보고 있는지 시각적으로 검증합니다.

---

## 시스템 구조

```
┌─────────────────────────────────────────────────────────┐
│                     CARLA 시뮬레이터                     │
│  Front Car 2 ─MQTT(차선)─► Front Car ─MQTT(속도+이미지)─► Rear Car │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─── 인지(Perception) ───┐   ┌─── 통신(Communication) ───┐
│  YOLOv5 객체 탐지       │   │  MQTT (20 Hz)              │
│  ROI Align (7×7)       │   │  속도 / 이미지 / 차선 정보  │
│  씬 매칭                │   └───────────────────────────┘
│  거리 추정              │
└──────┬─────────────────┘
       ▼
┌─── 판단(Decision-making) ───┐   ┌─── 제어(Control) ───┐
│  60m 미만  → 감속            │──►│  스로틀               │
│  60~75m   → 유지             │   │  브레이크             │
│  75m 초과  → 가속            │   │  조향                 │
│  30m 미만  → 차선 변경       │   │  Fail-Safe            │
└─────────────────────────────┘   └─────────────────────┘
```

---

## 주요 특징

- **YOLOv5 객체 탐지** — CARLA 안개 환경에서 가로수, 가로등, 표지판, 광고판 등을 탐지하는 커스텀 학습 모델
- **MQTT 기반 V2V 통신** — 선행 차량의 실시간 속도·이미지·차선 정보를 20Hz로 수신
- **ROI Align 씬 매칭** — 객체별 7×7 고정 크기 특징 맵을 추출하고 코사인 유사도로 두 차량의 시각적 일치성 검증
- **강화된 거리 추정** — V2V 속도 × 시간 차이로 초기 거리 추정 후, 매칭 객체의 바운딩 박스 면적 비율로 보정
- **위험 인지 주행 판단** — 추정 거리 기반 감속 / 가속 / 차선 변경 결정
- **Fail-Safe 메커니즘** — 씬 매칭 실패(유사도 < 0.75) 시 최저 속도 안전 모드로 자동 전환

---

## 거리 추정 수식

**초기 추정:**

$$D = v_f \cdot \Delta t$$

**보정 후 최종 추정:**

$$\hat{D} = v_f \cdot \Delta t - \alpha \cdot \left( \sqrt{\frac{1}{n}\sum_{i=1}^{n}\frac{A_{r,i}}{A_{f,i}}} - 1 \right)$$

- $v_f$: 선행 차량 평균 속도
- $\Delta t$: 씬 매칭으로 확인된 이미지 촬영 시점 간 시간 차이
- $A_{r,i}$ / $A_{f,i}$: 후행/선행 차량의 $i$번째 매칭 객체 바운딩 박스 면적
- $\alpha = 0.15$ (실험적으로 결정)

---

## 실험 결과

| 지표 | 값 |
|---|---|
| 거리 추정 정확도 (안개 없음, 보정 후) | 94.70% |
| 거리 추정 정확도 (안개 100%, 보정 후) | 94.18% |
| 씬 매칭 처리 시간 | 평균 ~0.3초 (실시간 충족) |
| 유지된 안전 주행 거리 | 60~75m |
| 카메라 단독 시 안개 탐지 가능 거리 | ~13.6m (비교군) |

**클래스별 객체 탐지 성능 (안개 환경):**

| 클래스 | AP Score | F1 Score |
|---|---|---|
| Tree | 0.939 | 0.915 |
| Streetlamp | 0.967 | 0.936 |
| Streetsign01 | 0.972 | 0.942 |
| Streetsign04 | 0.978 | 0.935 |
| Advertisement | 0.967 | 0.890 |
| Warning Accident | 0.946 | 0.903 |

---

## 파일 구조

```
v2v-fog-driving/
├── communication/              # MQTT 발행/구독 모듈
├── config/                     # 시뮬레이션 및 하이퍼파라미터 설정
├── control/                    # 차량 제어 로직 (스로틀, 브레이크, 조향)
├── detection/                  # YOLOv5 추론 + ROI Align + 씬 매칭
├── model/                      # 커스텀 학습된 YOLOv5 가중치
├── main_simulation.py          # ✅ 메인 실행 파일
├── experiment_image_final.py   # ✅ 최종 실험 스크립트
├── compare.py                  # 거리/씬 비교 유틸리티
├── prop_spawn_points_20m.csv   # Town04 객체 스폰 좌표
├── prop_spawn_points_20m_2.csv
├── requirements.txt
└── yolov5s.pt                  # YOLOv5s 사전 학습 가중치
```

---

## 설치 및 실행

### 사전 준비

- CARLA 0.9.15
- Python 3.8 이상
- MQTT 브로커 (예: Mosquitto) — `localhost`에서 실행

### 설치

```bash
git clone https://github.com/jiwon612/v2v-fog-driving.git
cd v2v-fog-driving
pip install -r requirements.txt
```

### 실행

1. CARLA 서버 실행 (`./CarlaUE4.sh` 또는 `CarlaUE4.exe`)
2. MQTT 브로커 실행 (`mosquitto`)
3. 메인 시뮬레이션 실행:

```bash
python main_simulation.py
```

실험 평가 실행:

```bash
python experiment_image_final.py
```

---

## 실험 환경

| 항목 | 내용 |
|---|---|
| 시뮬레이터 | CARLA 0.9.15 (Town04 고속도로 직선 구간) |
| 차량 | Tesla Model 3 블루프린트 |
| 카메라 | 800×600 RGB, FOV 90°, 20 FPS |
| 안개 밀도 조건 | 0, 20, 30, 40, 60, 100 |
| V2V 프로토콜 | MQTT @ 20 Hz |

---

## 향후 연구

- 레이더 융합을 통한 거리 추정 정확도 및 견고성 향상
- BranchyNet 통합으로 Early Exit 전략 구현 (연산량 감소 및 실시간성 강화)
- 안개 농도별 균형 잡힌 학습 데이터 구성으로 씬 매칭 일반화 성능 개선

---

## 인용

```
이지원, 정연수, "Fog-Aware Autonomous Driving via YOLOv5 and V2V Communication
in CARLA Simulator," 경희대학교 캡스톤디자인, 2025.
```

---

## 라이선스

본 프로젝트는 학술 목적으로 작성되었습니다. 저작권은 저자에게 있습니다.
