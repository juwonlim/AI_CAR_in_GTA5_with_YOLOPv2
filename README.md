# AI_CAR_in_GTA5_with_YOLOPv2

# GTA5 YOLOPv2 자율주행 시스템

GTA5 환경에서 자율주행 시스템을 구현하기 위한 실험용 프로젝트입니다.  
YOLOPv2를 활용하여 **차선 인식, 도로 추론, 객체 감지**를 동시에 수행하며,  
화면 내 속도 인식 및 거리 기반 감속 제어 기능까지 통합하고 있습니다.

##  주요 기능

- YOLOPv2 기반 실시간 차선 및 객체 감지
- 가상 중심선 생성 및 시각화
- 차량(바이크 포함) 자율 전진/조향
- OCR(Tesseract)을 통한 속도 인식 및 제어
- 객체와의 거리 기반 감속 로직

## 실행 환경

- Python 3.8+
- PyTorch
- OpenCV
- pytesseract
- GTA5 (PC 버전)
- mss / pywin32 / pydirectinput 등

##  주의사항

- `yolopv2.pt` 모델 파일은 GitHub 업로드 제한(100MB)을 초과하므로 **저장소에 포함되지 않습니다**.
- 직접 [YOLOPv2 원본 저장소](https://github.com/hustvl/YOLOP) 등에서 모델 파일을 다운받아  
  `data/weights/yolopv2.pt` 경로에 배치해야 합니다.

##  디렉토리 구조 예시
AI_CAR_in_GTA5_with_YOLOPv2/
├── yolopv2_model_demo_rev04.py
├── driving/
│ └── drive_with_yolopv2_control_rev05.py
├── yolopv2_inference/
│ ├── data/
│ │ └── weights/
│ └── utils/



##  향후 계획

- 차선 사라짐에 대한 대응 로직 추가
- 속도 인식 정확도 향상
- Steering 보정 및 HDA-like 주행 로직 개선

---

> 개발 중인 실험 프로젝트이며, 학습 목적의 비상업적 용도로 사용됩니다.

