# 논문 주요 Figure 생성 코드
해당 디렉토리에는 논문에 사용된 주요 Figure를 재현하기 위한 코드가 포함되어 있습니다.

## Noise Injection Visualization

<img width="1135" height="623" alt="Noise Injection Image" src="https://github.com/user-attachments/assets/f71bf753-5e02-4bc0-8570-cf84496d7ea0" />

- Script: `make_noise_injection.py`  
- 설명:  입력 이미지에 노이즈를 주입한 결과를 시각화하기 위한 코드입니다.

- 주요 하이퍼파라미터
  - `output_dir`: 결과 이미지 저장 경로
  - `data_limit`: 저장할 이미지 개수
  - `visualize_noise_image`: 이미지 저장 여부 (`True`로 설정 시 저장)

- 참고 사항
  - 노이즈는 매 실행마다 랜덤한 위치에 주입됩니다.
  - 하나의 이미지에 대해 다양한 노이즈 주입 결과를 보고 싶다면 `epochs` 값을 증가시키세요.

---

## Margin Plot

<img width="803" height="290" alt="Margin Plot" src="https://github.com/user-attachments/assets/d7a74927-28ae-4b4b-bfe9-7054a2bdd5e5" />

- Scripts
  - `margin_on_training.py`
  - `make_plot.py`

- 설명
  - 학습 과정에서의 margin 변화를 기록하고 시각화하기 위한 코드입니다.
  - 본 구현은 Google Research의 공식 구현을 기반으로 합니다.

- 실행 방법
  1. Google Research 저장소를 git clone
      ```bash
      git clone https://github.com/google-research/google-research.git
      ```
  2. `google-research` 저장소를 동일한 root 디렉터리에 위치
  3. `margin_on_training.py` 실행
  4. 실행 후 `train_output_dir`에 생성된 CSV 파일을 `make_plot.py`에 입력하여 plot 생성
