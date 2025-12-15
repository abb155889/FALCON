# FALCON
[FALCON__Few_shot_Anomaly_detection_via_seLective_knowledge_distillation_and_Consecutive_mOdule_compositioN__(manuscript).pdf](https://github.com/user-attachments/files/24159886/FALCON__Few_shot_Anomaly_detection_via_seLective_knowledge_distillation_and_Consecutive_mOdule_compositioN__.manuscript.pdf)
[FALCON__Few_shot_Anomaly_detection_via_seLective_knowledge_distillation_and_Consecutive_mOdule_compositioN_(supplemental).pdf](https://github.com/user-attachments/files/24159891/FALCON__Few_shot_Anomaly_detection_via_seLective_knowledge_distillation_and_Consecutive_mOdule_compositioN_.supplemental.pdf)


FALCON: Few-shot Anomaly detection via seLective knowledge distillation and Consecutive mOdule compositioN

Few-shot Anomaly Detection (FSAD) aims to detect diverse
types of anomalies using only a few normal images and
knowledge distillation has proven to be a powerful approach
for defect detection and preventing over-generalization in
FSAD, as it enables the transfer of knowledge from a pre-
trained teacher to a student. However, it suffers from train-
ing instability and over-generalization in FSAD settings. To
address these issues, we proposes FALCON, a framework
that integrates selective knowledge distillation with consecu-
tive training module. FALCON injects noise via CAM-guided
masking to ensure that it is localized to object regions, facil-
itating robust learning against potential defects. It also trans-
fers mid-level features from the teacher network to both the
student and the autoencoder as distilled inputs, mitigating
representation collapse. In addition, integrating features from
the student and the autoencoder enables the model to simulta-
neously capture both local and global features. Moreover, we
add a discriminator module to enable the model to recognize
defects area. Consequently, FALCON achieves up to an 8%
improvement in image-level AUROC over previous state-of-
the-art methods, as demonstrated through comprehensive ex-
periments on FSAD benchmarks including MVTec, MPDD,
and ViSA. 

<img width="1450" height="506" alt="image" src="https://github.com/user-attachments/assets/e322ecac-766b-4c99-a0be-d1bc09aa4a75" />

# Get Started
  1. Install Python 3.9.23 Pytorch 2.0.0

  2. This experiment requires the data to be downloaded 
     
    MVTecAD: https://www.mvtec.com/company/research/datasets/mvtec-ad
    ViSA: https://registry.opendata.aws/visa/ 
    MPDD: https://github.com/stepanje/MPDD
  
  3. You can reproduce the experiment result by FALCON.py.

    git clone https://github.com/abb155889/FALCON.git
    cd FALCON
    # Edit dataset paths in run_experiments.sh before running
    bash run_experiments.sh


# Main Result
<!--
<img width="1450" height="506" alt="image" src="https://github.com/user-attachments/assets/1af51688-883e-4464-99d0-c648a341f73b" />
<img width="1450" height="506" alt="image" src="https://github.com/user-attachments/assets/9f75096a-81c4-472b-b9e6-31bc9ecc3a6e" />
-->

<p align="center">
  <img src="https://github.com/user-attachments/assets/1af51688-883e-4464-99d0-c648a341f73b" width="49%" />
  <img src="https://github.com/user-attachments/assets/9f75096a-81c4-472b-b9e6-31bc9ecc3a6e" width="49%" />
</p>

# 비교군 실험 시 유의사항
공정하고 일관된 비교 실험을 위해 아래 설정을 동일하게 적용합니다.

### 1. Validation Set 사용 여부
- FALCON은 validation set을 사용하지 않습니다.
- 학습에는 train image만 사용합니다.

### 2. Validation Set이 존재하는 비교 방법 설정
Validation set을 사용하는 비교 방법의 경우, FALCON과의 공정한 비교를 위해 다음과 같이 설정합니다.

- 필수인 경우 (e.g. EfficientAD)
  - 아래 방법 중 하나를 적용합니다.
    - Train set의 일부를 validation set으로 분할
    - Train set 크기의 10–20%에 해당하는 추가 데이터를 validation set으로 사용

- 옵션인 경우 (e.g. FewSOME)
  - Test set을 이용한 early stopping 옵션을 비활성화 합니다.

### 3. Backbone 구조 통일
- 모든 방법은 ResNet-18을 backbone으로 사용합니다.
- 자체 backbone 구조를 사용하는 EfficientAD의 경우:
  - ResNet-18의 1, 2, 3, 4번 layer를 사용하여 4-layer 구조와 대응시킵니다.
    
