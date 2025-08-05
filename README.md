# FALCON

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
     bash run_experiments.sh

# Main Result
<img width="1054" height="572" alt="image" src="https://github.com/user-attachments/assets/ef93ce84-0c9b-412e-b93f-b80843c962f0" />

    
    
