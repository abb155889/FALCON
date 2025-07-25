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
