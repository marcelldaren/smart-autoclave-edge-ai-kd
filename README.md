# Smart Autoclave — Edge AI Instance Segmentation with Knowledge Distillation

This repository contains the dataset and training code accompanying the paper:

**"Design of an Edge AI-Driven Smart Autoclave System"**
Marcell D Febriyan, Winda Astuti, E. Sitepu, Ade Firdaus, Agus Sulistyo, Prima Yuriandro
BINUS ASO School of Engineering, Bina Nusantara University, Jakarta, Indonesia

## Overview

This work proposes an Edge AI-driven smart autoclave system that automatically identifies surgical instruments via instance segmentation and selects the appropriate sterilization program without operator input. Knowledge Distillation (KD) is used to compress high-accuracy segmentation models for deployment on an NVIDIA Jetson Orin Nano.

Three teacher-student KD pipelines were evaluated, each representing a distinct architectural paradigm:

| Pipeline | Teacher | Student | Paradigm |
|---|---|---|---|
| A | YOLOv11x-seg | YOLOv11s-seg | CNN → CNN |
| B | Swin-S Mask R-CNN | RTMDet-S | Transformer → CNN |
| C | Mask2Former Swin-B | Mask2Former ResNet-50 | Transformer → Transformer |

## Repository Structure

```
.
├── dataset/
│   ├── images/                  # 82 original surgical instrument images
│   └── annotations.json         # COCO-format instance segmentation annotations
├── training_code/
│   ├── pipeline_A_yolo/         # YOLOv11x → YOLOv11s KD training (Ultralytics)
│   ├── pipeline_B_mmdet/        # Swin-S Mask R-CNN → RTMDet-S KD training (MMDetection)
│   └── pipeline_C_detectron2/   # Mask2Former Swin-B → ResNet-50 KD training (Detectron2)
└── README.md
```

## Dataset

The dataset comprises **123 original images** self-collected and annotated by the authors, covering 10 surgical/laboratory instrument classes:

- Beaker
- Clamp-Dressing Forceps
- Dental-HandPiece
- Erlenmeyer
- Hemostats Forcep
- Reaction Tube
- Retractor Volkmann
- Scalpel Handle
- Scissor
- Wire Tray

Annotations are provided in COCO instance segmentation format. The 123 original images were offline-augmented (×20 multiplier via Albumentations) for model training; only the original, pre-augmentation images are included in this repository.

## Training Code

Each pipeline folder contains the Google Colab notebook used to train the teacher, baseline student, and KD student models for that pipeline:

- **Pipeline A** — trained with Ultralytics (YOLO), SGD optimizer, CWD + KL-Divergence distillation loss
- **Pipeline B** — trained with MMDetection, AdamW optimizer, CWD distillation loss with adapter layers
- **Pipeline C** — trained with Detectron2/Mask2Former, AdamW optimizer, combined Logit + CWD + Query KD loss

All models were trained on a single NVIDIA T4 GPU via Google Colab.

