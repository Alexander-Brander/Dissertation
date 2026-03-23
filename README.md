# Knee Osteoarthritis X-Ray Classification with Spatial Transformer Networks

**BSc Computing Science Dissertation - University of Aberdeen (2024)**

Investigating whether Spatial Transformer Networks (STNs) can improve CNN-based classification of knee osteoarthritis severity from X-ray images by correcting spatial variabilities such as rotation, scale, and positioning.

> **Note:** This repository contains the original dissertation code submitted in April 2024. A substantially revised and improved version of this project is available at [knee-stn](https://github.com/Alexander-Brander/knee-stn), which addresses the training instability issues documented here and adds proper transfer learning, data augmentation, class-weighted loss, GradCAM explainability, and ResNet-50.

## Overview

The project explores integrating an STN module into a pretrained ResNet-18 CNN for classifying knee X-ray images into Kellgren-Lawrence (KL) grades 0–4. The dissertation documents three implementation iterations - two that failed to train effectively, and a final working version using a pretrained ResNet-18 backbone.

### Key Findings (Original Dissertation)

- The baseline ResNet-18 achieved ~48% test accuracy on a reduced dataset subset
- The STN-enhanced model showed marginally higher precision in some configurations but did not consistently outperform the baseline on overall accuracy
- STN transformations were subtle (slight translations) after longer training, and more aggressive in early epochs
- The STN model showed promising signs for Grade 3 and Grade 4 classification where spatial features are more pronounced
- Training instability was a persistent challenge across all implementations, primarily due to simultaneous STN + CNN training without a phased approach

### What Went Wrong and What I Learned

The dissertation honestly documents the iterative process of getting STNs to work with medical images:

1. **First implementation (TensorFlow/Keras):** Custom grid generator and bilinear sampler produced blank or heavily skewed images. Likely caused by incorrect manual implementations of these components.
2. **Second implementation (PyTorch, custom CNN):** Used PyTorch's built-in `affine_grid` and `grid_sample`, which fixed the transformation output, but the CNN was too shallow to learn meaningful features from the knee X-rays. Model predicted a single class for all inputs.
3. **Third implementation (PyTorch, pretrained ResNet-18):** Using a pretrained backbone solved the feature extraction problem. The model trained successfully and produced meaningful classifications and visible transformations.

These failures informed the [revised version](https://github.com/Alexander-Brander/knee-stn), which uses two-phase training with differential learning rates to prevent the STN collapse problem.

## Dataset

[Knee Osteoarthritis Severity Grading Dataset](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity) (Chen, 2018) - X-ray images labelled with KL grades 0–4.

The dissertation used two reduced subsets due to Google Colab RAM limitations:
- **Dataset X:** 940 training / 155 validation images
- **Dataset L:** 865 training / 135 validation images (evenly distributed across grades)

## Project Structure

```
Dissertation/
├── STN_Implementation_3.ipynb    # Final working implementation (ResNet-18 + STN)
├── Baseline_CNN.ipynb            # Baseline ResNet-18 without STN
└── README.md
```

## Setup

The notebooks were developed in Google Colab with GPU runtime.

### Dependencies
- Python 3.10+
- PyTorch, torchvision
- NumPy, Matplotlib, scikit-learn

### Running
1. Upload the dataset to Google Drive
2. Open the notebooks in Google Colab
3. Set runtime to GPU (Runtime → Change runtime type → T4 GPU)
4. Update the `base_dir` path to point to your dataset location

## Revised Version

The [revised project](https://github.com/Alexander-Brander/knee-stn) addresses the limitations of this dissertation with:

| Aspect | This Version (2024) | Revised Version (2026) |
|---|---|---|
| Architecture | ResNet-18 | ResNet-50 |
| Dataset | Reduced subsets (~900 images) | Full dataset (5,778 training) |
| Training | Single-phase, fixed LR | Two-phase transfer learning, LR scheduling |
| Augmentation | None | Random flip, rotation, colour jitter |
| Class imbalance | Not addressed | Inverse-frequency weighted loss |
| STN training | Simultaneous (unstable) | Two-phase with differential LR |
| Early stopping | Not implemented | Best model by validation F1 |
| Explainability | Visual inspection only | GradCAM, difference maps |
| Best accuracy | ~48% | 64.9% |

## References

- Jaderberg, M. et al. (2015). *Spatial Transformer Networks.* NeurIPS.
- Chen, P. et al. (2019). *Fully automatic knee osteoarthritis severity grading using deep neural networks.* Computerized Medical Imaging and Graphics.
- Ahmed, S.M. & Mstafa, R.J. (2022). *Identifying severity grading of knee osteoarthritis from X-ray images.* Diagnostics.
