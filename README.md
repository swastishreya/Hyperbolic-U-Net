# Hyperbolic U-Net

A PyTorch implementation of Hyperbolic U-Net architectures for robust medical image segmentation. This repository contains both Euclidean and Hyperbolic variants of U-Net models, including support for flexible curvature settings and various loss functions optimized for medical image analysis.

## Overview

This project implements state-of-the-art U-Net models for medical image segmentation with the following features:

- **Euclidean U-Net**: Traditional Euclidean space U-Net architecture
- **Hyperbolic U-Net**: Hyperbolic geometry-based U-Net for improved robustness
- **Flexible Architectures**: Support for nested U-Net variants
- **Multiple Loss Functions**: Dice, Tversky, Focal, Cross-Entropy, and combinations thereof
- **Riemannian Optimization**: Support for Riemannian Adam and SGD optimizers for hyperbolic models
- **Multi-Dataset Support**: Built-in support for ISIC, REFUGE2, OCTA, CVC-ColonDB, and other medical imaging datasets
- **Experiment Tracking**: Integration with Weights & Biases (W&B) for experiment monitoring

## Requirements

### System Requirements
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- Conda (recommended for environment management)

### Key Dependencies
- PyTorch 2.0+
- torchvision
- albumentations (data augmentation)
- nibabel (NIfTI format support)
- scikit-image
- medpy (medical image metrics)
- wandb (experiment tracking)
- hypll (Hyperbolic Learning Library)

## Installation

### Using environment.yml (Recommended)

The repository includes an `environment.yml` file with all dependencies pre-configured:

```bash
conda env create -f environment.yml
conda activate hypunet
```

### Fallback: Manual Installation

If the `environment.yml` file fails, you can set up the environment manually:

#### Step 1: Create Conda Environment

```bash
conda create -n hypunet python=3.10
conda activate hypunet
```

#### Step 2: Install PyTorch

Choose the appropriate command based on your system:

**For CUDA 12.1 (Recommended):**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**For CUDA 11.8:**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**For CPU-only:**
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

#### Step 3: Install Key Dependencies

```bash
pip install torch==2.4.0 torchvision==0.19.0 wandb==0.13.5 albumentations==2.0.0 nibabel==5.3.2 scikit-image==0.25.2 medpy==0.5.2 hypll==0.1.1 pandas numpy scipy tqdm
```


## Data Preparation

The repository supports multiple medical imaging datasets with automatic dataloader configuration. Please download them from the following links:

### Supported Datasets
- **ISIC** / **ISIC18**: Skin lesion segmentation
- **REFUGE2** : Optic disc/cup segmentation
- **OCTA**: Optical Coherence Tomography Angiography
- **CVC-ColonDB**: Polyp segmentation
- **DRIVE**: Retinal vessel segmentation
- **BUSI**: Breast Ultrasound Images
- **KVASIR**: Gastrointestinal polyp dataset
- **PROSTATE**: Prostate segmentation
- **PANXRAYS**: Panoramic X-rays
- **nnU-Net Format**: Any dataset in nnU-Net format with `imagesTr/labelsTr` and `imagesTs/labelsTs` directories

### Dataset Structure Example

The `MakeDataset` class automatically handles various dataset formats. To verify the dataset formats check the [make_dataset.py](dataloading/make_dataset.py) file. Please ensure that the `img_suffix` and `mask_suffix` for the input image names and target masks names respectively are entered correctly. We have given the example of ISIC16 dataset below:

Example dataset structure for ISIC16:

```
ISIC/
├── ISBI2016_ISIC_Part1_Training_Data/                  # Training/input images
├── ISBI2016_ISIC_Part1_Training_GroundTruth/           # Training/target masks
├── ISBI2016_ISIC_Part1_Test_Data/                      # Test images
└── ISBI2016_ISIC_Part1_Test_GroundTruth/               # Test masks
```

## Training

### Example: Training Hyperbolic U-Net

```bash
python train.py \
  --project medical_seg \
  --dataset ./datasets/ISIC \
  --model hyp \
  --channels 3 \
  --classes 2 \
  --init_feats 8 \
  --depth 4 \
  --optim adam \
  --loss dice+focal \
  --epochs 30 \
  --batch-size 8 \
  --learning-rate 1e-3 \
  --scale 1.0 \
  --curvature 0.1 \
  --validation 10 \
  --trainable
```

### Example: Training Standard U-Net

```bash
python train.py \
  --project medical_seg \
  --dataset ./datasets/REFUGE2 \
  --model euc \
  --channels 3 \
  --classes 2 \
  --optim adam \
  --loss tversky+focal \
  --alpha 0.3 \
  --beta 0.7 \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 1e-3 \
  --scale 1.0 \
  --validation 10 \
  --amp
```

## Evaluation and Testing

### Testing Script

The `test.py` script provides comprehensive evaluation with various metrics:

```bash
python test.py \
  --dataset <path_to_dataset> \
  --model_path <path_to_checkpoint> \
  --model <model_type> \
  --channels 3 \
  --classes 2 \
  --init_feats 8 \
  --depth 4 \
  --batch-size 8 \
  --scale 1.0 \
  --curvature 0.1
  [additional arguments]
```

### Evaluation Metrics

The framework automatically computes:
- **Dice Score**: F1 score for segmentation
- **IoU (Intersection over Union)**: Jaccard index
- **Sensitivity (Recall)**: True positive rate
- **Specificity**: True negative rate
- **Hausdorff Distance**: Surface distance metric
- **Hausdorff Distance 95**: 95th percentile of surface distances

## Model Architectures

### Euclidean U-Net (`euc`)
Standard U-Net architecture operating in Euclidean space with:
- Symmetric encoder-decoder structure
- Skip connections
- Configurable depth and initial feature maps
- Optional bilinear upsampling

### Hyperbolic U-Net (`hyp`)
U-Net operating in hyperbolic space (Poincaré ball model) with:
- Riemannian operations on hyperbolic manifolds
- Configurable manifold curvature
- Optional trainable curvature parameters
- Improved robustness for hierarchical data

### Nested U-Net (`nunet`, `hnunet`)
U-Net with nested skip connections (dense connections within the architecture)

### Nested U-Net with Deep Supervision (`nestedunet`, `hnestedunet`)
Nested architectures with deep supervision for multi-scale learning

## Loss Functions

Available loss combinations for different segmentation tasks:

- **Dice + Cross-Entropy** (`dice+CE`): Balanced for general segmentation
- **Dice + Focal** (`dice+focal`): Better for class imbalance
- **Tversky + Cross-Entropy** (`tversky+CE`): Customizable false positive/negative trade-off
- **Tversky + Focal** (`tversky+focal`): Combines Tversky and focal strengths

### Loss Parameters
- **alpha**: Weight for false positives (typically 0.2-0.3 for small objects)
- **beta**: Weight for false negatives (typically 0.7-0.8 for small objects)
- **gamma**: Focus strength for focal loss (typically 1.33-2.0)

## Experiment Tracking

This project integrates with **Weights & Biases** (W&B) for experiment tracking:

1. **Initialize W&B** (one-time setup):
```bash
wandb login
```

2. **Automatic Logging**: All training metrics, configurations, and checkpoints are logged to W&B
3. **Monitor**: View real-time training progress at https://wandb.ai/

## Output

Training outputs are saved in:
```
./checkpoints/<dataset_name>/
```

Each checkpoint includes:
- Model weights
- Training configuration
- Mask values (for reconstruction)
- Epoch and step information

## GPU Memory Management

If you encounter out-of-memory errors:

1. **Reduce batch size**: `--batch-size 1`
2. **Enable gradient checkpointing**: Automatically activated on OOM errors
3. **Reduce image scale**: `--scale 0.25`

## Citation

If you use this code in your research, please cite:

```bibtex
@software{hyperbolic_unet_2025,
  author = {Mishra, Swasti Shreya},
  title = {Hyperbolic U-Net for Medical Image Segmentation},
  year = {2025},
  url = {https://github.com/yourusername/Hyperbolic-U-Net}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)
- Hyperbolic Learning Library: https://github.com/maxvanspengler/hyperbolic_learning_library

---

For questions or issues, please open a GitHub issue or contact the maintainers.
