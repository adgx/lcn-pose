# Machine Learning for Vision and Multimedia

## 3D Human Pose Estimation

A comprehensive implementation for training, evaluating, and visualizing 3D human pose estimation models using deep learning techniques.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Visualization](#visualization)
- [Repository Structure](#repository-structure)
- [Models](#models)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

This repository contains all the code required to train, evaluate, and visualize 3D human pose estimation models. The implementation is based on state-of-the-art deep learning approaches and supports multiple datasets and evaluation protocols. The visualization is dedicated only to humansc3d

### Key Features

- Support for multiple datasets (H36M, HumanSC3D, MPII)
- Flexible network architectures with configurable layers
- Multiple evaluation protocols (Protocol #1 and Protocol #2 with Procrustes alignment)
- Data augmentation techniques (flip, rotation, translation)
- Interactive visualization tools
- Comprehensive training and evaluation scripts

### Project Details

- **Course**: Machine Learning for Vision and Multimedia
- **Date**: July 17, 2025

---

## Quick Start

Already have the checkpoints? Clone → install → run the example commands below.

```bash
# Clone and enter repository
git clone <repo-url> && cd lcn-pose

# Create a fresh environment (recommended)
python3.8.20 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)" 
```

### Checkpoint Structure

When working with pre-trained models, organize your checkpoints as follows:

```ini
experiment/
│
├── test1/
│   └── checkpoints/
│       ├── best/
│       │   └── checkpoint
│       └── final/
│           └── checkpoint
└── test2/
    └── checkpoints/
        ├── best/
        │   └── checkpoint
        └── final/
            └── checkpoint
```

**Important**: When moving checkpoints from Kaggle to your local machine, you must update the absolute paths stored inside each checkpoint file located in `best/` and `final/` directories.

---

## Installation

### Requirements

tensorflow==2.13.0
numpy==1.24.3
opencv-python==4.11.0.86
scipy==1.10.1
pandas==2.0.3
matplotlib==3.7.5
pyrender==0.1.45
trimesh==4.6.5
PyOpenGL==3.1.0
prettytable==3.11.0
PyYAML==6.0.2

### Core Dependencies

```bash
# Install core requirements
pip install -r requirements.txt
```

### Optional Dependencies

For running Jupyter notebooks:

```bash
pip install notebook
```

### Environment Setup

We recommend using a virtual environment:

---

## Training

Train models using the `train.py` script with various configuration options.

### Basic Training Command

```bash
python train.py \
  --train_set h36m \
  --test_set h36m \
  --test-indices 3 \
  --mask-type locally_connected \
  --knn 3 \
  --layers 3 \
  --in-F 2
```

### Common Training Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--flip-data` | Enable horizontal flip augmentation | False |
| `--rotation-data` | Enable rotation augmentation | False |
| `--translate_data` | Enable translation augmentation | False |
| `--epochs` | Number of training epochs | 200 |
| `--resume_from` | Resume training from existing checkpoint | None |
| `--output_file` | Training log output file | None |

### Advanced Configuration

For hyperparameter tuning and ablation studies:

- `--mask-type`: Choose from `locally_connected`, `fully_connected`, or `dilated`
- `--init-type`: Initialization strategy - `same`, `ones`, or `random` (default)
- `--graph`: Graph topology index (integer)

### Example Training Commands

```bash
# Training with data augmentation
python train.py \
  --train_set h36m \
  --test_set h36m \
  --test-indices 3 \
  --mask-type locally_connected \
  --knn 3 \
  --layers 3 \
  --in-F 2 \
  --flip-data \
  --rotation-data \
  --translate_data \
  --epochs 300

# Resume training from checkpoint
python train.py \
  --train_set h36m \
  --test_set h36m \
  --resume_from experiment/test1/checkpoints/best/checkpoint
```

---

## Evaluation

Evaluate trained models using standard metrics and protocols.

### Protocol #1 (Standard Evaluation)

```bash
python evaluate.py \
  --filename h36m \
  --test-indices 3 \
  --per-joint              # optional: joint-wise results
```

### Protocol #2 (Procrustes-Aligned Evaluation)

```bash
python evaluate.py \
  --filename h36m \
  --test-indices 3 \
  --protocol2
```

### Evaluation Metrics

The evaluation script computes:

- Mean Per Joint Position Error (MPJPE)
- Procrustes-aligned MPJPE (when using `--protocol2`)
- Per-joint error analysis (when using `--per-joint`)

---

## Inference

Generate predictions on new data and save results to disk.

### Basic Inference

```bash
python inference.py \
  --train_set h36m \
  --test_set h36m \
  --test-indices 3 \
  --mask-type locally_connected \
  --knn 3 \
  --layers 3 \
  --in-F 2 \
  --checkpoints best            # "best" or "final"
```

### Test-Time Augmentation

Enable data augmentation during inference for improved results:

```bash
python inference.py \
  --train_set h36m \
  --test_set h36m \
  --test-indices 3 \
  --mask-type locally_connected \
  --knn 3 \
  --layers 3 \
  --in-F 2 \
  --checkpoints best \
  --augmentation f r t          # f=flip, r=rotation, t=translate
```

---

## Visualization

Interactive tools for qualitative analysis of model predictions.

### Jupyter Notebook

Launch the interactive visualization notebook:

```bash
jupyter notebook visualise_human_prediction.ipynb
```

The notebook provides:

- 3D pose visualization
- Comparison between ground truth and predictions
- Error analysis per joint
- Interactive 3D plotting capabilities

---

## Repository Structure

```ini
ml-pose-estimation/
├── analysis/                           # Core network components
│   ├── data/
│   └────  training/ 
│   └────  validation/
│   ├── main.py
├── datasets/                       # Dataset loaders and preprocessors
│   ├── human3.6/                       # Human3.6M dataset
│   ├── humansc3d/                  # HumanSC3D dataset
│   └── mpii/                       # MPII dataset
│   └── h36m_test.pkl
│   └── h36m_train.pkl
│   └── humansc3d_test.pkl
│   └── humansc3d_train.pkl
│   └── mpii_test.pkl
│   └── mpii_train.pkl
├── network/                        
│   └── models_att.py               # Model architecture definitions
├── visualization/
│   ├── util/      
│   └──── .....
│   ├── notebook/                   # Jupyter notebooks
│   └──── visualise_human_prediction.ipynb
├── experiments/                    # Experiment outputs
│   └─- .....
├── tools/  
│   └─- .....
├── train.py                        # Training script
├── random_search.py 
├── evaluate.py                     # Evaluation script
├── inference.py                    # Inference script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Models

### Pre-trained Models

The best performing models are available for download. These models have been trained on standard datasets and achieve state-of-the-art performance.

**Download Link**: <a href="https://drive.google.com/drive/folders/1uXE9XgH0LC0C9EASp3MJXs6atdANXJM7?usp=sharing"> Link </a>

### Experiment Tracking

All experimental results and hyperparameter configurations are documented and can be accessed through the provided links above.

**Excel Link**: <a href="https://drive.google.com/drive/folders/1uXE9XgH0LC0C9EASp3MJXs6atdANXJM7?usp=sharing"> Link </a>

---

## Contact

For questions, issues, or contributions, please:

1. Open an issue on GitHub

---

## Acknowledgements

This repository is based on the original TensorFlow implementation released by the authors of the ICCV 2019 paper <a href="https://github.com/CHUNYUWANG/lcn-pose"> Link</a> <br>
We thank them for making their research publicly available and contributing to the open-source community.

### References

- Original paper: [Paper Title] (ICCV 2019)
- Dataset providers: Human3.6M, HumanSC3D, MPII
- TensorFlow community for framework support

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ml-pose-estimation-2025,
  title={3D Human Pose Estimation},
  author={[Andrea Ongaro and Antonino di Gregorio]},
  year={2025},
  howpublished={\url{https://github.com/adgx/lcn-pose}}
}
```