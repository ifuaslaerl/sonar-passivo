# Sonar Passivo - Adversarial Robustness

A Deep Learning framework for classifying passive sonar signals using 1D Convolutional Neural Networks (CNN). This project investigates the robustness of CNNs against adversarial attacks (FGSM) and implements adversarial training strategies.

## Structure

```text
.
├── data/                   # Data storage (ignored by git)
├── scripts/                # Executable scripts
│   ├── train.py            # Main training script
│   ├── evaluate.py         # Evaluation script
│   └── generate_adv.py     # Adversarial sample generator
├── src/
│   └── sonar_passivo/      # Main package
│       ├── model.py        # SonarCNN Architecture
│       ├── dataset.py      # Data loading logic
│       ├── engine.py       # Training and Testing loops
│       └── adversarial.py  # FGSM attack logic
├── pyproject.toml          # Dependencies and config
└── README.md
```

## Installation

1. Clone the repository.
2. Install the package in editable mode:

```bash
pip install -e .
```

## Usage

### 1. Training
Train the model on the standard dataset.

```bash
python scripts/train.py --epochs 75 --batch_size 1 --mode standard
```

### 2. Generating Adversarial Examples
Generate adversarial samples using a trained model.

```bash
python scripts/generate_adv.py --model_path data/Networks/robust/best_model.pth --epsilon 0.01
```

### 3. Adversarial Training
Train the model using a mix of clean and adversarial data.

```bash
python scripts/train.py --mode adversarial --mix_ratio 0.2
```
