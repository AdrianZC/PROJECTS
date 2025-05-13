# CNN Image Classification with CIFAR-10

This repository contains a comprehensive implementation and evaluation of Convolutional Neural Networks (CNNs) for image classification on the CIFAR-10 dataset using PyTorch.

## Project Overview

This project investigates the performance of different CNN architectures (SimpleCNN and ResNet18) and their hyperparameter configurations for image classification. It provides a systematic approach to:
- Compare different network architectures
- Evaluate various optimization techniques
- Analyze the effects of hyperparameter tuning
- Visualize training and validation metrics

## Key Findings

- ResNet18 with Adam optimizer and learning rate of 0.01 achieved the highest accuracy (81.28%)
- SimpleCNN demonstrated competitive performance with appropriate hyperparameter settings
- Learning rate emerged as the most influential hyperparameter
- Batch size had minimal impact on final model performance
- Standard data augmentation consistently outperformed advanced techniques

## Repository Structure

- `C181FP.py`: Main Python script containing model definitions, training loops, and evaluation code
- `experiment_results/`: Directory containing experimental outputs
  - `all_results.pkl`: Pickle file with all experiment results
  - `experiment_summary.csv`: CSV summary of all experiment configurations and results
  - Various plot files showing performance comparisons

## Model Architectures

### SimpleCNN
A custom CNN designed to balance complexity and performance:
- 3 convolutional layers (3→32→64→128 channels)
- Max pooling and ReLU activations
- Configurable dropout rate
- 2 fully connected layers (8192→256→10)

### ResNet18
A pre-implemented ResNet variant with 18 layers from torchvision, modified for CIFAR-10.

## Features

- **Regularization Techniques**:
  - Label smoothing
  - Dropout
  - Weight decay
  - Data augmentation

- **Optimization Methods**:
  - Adam optimizer
  - SGD with momentum
  - Cosine annealing learning rate scheduling

- **Data Augmentation**:
  - Standard techniques (random cropping, horizontal flipping)
  - Advanced techniques (RandAugment)

- **Comprehensive Visualization**:
  - Learning curves
  - Performance comparisons by architecture, optimizer, and hyperparameters
  - Batch size analysis
  - Augmentation effectiveness comparison

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- pandas
- numpy

## Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cnn-cifar10-classification.git
cd cnn-cifar10-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the experiments:
```bash
python C181FP.py
```

For GPU acceleration with larger batch sizes:
```bash
python C181FP.py --gpu_batch_multiplier 2
```

## Experiment Configuration

The project systematically explores different hyperparameter configurations including:
- Architectures: SimpleCNN, ResNet18
- Optimizers: Adam, SGD
- Learning rates: 0.001, 0.01
- Batch sizes: 64, 128, 256
- Dropout rates: 0.3, 0.5
- Label smoothing: 0.0, 0.1
- Learning rate scheduling: with/without cosine annealing
- Data augmentation: standard vs. advanced techniques

## Results

The experimental results demonstrate that:
1. Architecture Choice: ResNet18 achieved highest accuracy (81.28%), but SimpleCNN was competitive (80.90%) with proper tuning
2. Optimizer-Architecture Interaction: ResNet18 performed best with Adam, while SimpleCNN showed strong performance with SGD at higher learning rates
3. Learning Rate Significance: Optimal learning rates depended on architecture-optimizer combinations
4. Batch Size Flexibility: Minimal impact on accuracy across tested ranges
5. Augmentation Strategy: Simpler techniques often outperformed more complex ones

## Future Work

Potential areas for future exploration include:
- Longer training durations to observe longer-term trends
- Testing on more complex datasets (TinyImageNet, ImageNet)
- Additional architectures (EfficientNet, MobileNetV2)
- More advanced regularization techniques (MixUp, CutMix)

## License

[MIT License](LICENSE)

## Acknowledgements

- The CIFAR-10 dataset (Krizhevsky, Nair, and Hinton)
- PyTorch and torchvision libraries