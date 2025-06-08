# Comparative Analysis of Deep Generative Models: DC-GAN, VAE, and WGAN-GP

This project investigates the performance characteristics and trade-offs of three major deep generative modeling paradigms on the CIFAR-10 dataset, focusing on Deep Convolutional Generative Adversarial Networks (DC-GAN), Variational Autoencoders (VAE), and Wasserstein GANs with Gradient Penalty (WGAN-GP).

## Project Overview

This study compares the performance of three different generative modeling approaches using a systematic experimental framework. The evaluation examines how different architectural choices and hyperparameter settings affect metrics such as:
- Sample quality (Inception Score)
- Distribution similarity (Fréchet Inception Distance)
- Sample diversity and mode coverage
- Training stability and convergence
- Computational efficiency

The implementation addresses the challenges of fair comparison between fundamentally different generative paradigms through consistent architectural design and comprehensive evaluation methodology.

## Repository Structure

- `generative_models.py` - Core implementations of all three generative models
- `evaluation_tools.py` - Evaluation metrics, visualization tools, and hyperparameter analysis
- `project_runner.py` - Main execution pipeline with configuration management
- `README.md` - Project documentation and usage guide

## Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision
- NumPy
- Matplotlib
- pandas
- scipy
- scikit-learn
- tqdm
- seaborn (optional)

Install dependencies with:
```bash
pip install torch torchvision matplotlib pandas numpy scipy scikit-learn tqdm seaborn
```

## Usage Guide

### Quick Test Run

Run all three models with a subset of data for quick testing:

```bash
python project_runner.py --subset --epochs 5
```

### Full Training

Run the complete experiment with all models:

```bash
python project_runner.py --epochs 30
```

### With Hyperparameter Optimization

Run comprehensive hyperparameter search:

```bash
python project_runner.py --epochs 20 --hyperparameter_search
```

### Custom Configuration

Specify custom output directory and training parameters:

```bash
python project_runner.py --epochs 50 --output_dir my_results
```

Options:
- `--epochs`: Number of training epochs (default: 30)
- `--subset`: Use dataset subset for faster training
- `--hyperparameter_search`: Run hyperparameter optimization
- `--output_dir`: Output directory path (default: 'project_results')

## File Descriptions

### Core Files

- `generative_models.py`: Contains implementations of:
  - `DCGANGenerator` & `DCGANDiscriminator`: DC-GAN architecture
  - `VAE` with `VAEEncoder` & `VAEDecoder`: Variational Autoencoder
  - `WGANGPGenerator` & `WGANGPCritic`: WGAN-GP architecture
  - Training functions for all models
  - Data loading utilities

### Supporting Files

- `evaluation_tools.py`: Implements functions for:
  - Inception Score and FID calculation
  - Sample diversity metrics
  - Training curve visualization
  - Hyperparameter analysis and plotting
- `project_runner.py`: Provides complete experimental pipeline with:
  - Automated training orchestration
  - Result logging and visualization
  - Comprehensive evaluation and comparison

## Model Architectures

### DC-GAN
- **Generator**: 4-layer transposed CNN (100D noise → 32×32×3 images)
- **Discriminator**: 4-layer CNN with batch normalization
- **Training**: Adversarial minimax loss with Adam optimizer

### VAE
- **Encoder**: CNN with reparameterization trick (128D latent space)
- **Decoder**: Transposed CNN reconstruction
- **Training**: ELBO loss (reconstruction + KL divergence)

### WGAN-GP
- **Generator**: Similar to DC-GAN architecture
- **Critic**: CNN without sigmoid activation
- **Training**: Wasserstein loss with gradient penalty

## Key Findings

The detailed results of the experiments can be found in the `project_results/` directory after running the provided scripts. Key aspects evaluated include:

- How different loss functions affect training stability
- Impact of latent space dimensionality on sample quality
- Trade-offs between sample sharpness and mode coverage
- Computational efficiency across different architectures
- Convergence patterns and training dynamics
- Performance on standard generative modeling benchmarks

## Output Structure

After running experiments, results are organized as:

```
project_results/
├── models/                    # Saved model weights
│   ├── dcgan_generator.pth
│   ├── wgangp_generator.pth
│   └── vae_model.pth
├── figures/                   # Generated visualizations
│   ├── training_curves.png
│   ├── sample_comparison.png
│   └── evaluation_metrics.png
├── data/                      # Quantitative results
│   ├── evaluation_results.csv
│   ├── complete_results.json
│   └── hyperparameter_results.json
└── project_log.txt           # Detailed training logs
```

## Academic Context

This project was developed as the final project for COGS 185 - Deep Learning, implementing a comprehensive comparative study that meets the following academic requirements:

- **Problem Significance**: Systematic comparison of major generative modeling paradigms
- **Dataset Challenge**: CIFAR-10 with comprehensive quantitative evaluation
- **Novel Contributions**: Unified comparison framework and stability analysis
- **Experimental Rigor**: Hyperparameter optimization and multiple evaluation metrics
- **Professional Documentation**: Research-quality implementation and reporting

## Acknowledgements

The implementation builds upon foundational work in generative modeling, particularly the original papers by Goodfellow et al. (GANs), Kingma & Welling (VAEs), and Gulrajani et al. (WGAN-GP).