#!/usr/bin/env python3
"""
Final Project: Comparative Analysis of Deep Generative Models
Complete execution script for training and evaluation

This script orchestrates the entire project workflow:
1. Data preparation
2. Model training (DC-GAN, VAE, WGAN-GP)
3. Hyperparameter optimization
4. Comprehensive evaluation
5. Report generation support

Author: [Your Name]
Date: June 2025
"""

import os
import sys
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('project_log.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import our modules (assuming they're in the same directory)
from generative_models import *
from evaluation_tools import *

class ProjectRunner:
    """Main class to orchestrate the entire project"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.start_time = time.time()
        
        # Create output directories
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], 'models'), exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], 'figures'), exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], 'data'), exist_ok=True)
        
        logger.info(f"Project initialized. Using device: {self.device}")
        logger.info(f"Output directory: {config['output_dir']}")
    
    def prepare_data(self):
        """Prepare CIFAR-10 dataset"""
        logger.info("Preparing CIFAR-10 dataset...")
        
        self.full_dataloader = get_cifar10_loader(
            batch_size=self.config['batch_size'],
            subset_size=None
        )
        
        # Create subset for faster experimentation if needed
        if self.config['use_subset']:
            self.train_dataloader = get_cifar10_loader(
                batch_size=self.config['batch_size'],
                subset_size=self.config['subset_size']
            )
            logger.info(f"Using subset of {self.config['subset_size']} images")
        else:
            self.train_dataloader = self.full_dataloader
            logger.info("Using full CIFAR-10 dataset")
    
    def train_models(self):
        """Train all three generative models"""
        logger.info("Starting model training phase...")
        
        models_to_train = self.config['models_to_train']
        
        # Initialize models
        if 'dcgan' in models_to_train:
            logger.info("Initializing DC-GAN...")
            self.gen_dcgan = DCGANGenerator(self.config['latent_dim']).to(self.device)
            self.disc_dcgan = DCGANDiscriminator().to(self.device)
        
        if 'wgangp' in models_to_train:
            logger.info("Initializing WGAN-GP...")
            self.gen_wgangp = WGANGPGenerator(self.config['latent_dim']).to(self.device)
            self.critic_wgangp = WGANGPCritic().to(self.device)
        
        if 'vae' in models_to_train:
            logger.info("Initializing VAE...")
            self.vae = VAE(self.config['vae_latent_dim']).to(self.device)
        
        # Training
        if 'dcgan' in models_to_train:
            logger.info("Training DC-GAN...")
            start_time = time.time()
            dcgan_gen_losses, dcgan_disc_losses = train_dcgan(
                self.gen_dcgan, self.disc_dcgan, 
                self.train_dataloader,
                num_epochs=self.config['num_epochs'],
                lr=self.config['learning_rate']
            )
            train_time = time.time() - start_time
            
            self.results['dcgan'] = {
                'losses': (dcgan_gen_losses, dcgan_disc_losses),
                'train_time': train_time
            }
            
            # Save model
            torch.save(self.gen_dcgan.state_dict(), 
                      os.path.join(self.config['output_dir'], 'models', 'dcgan_generator.pth'))
            torch.save(self.disc_dcgan.state_dict(),
                      os.path.join(self.config['output_dir'], 'models', 'dcgan_discriminator.pth'))
            
            logger.info(f"DC-GAN training completed in {train_time:.2f} seconds")
        
        if 'wgangp' in models_to_train:
            logger.info("Training WGAN-GP...")
            start_time = time.time()
            wgangp_gen_losses, wgangp_critic_losses = train_wgangp(
                self.gen_wgangp, self.critic_wgangp,
                self.train_dataloader,
                num_epochs=self.config['num_epochs'],
                lr=self.config['learning_rate']
            )
            train_time = time.time() - start_time
            
            self.results['wgangp'] = {
                'losses': (wgangp_gen_losses, wgangp_critic_losses),
                'train_time': train_time
            }
            
            # Save model
            torch.save(self.gen_wgangp.state_dict(),
                      os.path.join(self.config['output_dir'], 'models', 'wgangp_generator.pth'))
            torch.save(self.critic_wgangp.state_dict(),
                      os.path.join(self.config['output_dir'], 'models', 'wgangp_critic.pth'))
            
            logger.info(f"WGAN-GP training completed in {train_time:.2f} seconds")
        
        if 'vae' in models_to_train:
            logger.info("Training VAE...")
            start_time = time.time()
            vae_losses = train_vae(
                self.vae, self.train_dataloader,
                num_epochs=self.config['num_epochs'],
                lr=self.config['vae_learning_rate']
            )
            train_time = time.time() - start_time
            
            self.results['vae'] = {
                'losses': vae_losses,
                'train_time': train_time
            }
            
            # Save model
            torch.save(self.vae.state_dict(),
                      os.path.join(self.config['output_dir'], 'models', 'vae_model.pth'))
            
            logger.info(f"VAE training completed in {train_time:.2f} seconds")
    
    def hyperparameter_optimization(self):
        """Run hyperparameter optimization"""
        if not self.config['run_hyperparameter_search']:
            logger.info("Skipping hyperparameter optimization")
            return
        
        logger.info("Starting hyperparameter optimization...")
        
        analyzer = HyperparameterAnalyzer()
        
        param_grid = {
            'lr': [0.0001, 0.0002, 0.0005],
            'batch_size': [32, 64, 128],
            'latent_dim': [64, 100, 128]
        }
        
        # Run hyperparameter search for each model (simplified for time constraints)
        hp_results = {}
        
        for model_name in self.config['models_to_train']:
            logger.info(f"Hyperparameter search for {model_name}...")
            
            if model_name == 'vae':
                model_class = VAE
            elif model_name == 'dcgan':
                model_class = DCGANGenerator
            elif model_name == 'wgangp':
                model_class = WGANGPGenerator
            else:
                continue
            
            best_params, best_score, results = analyzer.run_hyperparameter_search(
                model_class, self.train_dataloader, self.device, param_grid, num_epochs=5
            )
            
            hp_results[model_name] = {
                'best_params': best_params,
                'best_score': best_score,
                'all_results': results
            }
            
            logger.info(f"Best parameters for {model_name}: {best_params}")
        
        # Save hyperparameter results
        with open(os.path.join(self.config['output_dir'], 'data', 'hyperparameter_results.json'), 'w') as f:
            json.dump(hp_results, f, indent=2)
        
        self.results['hyperparameters'] = hp_results
    
    def comprehensive_evaluation(self):
        """Run comprehensive evaluation of all trained models"""
        logger.info("Starting comprehensive evaluation...")
        
        # Collect trained models
        models = []
        model_names = []
        
        if hasattr(self, 'gen_dcgan'):
            models.append(self.gen_dcgan)
            model_names.append('DC-GAN')
        
        if hasattr(self, 'gen_wgangp'):
            models.append(self.gen_wgangp)
            model_names.append('WGAN-GP')
        
        if hasattr(self, 'vae'):
            models.append(self.vae)
            model_names.append('VAE')
        
        if not models:
            logger.warning("No trained models found for evaluation")
            return
        
        # Run evaluation
        evaluator = ModelEvaluator(self.device)
        visualizer = ResultVisualizer()
        
        # Get real images for comparison
        real_images = []
        for batch_idx, (data, _) in enumerate(self.full_dataloader):
            real_images.append(data)
            if batch_idx >= 10:  # Limit for speed
                break
        real_images = torch.cat(real_images, dim=0)
        
        # Evaluate each model
        evaluation_results = {}
        
        for model, name in zip(models, model_names):
            logger.info(f"Evaluating {name}...")
            
            # Generate samples
            model.eval()
            generated_images = []
            
            with torch.no_grad():
                for i in range(10):
                    if name == 'VAE':
                        z = torch.randn(32, self.config['vae_latent_dim'], device=self.device)
                        samples = model.decoder(z)
                    else:
                        z = torch.randn(32, self.config['latent_dim'], 1, 1, device=self.device)
                        samples = model(z)
                    generated_images.append(samples.cpu())
            
            generated_images = torch.cat(generated_images, dim=0)
            
            # Calculate metrics
            is_mean, is_std = evaluator.calculate_inception_score(generated_images)
            fid_score = evaluator.calculate_fid(real_images[:len(generated_images)], generated_images)
            div_mean, div_std = evaluator.evaluate_diversity(generated_images)
            
            evaluation_results[name] = {
                'IS': (is_mean, is_std),
                'FID': fid_score,
                'Diversity': (div_mean, div_std)
            }
            
            logger.info(f"{name} - IS: {is_mean:.3f}±{is_std:.3f}, FID: {fid_score:.3f}, Div: {div_mean:.3f}±{div_std:.3f}")
        
        # Create visualizations
        visualizer.plot_evaluation_metrics(
            evaluation_results,
            save_path=os.path.join(self.config['output_dir'], 'figures', 'evaluation_metrics.png')
        )
        
        visualizer.plot_sample_comparison(models, self.device)
        plt.savefig(os.path.join(self.config['output_dir'], 'figures', 'sample_comparison.png'))
        
        # Plot training curves if available
        training_results = {}
        if 'dcgan' in self.results:
            training_results['dcgan_losses'] = self.results['dcgan']['losses']
        if 'wgangp' in self.results:
            training_results['wgangp_losses'] = self.results['wgangp']['losses']
        if 'vae' in self.results:
            training_results['vae_losses'] = self.results['vae']['losses']
        
        if training_results:
            visualizer.plot_training_curves(
                training_results,
                save_path=os.path.join(self.config['output_dir'], 'figures', 'training_curves.png')
            )
        
        # Save evaluation results
        import pandas as pd
        results_df = pd.DataFrame({
            'Model': list(evaluation_results.keys()),
            'IS_Mean': [evaluation_results[m]['IS'][0] for m in evaluation_results.keys()],
            'IS_Std': [evaluation_results[m]['IS'][1] for m in evaluation_results.keys()],
            'FID': [evaluation_results[m]['FID'] for m in evaluation_results.keys()],
            'Diversity_Mean': [evaluation_results[m]['Diversity'][0] for m in evaluation_results.keys()],
            'Diversity_Std': [evaluation_results[m]['Diversity'][1] for m in evaluation_results.keys()],
        })
        
        results_df.to_csv(os.path.join(self.config['output_dir'], 'data', 'evaluation_results.csv'), index=False)
        
        self.results['evaluation'] = evaluation_results
    
    def generate_report_data(self):
        """Generate data for the final report"""
        logger.info("Generating report data...")
        
        # Compile all results
        report_data = {
            'experiment_info': {
                'date': datetime.now().isoformat(),
                'device': str(self.device),
                'total_runtime': time.time() - self.start_time,
                'config': self.config
            },
            'training_results': {},
            'evaluation_results': self.results.get('evaluation', {}),
            'hyperparameter_results': self.results.get('hyperparameters', {})
        }
        
        # Add training information
        for model_name in self.config['models_to_train']:
            if model_name in self.results:
                report_data['training_results'][model_name] = {
                    'train_time': self.results[model_name]['train_time'],
                    'final_loss': self.results[model_name]['losses'][-1] if isinstance(self.results[model_name]['losses'], list) else self.results[model_name]['losses'][0][-1]
                }
        
        # Save comprehensive results
        with open(os.path.join(self.config['output_dir'], 'data', 'complete_results.json'), 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Report data saved to {self.config['output_dir']}/data/complete_results.json")
        
        # Print summary for report
        self.print_summary()
    
    def print_summary(self):
        """Print project summary for inclusion in report"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("PROJECT SUMMARY")
        print("="*60)
        print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"Models trained: {', '.join(self.config['models_to_train'])}")
        print(f"Dataset: CIFAR-10 ({'subset' if self.config['use_subset'] else 'full'})")
        print(f"Device used: {self.device}")
        print(f"Output directory: {self.config['output_dir']}")
        
        if 'evaluation' in self.results:
            print("\nQUANTITATIVE RESULTS:")
            for model, metrics in self.results['evaluation'].items():
                print(f"{model}:")
                print(f"  Inception Score: {metrics['IS'][0]:.3f} ± {metrics['IS'][1]:.3f}")
                print(f"  FID Score: {metrics['FID']:.3f}")
                print(f"  Diversity: {metrics['Diversity'][0]:.3f} ± {metrics['Diversity'][1]:.3f}")
        
        print("\nFILES GENERATED:")
        print("- Models saved in: models/")
        print("- Figures saved in: figures/")
        print("- Data files saved in: data/")
        print("="*60)
    
    def run_complete_pipeline(self):
        """Execute the complete project pipeline"""
        logger.info("Starting complete project pipeline...")
        
        try:
            # Step 1: Data preparation
            self.prepare_data()
            
            # Step 2: Model training
            self.train_models()
            
            # Step 3: Hyperparameter optimization (optional)
            if self.config['run_hyperparameter_search']:
                self.hyperparameter_optimization()
            
            # Step 4: Comprehensive evaluation
            self.comprehensive_evaluation()
            
            # Step 5: Generate report data
            self.generate_report_data()
            
            logger.info("Project pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            raise

# Default configuration
DEFAULT_CONFIG = {
    'output_dir': 'project_results',
    'models_to_train': ['dcgan', 'wgangp', 'vae'],
    'batch_size': 64,
    'num_epochs': 30,
    'learning_rate': 0.0002,
    'vae_learning_rate': 0.001,
    'latent_dim': 100,
    'vae_latent_dim': 128,
    'use_subset': True,  # Set to False for full dataset
    'subset_size': 10000,
    'run_hyperparameter_search': False  # Set to True for hyperparameter optimization
}

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Deep Generative Models Final Project')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--output_dir', type=str, default='project_results', help='Output directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--subset', action='store_true', help='Use dataset subset for faster training')
    parser.add_argument('--hyperparameter_search', action='store_true', help='Run hyperparameter optimization')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG.copy()
    
    # Override with command line arguments
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.subset:
        config['use_subset'] = True
    if args.hyperparameter_search:
        config['run_hyperparameter_search'] = True
    
    # Print configuration
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Run project
    runner = ProjectRunner(config)
    runner.run_complete_pipeline()

if __name__ == "__main__":
    main()