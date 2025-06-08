import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torchvision.models import inception_v3
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ========================
# 1. DATA PREPARATION
# ========================

def get_cifar10_loader(batch_size=64, subset_size=None):
    """Load CIFAR-10 dataset with preprocessing"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root=os.path.join(os.getcwd(), 'data'), train=True, download=True, transform=transform
    )
    
    # Use subset if specified
    if subset_size:
        dataset = torch.utils.data.Subset(dataset, range(subset_size))
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader

# ========================
# 2. DC-GAN IMPLEMENTATION
# ========================

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, channels=3, features_g=64):
        super(DCGANGenerator, self).__init__()
        
        # Input: latent_dim x 1 x 1
        self.net = nn.Sequential(
            # First layer: 100 -> 4x4x512
            nn.ConvTranspose2d(latent_dim, features_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            
            # 4x4x512 -> 8x8x256
            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            
            # 8x8x256 -> 16x16x128
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            
            # 16x16x128 -> 32x32x3
            nn.ConvTranspose2d(features_g * 2, channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, x):
        return self.net(x)

class DCGANDiscriminator(nn.Module):
    def __init__(self, channels=3, features_d=64):
        super(DCGANDiscriminator, self).__init__()
        
        self.net = nn.Sequential(
            # 32x32x3 -> 16x16x64
            nn.Conv2d(channels, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16x64 -> 8x8x128
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8x128 -> 4x4x256
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4x256 -> 1x1x1
            nn.Conv2d(features_d * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).view(-1, 1).squeeze(1)

# ========================
# 3. VAE IMPLEMENTATION
# ========================

class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=128, channels=3):
        super(VAEEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # 32x32x3 -> 16x16x32
            nn.Conv2d(channels, 32, 4, 2, 1),
            nn.ReLU(),
            
            # 16x16x32 -> 8x8x64
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            
            # 8x8x64 -> 4x4x128
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            
            # 4x4x128 -> 2x2x256
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=128, channels=3):
        super(VAEDecoder, self).__init__()
        
        self.fc = nn.Linear(latent_dim, 256 * 2 * 2)
        
        self.deconv_layers = nn.Sequential(
            # 2x2x256 -> 4x4x128
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            
            # 4x4x128 -> 8x8x64
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            
            # 8x8x64 -> 16x16x32
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            
            # 16x16x32 -> 32x32x3
            nn.ConvTranspose2d(32, channels, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 2, 2)
        x = self.deconv_layers(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=128, channels=3):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(latent_dim, channels)
        self.decoder = VAEDecoder(latent_dim, channels)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
    
    def generate(self, num_samples, device):
        with torch.no_grad():
            z = torch.randn(num_samples, 128, device=device)
            return self.decoder(z)

# ========================
# 4. WGAN-GP IMPLEMENTATION
# ========================

class WGANGPGenerator(nn.Module):
    def __init__(self, latent_dim=100, channels=3, features_g=64):
        super(WGANGPGenerator, self).__init__()
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, features_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(features_g * 2, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x)

class WGANGPCritic(nn.Module):
    def __init__(self, channels=3, features_d=64):
        super(WGANGPCritic, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(channels, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features_d * 4, 1, 4, 1, 0, bias=False)
        )
    
    def forward(self, x):
        return self.net(x).view(-1, 1).squeeze(1)

def gradient_penalty(critic, real, fake, device):
    """Calculate gradient penalty for WGAN-GP"""
    batch_size, c, h, w = real.shape
    epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, c, h, w).to(device)
    
    interpolated_images = real * epsilon + fake * (1 - epsilon)
    interpolated_images.requires_grad_(True)
    
    # Calculate critic scores
    mixed_scores = critic(interpolated_images)
    
    # Calculate gradients
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Calculate penalty
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    
    return penalty

# ========================
# 5. TRAINING FUNCTIONS
# ========================

def train_dcgan(generator, discriminator, dataloader, num_epochs=50, lr=0.0002):
    """Train DC-GAN"""
    criterion = nn.BCELoss()
    opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)
    gen_losses, disc_losses = [], []
    
    generator.train()
    discriminator.train()
    
    for epoch in range(num_epochs):
        epoch_gen_loss, epoch_disc_loss = 0, 0
        
        for batch_idx, (real, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            real = real.to(device)
            batch_size = real.size(0)
            
            # Train Discriminator
            opt_disc.zero_grad()
            
            # Real images
            label_real = torch.ones(batch_size, device=device)
            output_real = discriminator(real)
            loss_disc_real = criterion(output_real, label_real)
            
            # Fake images
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake = generator(noise)
            label_fake = torch.zeros(batch_size, device=device)
            output_fake = discriminator(fake.detach())
            loss_disc_fake = criterion(output_fake, label_fake)
            
            loss_disc = loss_disc_real + loss_disc_fake
            loss_disc.backward()
            opt_disc.step()
            
            # Train Generator
            opt_gen.zero_grad()
            output = discriminator(fake)
            loss_gen = criterion(output, label_real)
            loss_gen.backward()
            opt_gen.step()
            
            epoch_gen_loss += loss_gen.item()
            epoch_disc_loss += loss_disc.item()
        
        avg_gen_loss = epoch_gen_loss / len(dataloader)
        avg_disc_loss = epoch_disc_loss / len(dataloader)
        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)
        
        print(f"Epoch {epoch+1}: Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")
    
    return gen_losses, disc_losses

def train_vae(model, dataloader, num_epochs=50, lr=0.001):
    """Train VAE"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            
            # VAE loss = Reconstruction loss + KL divergence
            recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader.dataset)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss: {avg_loss:.4f}")
    
    return losses

def train_wgangp(generator, critic, dataloader, num_epochs=50, lr=0.0002, lambda_gp=10):
    """Train WGAN-GP"""
    opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))
    
    gen_losses, critic_losses = [], []
    
    generator.train()
    critic.train()
    
    for epoch in range(num_epochs):
        epoch_gen_loss, epoch_critic_loss = 0, 0
        
        for batch_idx, (real, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            real = real.to(device)
            batch_size = real.size(0)
            
            # Train Critic
            for _ in range(5):  # Train critic 5 times per generator step
                opt_critic.zero_grad()
                
                noise = torch.randn(batch_size, 100, 1, 1, device=device)
                fake = generator(noise)
                
                critic_real = critic(real)
                critic_fake = critic(fake.detach())
                
                gp = gradient_penalty(critic, real, fake.detach(), device)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp
                
                loss_critic.backward()
                opt_critic.step()
            
            # Train Generator
            opt_gen.zero_grad()
            fake = generator(noise)
            gen_fake = critic(fake)
            loss_gen = -torch.mean(gen_fake)
            loss_gen.backward()
            opt_gen.step()
            
            epoch_gen_loss += loss_gen.item()
            epoch_critic_loss += loss_critic.item()
        
        avg_gen_loss = epoch_gen_loss / len(dataloader)
        avg_critic_loss = epoch_critic_loss / len(dataloader)
        gen_losses.append(avg_gen_loss)
        critic_losses.append(avg_critic_loss)
        
        print(f"Epoch {epoch+1}: Gen Loss: {avg_gen_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}")
    
    return gen_losses, critic_losses

# ========================
# 6. EVALUATION METRICS
# ========================

def calculate_inception_score(images, batch_size=32, splits=10):
    """Calculate Inception Score"""
    # Load pretrained InceptionV3
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    
    # Preprocess images
    resize = transforms.Resize((299, 299))
    
    N = len(images)
    scores = []
    
    for i in range(splits):
        part = images[i * N // splits: (i + 1) * N // splits]
        
        # Resize and get predictions
        resized_images = torch.stack([resize(img) for img in part])
        
        with torch.no_grad():
            preds = inception_model(resized_images.to(device))
            preds = F.softmax(preds, dim=1)
        
        # Calculate IS
        py = torch.mean(preds, dim=0)
        kl_div = preds * (torch.log(preds) - torch.log(py))
        kl_div = torch.mean(torch.sum(kl_div, dim=1))
        scores.append(torch.exp(kl_div).cpu().item())
    
    return np.mean(scores), np.std(scores)

def visualize_results(generator_dcgan, generator_wgangp, vae_model, num_samples=64):
    """Generate and visualize samples from all models"""
    
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    
    # DC-GAN samples
    with torch.no_grad():
        noise = torch.randn(num_samples, 100, 1, 1, device=device)
        dcgan_samples = generator_dcgan(noise)
        dcgan_samples = (dcgan_samples + 1) / 2  # Denormalize
        
        for i in range(8):
            axes[0, i].imshow(dcgan_samples[i].cpu().permute(1, 2, 0))
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('DC-GAN', rotation=90, fontsize=12)
    
    # WGAN-GP samples
    with torch.no_grad():
        noise = torch.randn(num_samples, 100, 1, 1, device=device)
        wgangp_samples = generator_wgangp(noise)
        wgangp_samples = (wgangp_samples + 1) / 2
        
        for i in range(8):
            axes[1, i].imshow(wgangp_samples[i].cpu().permute(1, 2, 0))
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel('WGAN-GP', rotation=90, fontsize=12)
    
    # VAE samples
    with torch.no_grad():
        vae_samples = vae_model.generate(num_samples, device)
        vae_samples = (vae_samples + 1) / 2
        
        for i in range(8):
            axes[2, i].imshow(vae_samples[i].cpu().permute(1, 2, 0))
            axes[2, i].axis('off')
            if i == 0:
                axes[2, i].set_ylabel('VAE', rotation=90, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('generated_samples_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# ========================
# 7. MAIN EXECUTION
# ========================

def main():
    """Main training and evaluation pipeline"""
    
    # Configuration
    batch_size = 64
    num_epochs = 30  # Reduced for faster training
    lr = 0.0002
    latent_dim = 100
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    dataloader = get_cifar10_loader(batch_size=batch_size, subset_size=10000)  # Use subset for faster training
    
    # Initialize models
    print("Initializing models...")
    
    # DC-GAN
    gen_dcgan = DCGANGenerator(latent_dim).to(device)
    disc_dcgan = DCGANDiscriminator().to(device)
    
    # WGAN-GP
    gen_wgangp = WGANGPGenerator(latent_dim).to(device)
    critic_wgangp = WGANGPCritic().to(device)
    
    # VAE
    vae = VAE(latent_dim=128).to(device)
    
    # Train models
    print("\n" + "="*50)
    print("Training DC-GAN...")
    dcgan_gen_losses, dcgan_disc_losses = train_dcgan(
        gen_dcgan, disc_dcgan, dataloader, num_epochs, lr
    )
    
    print("\n" + "="*50)
    print("Training WGAN-GP...")
    wgangp_gen_losses, wgangp_critic_losses = train_wgangp(
        gen_wgangp, critic_wgangp, dataloader, num_epochs, lr
    )
    
    print("\n" + "="*50)
    print("Training VAE...")
    vae_losses = train_vae(vae, dataloader, num_epochs, lr=0.001)
    
    # Generate and visualize results
    print("\n" + "="*50)
    print("Generating comparison visualizations...")
    visualize_results(gen_dcgan, gen_wgangp, vae)
    
    # Save models
    torch.save(gen_dcgan.state_dict(), 'dcgan_generator.pth')
    torch.save(gen_wgangp.state_dict(), 'wgangp_generator.pth')
    torch.save(vae.state_dict(), 'vae_model.pth')
    
    print("Training completed! Models saved.")
    
    return {
        'dcgan_losses': (dcgan_gen_losses, dcgan_disc_losses),
        'wgangp_losses': (wgangp_gen_losses, wgangp_critic_losses),
        'vae_losses': vae_losses,
        'models': (gen_dcgan, gen_wgangp, vae)
    }

if __name__ == "__main__":
    results = main()