import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

# Load a pre-trained ResNet as the encoder backbone
class ResNetEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ResNetEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)  # Use ResNet-50, or resnet18 for smaller model
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove fully connected layers
        self.flatten = nn.Flatten()  # Flatten the output feature map
        self.fc_mu = nn.Linear(2048 * 7 * 7, latent_dim)  # Mean
        self.fc_logvar = nn.Linear(2048 * 7 * 7, latent_dim)  # Log variance
    
    def forward(self, x):
        import pdb
        pdb.set_trace
        x = self.resnet(x)  # Extract features using ResNet backbone
        x = self.flatten(x)  # Flatten the features
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Define the decoder network
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_channels, img_size):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim // 16, hidden_dim // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim // 8, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim // 4, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # To ensure the output is between 0 and 1
        )
        self.img_size = img_size
    
    def forward(self, z):
        import pdb
        pdb.set_trace
        
        h = torch.relu(self.fc(z))
        h = h.view(-1, h.size(1) // 16, self.img_size // 16, self.img_size // 16)
        x_recon = self.deconv(h)
        return x_recon

# Define the VAE network
class VAE(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_channels, img_size):
        super(VAE, self).__init__()
        self.encoder = ResNetEncoder(latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_channels, img_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

# Define the loss function (Reconstruction + KL Divergence loss)
def vae_loss(x_recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')  # Use MSE for reconstruction loss
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL Divergence
    return recon_loss + kl_div

# Training settings
latent_dim = 128
hidden_dim = 512 * 16  # This is flexible based on the decoder architecture
output_channels = 3  # For RGB images
img_size = 224  # Size of ImageNet images
learning_rate = 1e-4
batch_size = 64
epochs = 10

# Define data transformations for ImageNet
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet dataset
train_dataset = datasets.ImageFolder(root='/lustre/orion/bif146/world-shared/enzhi/imagenet2012', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Initialize the VAE model, optimizer, and loss function
vae = VAE(latent_dim, hidden_dim, output_channels, img_size)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# Training loop
vae.train()
for epoch in range(epochs):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item() / len(data)}')

    print(f"Epoch {epoch + 1}, Average Loss: {train_loss / len(train_loader.dataset)}")

# Save the trained VAE model
torch.save(vae.state_dict(), 'vae_resnet_imagenet.pth')