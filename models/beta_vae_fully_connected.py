import torch
import torch.nn as nn


class VAE_FConnected(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=256, latent_dim=16):
        super(VAE_FConnected, self).__init__()

        # encoder: convolutional layers + fully connected layers
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),  # output: [batch_size, 256, 64, 64]
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),  # output: [batch_size, 512, 32, 32]
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),  # output: [batch_size, 1024, 16, 16]
            nn.LeakyReLU(0.2)
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_dim * 4 * 16 * 16, latent_dim) # alterar 16 com base na proporção da img

        self.mean_layer = nn.Linear(latent_dim, latent_dim)
        self.logvar_layer = nn.Linear(latent_dim, latent_dim)

        # decoder: fully connected layers + convolutional transpose layers
        self.fc2 = nn.Linear(latent_dim, hidden_dim * 4 * 16 * 16)  # alterar 16 com base na proporção da img

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),  # output: [batch_size, 512, 32, 32]
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),  # output: [batch_size, 256, 64, 64]
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(hidden_dim, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # output: [batch_size, 1, 128, 128]
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        
        return mean, logvar

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z

    def decode(self, z):
        x = self.fc2(z)
        x = x.view(-1, 1024, 16, 16)
        return self.decoder_conv(x)

    def forward(self, x):
        mean, log_var = self.encode(x)

        z = self.reparameterization(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var
