import torch
import torch.nn as nn


class VAE_FConv(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=256, latent_dim=16):
        super(VAE_FConv, self).__init__()

        # Encoder: convolutional layers
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),  # output: [batch_size, 256, 64, 64]
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),  # output: [batch_size, 512, 32, 32]
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),  # output: [batch_size, 1024, 16, 16]
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim * 4, latent_dim, kernel_size=3, stride=2, padding=1)  # output: [batch_size, latent_dim, 8, 8]
        )

        # Layers for the mean and logvar
        self.mean_layer = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)
        self.logvar_layer = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)

        # Decoder: transposed convolutional layers
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dim * 4, kernel_size=3, stride=2, padding=1, output_padding=1),  # output: [batch_size, 1024, 16, 16]
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),  # output: [batch_size, 512, 32, 32]
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),  # output: [batch_size, 256, 64, 64]
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(hidden_dim, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # output: [batch_size, 1, 128, 128]
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z

    def decode(self, z):
        return self.decoder_conv(z)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var
