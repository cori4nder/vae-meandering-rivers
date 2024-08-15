import torch
import numpy as np
import matplotlib.pyplot as plt


class VAEEvaluator:

    def __init__(self, model, device='cuda', image_size=128):
        self.model = model
        self.device = device
        self.image_size = image_size

    def load_model(self, checkpoint_path, optimizer=None):
        if optimizer:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            print(f"| Loaded >> {checkpoint_path} | Epoch >> {epoch}")
            return epoch
        else:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"| Loaded >> {checkpoint_path}")
            return None

    def generate_latent_sample(self, latent_values, latent_dim):
        z_sample = torch.zeros(1, latent_dim).to(self.device)
        for i, value in enumerate(latent_values):
            z_sample[0, i] = value
        x_decoded = self.model.decode(z_sample)
        s_img = x_decoded.detach().cpu().reshape(self.image_size, self.image_size)
        plt.imshow(s_img, cmap='gray')
        plt.axis('off')
        plt.show()

    def visualize_latent_space(self, scale=2.0, n=25, figsize=15):
        figure = np.zeros((self.image_size * n, self.image_size * n))
        grid_x = np.linspace(-scale, scale, n)
        grid_y = np.linspace(-scale, scale, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = torch.zeros(1, 16).to(self.device)
                z_sample[0, 0] = xi
                z_sample[0, 1] = yi
                x_decoded = self.model.decode(z_sample)
                s_img = x_decoded[0].detach().cpu().reshape(self.image_size, self.image_size)
                figure[i * self.image_size : (i + 1) * self.image_size, j * self.image_size : (j + 1) * self.image_size] = s_img

        plt.figure(figsize=(figsize, figsize))
        plt.title('VAE Latent Space Visualization')
        start_range = self.image_size // 2
        end_range = n * self.image_size + start_range
        pixel_range = np.arange(start_range, end_range, self.image_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap="Greys_r")
        plt.show()

    def visualize_latent_variations(self, test_loader, num_images=5, steps=((0.1, 0.1), (0.2, 0.5))):

        with torch.no_grad():
            for x, in test_loader:
                x = x.to(self.device)
                x_hat, mean, _ = self.model(x)
                variations = self._generate_latent_variations(mean, steps)

                decoded_variations = [self.model.decode(var).cpu() for var in variations]
                x = x.cpu()
                x_hat = x_hat.cpu()

                fig, axes = plt.subplots(2 + len(steps), num_images, figsize=(num_images * 2, (2 + len(steps)) * 2))

                for i in range(num_images):
                    axes[0, i].imshow(x[i, 0], cmap='gray')
                    axes[0, i].axis('off')
                    if i == 0:
                        axes[0, i].set_title('Original')

                    axes[1, i].imshow(x_hat[i, 0], cmap='gray')
                    axes[1, i].axis('off')
                    if i == 0:
                        axes[1, i].set_title('Reconstructed')

                    for j, decoded_var in enumerate(decoded_variations):
                        axes[j + 2, i].imshow(decoded_var[i, 0], cmap='gray')
                        axes[j + 2, i].axis('off')
                        if i == 0:
                            axes[j + 2, i].set_title(f'Variation {steps[j]}')

                plt.show()
                break

    def show_reconstructions(self, test_loader, num_images=5):
        self.model.eval()

        with torch.no_grad():
            for x, in test_loader:
                x = x.to(self.device)
                x_hat, _, _ = self.model(x)

                x = x.cpu().numpy()
                x_hat = x_hat.cpu().numpy()

                fig, axes = plt.subplots(2, num_images, figsize=(num_images*2, 4))
                for i in range(num_images):
                    axes[0, i].imshow(x[i, 0], cmap='gray')
                    axes[0, i].axis('off')
                    if i == 0:
                        axes[0, i].set_title('Original')

                    axes[1, i].imshow(x_hat[i, 0], cmap='gray')
                    axes[1, i].axis('off')
                    if i == 0:
                        axes[1, i].set_title('Reconstruída')

                plt.show()
                break

    def _generate_latent_variations(self, mean, steps):
        variations = []

        for step in steps:
            step_vector = torch.zeros_like(mean, device=mean.device)
            step_vector[:, :len(step)] = torch.tensor(step, device=mean.device).view(1, -1)
            z = torch.clamp(mean + step_vector, 0.0, 1.0)
            variations.append(z)

        return variations
