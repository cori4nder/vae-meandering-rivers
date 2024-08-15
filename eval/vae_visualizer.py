import os
import torch
import imageio
import numpy as np

from PIL import Image, ImageDraw, ImageFont


#  Classe para visualização de variações latentes geradas por um VAE, incluindo a criação de GIFs e vídeos.

class VAEVisualizer:

    def __init__(self, vae, device='cuda', output_dir='./output'):
        self.vae = vae
        self.device = device
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_latent_variations(self, mean, num_variations, latent_range):
        variations = []
        min_val, max_val = latent_range
        steps = np.linspace(min_val, max_val, num=num_variations)
        for step_x in steps:
            for step_y in steps:
                step_vector = torch.zeros_like(mean, device=mean.device)
                step_vector[:, 0] = step_x
                step_vector[:, 1] = step_y
                z = torch.clamp(mean + step_vector, min_val, max_val)
                variations.append((z, (step_x, step_y)))
        return variations

    def generate_gif_or_video(self, test_loader, num_samples=3, num_variations=10, latent_range=(0, 1), output_type='gif', gif_speed=100):
        self.vae.eval()

        data_iter = iter(test_loader)
        images = next(data_iter)

        if isinstance(images, list):
            images = images[0]

        images = images.to(self.device)
        images = images[:num_samples]

        generated_images = []

        with torch.no_grad():
            x_hat, mean, _ = self.vae(images)
            variations = self.generate_latent_variations(mean, num_variations, latent_range)

            for var, (step_x, step_y) in variations:
                decoded_imgs = self.vae.decode(var).cpu()
                decoded_imgs = (decoded_imgs - decoded_imgs.min()) / (decoded_imgs.max() - decoded_imgs.min())
                decoded_imgs = decoded_imgs.numpy().transpose(0, 2, 3, 1)

                combined_img_list = []

                for idx, img in enumerate(decoded_imgs):
                    var_img = (img[:, :, 0] * 255).astype(np.uint8)
                    combined_img_list.append(var_img)

                combined_var_img = np.hstack(combined_img_list)
                img_pil = Image.fromarray(combined_var_img, mode='L')

                draw = ImageDraw.Draw(img_pil)
                text = f"({step_x:.2f}, {step_y:.2f})"
                font = ImageFont.load_default()
                text_position = (10, 10)
                draw.text(text_position, text, fill="white", font=font)

                generated_images.append(img_pil)

        output_path = os.path.join(self.output_dir, f'vae_animation.{output_type}')

        if output_type == 'gif':
            generated_images[0].save(output_path, save_all=True, append_images=generated_images[1:], duration=gif_speed, loop=0)
        elif output_type == 'video':
            imageio.mimsave(output_path, generated_images, fps=1000//gif_speed)

        print(f'{output_type.capitalize()} salvo com sucesso em {output_path}!')
