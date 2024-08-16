import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as TF

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class HDF5DataHandler:
    def __init__(self, h5_file_path, test_size=0.2, random_state=42):
        self.h5_file_path = h5_file_path
        self.test_size = test_size
        self.random_state = random_state
        self.train_images = None
        self.test_images = None
        self.train_loader = None
        self.test_loader = None

    def load_data(self, split: bool = False, norm: bool = False):

        with h5py.File(self.h5_file_path, 'r') as h5_file:
            print("| Keys  >> ", list(h5_file.keys()))

            self.images = h5_file['train_images'][:]
            print(f'| Shape >> {self.images.shape}')

        if norm:  # Normalizar imagens
            self.images = self.images.astype('float32') / 255.0
            print("| Normalizaed images >> float32 / 255.0")

        # Ajustar o formato de (N, 128, 128, 1) para (N, 1, 128, 128)
        if len(self.images.shape) == 4 and self.images.shape[-1] == 1:
            self.images = np.transpose(self.images, (0, 3, 1, 2))

        if split:
            # Dividir os dados em conjuntos de treino e teste
            self.train_images, self.test_images = train_test_split(self.images, test_size=self.test_size, random_state=self.random_state)
            
            print(f"| Train - {abs((self.test_size - 1.0)) * 100}% >> {self.train_images.shape}")
            print(f"| Test  - {self.test_size * 100}% >> {self.test_images.shape}")
            print("_________________________________________________________________________________")
            print("")

    def prepare_tensors(self):
        # Convertendo as imagens para tensores PyTorch
        self.train_images_tensor = torch.tensor(self.train_images, dtype=torch.float32)
        self.test_images_tensor = torch.tensor(self.test_images, dtype=torch.float32)

        # Criar datasets do PyTorch
        train_dataset = TensorDataset(self.train_images_tensor)
        test_dataset = TensorDataset(self.test_images_tensor)

        # Definir os DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    def get_loaders(self):
        # Retornar os DataLoaders preparados
        return self.train_loader, self.test_loader

    def resize_images(self, new_size=(64, 64)):

        resized_images = np.array([TF.functional.resize(torch.tensor(img), new_size).numpy() for img in self.images])
        self.train_images, self.test_images = train_test_split(resized_images, test_size=self.test_size, random_state=self.random_state)
        self.prepare_tensors()

    def show_samples(self, dataloader, num_images=8):
        # Obter um batch de dados do DataLoader
        images_batch = next(iter(dataloader))
        images = images_batch[0]  

        images = images[:num_images]

        plt.figure(figsize=(num_images, 1))

        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            img = images[i].squeeze()  
            plt.imshow(img, cmap='gray')
            plt.axis('off')

        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
        plt.show()

# Exemplo de uso:
# handler = HDF5DataHandler('/path/to/train_images.h5')
# handler.load_data()
# handler.prepare_tensors()
# train_loader, test_loader = handler.get_loaders()
