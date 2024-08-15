import os
import csv
import time
import torch
import torch.optim as optim
import argparse

from tqdm import tqdm
from utils.hdf5_utils import HDF5DataHandler

from models.vae_fully_conv import VAE_FConv
from models.beta_vae_fully_connected import VAE_FConnected
from utils.training_utils import loss_function, save_checkpoint


class Trainer:
    def __init__(self, model, optimizer, device, train_loader, epochs=20, batch_size=128, csv_file='training_log.csv'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.epochs = epochs
        self.batch_size = batch_size
        self.csv_file = csv_file

        os.makedirs('checkpoints', exist_ok=True)

    def train(self):
        self.model.train()
        best_loss = float('inf')
        best_model_state = None
        best_epoch = 0

        with open(self.csv_file, 'w', newline='') as csvfile:
            fieldnames = ['epoch', 'duration', 'average_loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for epoch in range(self.epochs):
                start_time = time.time()
                overall_loss = 0

                for batch_idx, (x,) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")):
                    x = x.to(self.device)

                    self.optimizer.zero_grad()

                    x_hat, mean, log_var = self.model(x)

                    loss = loss_function(x, x_hat, mean, log_var)

                    overall_loss += loss.item()

                    loss.backward()
                    self.optimizer.step()

                duration = time.time() - start_time
                average_loss = overall_loss / (batch_idx * self.batch_size)

                # Salva os dados no csv
                writer.writerow({'epoch': epoch + 1, 'duration': duration, 'average_loss': average_loss})

                print(">> Time", duration, "\tEpoch", epoch + 1, "\tAverage Loss: ", average_loss)
                print("")

                # Verifica se a perda atual é a melhor e armazena o modelo se for
                if average_loss < best_loss:
                    best_loss = average_loss
                    best_epoch = epoch + 1
                    best_model_state = self.model.state_dict()

            # Salva o melhor modelo ao final do treinamento
            if best_model_state:
                save_checkpoint(self.model, self.optimizer, best_epoch, path=f"checkpoints/best_vae_checkpoint_epoch_{best_epoch}.pth")

            # Salva o modelo da última época
            save_checkpoint(self.model, self.optimizer, self.epochs, path=f"checkpoints/last_vae_checkpoint_epoch_{self.epochs}.pth")

        return overall_loss


def main():
    parser = argparse.ArgumentParser(description='VAE Training')
    parser.add_argument('--path', type=str, required=True, help='Path to the HDF5 file')
    parser.add_argument('--model', type=str, choices=['fconv', 'fconnected'], required=True, help='Choose the VAE model: "fconv" or "fconnected"')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    args = parser.parse_args()

    handler = HDF5DataHandler(args.path, test_size=0.2)
    handler.load_data(split=True, norm=True)  
    handler.prepare_tensors()
    train_loader, _ = handler.get_loaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Escolha do modelo com base no argumento passado
    if args.model == 'fconv':
        model = VAE_FConv().to(device)
    elif args.model == 'fconnected':
        model = VAE_FConnected().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    trainer = Trainer(model=model, optimizer=optimizer, device=device, train_loader=train_loader, epochs=args.epochs, batch_size=args.batch_size)
    trainer.train()


if __name__ == "__main__":
    main()
    
