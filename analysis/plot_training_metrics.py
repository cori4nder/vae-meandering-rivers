import pandas as pd
import matplotlib.pyplot as plt


class TrainingPlotter:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)

    def plot_metrics(self):
        """Plota os gráficos de perda média e duração por época lado a lado."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot da perda média por época
        axes[0].plot(self.data['epoch'], self.data['average_loss'], marker='o', label='Average Loss', color='b')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Average Loss')
        axes[0].set_title('Average Loss per Epoch')
        axes[0].legend()
        axes[0].grid(True)

        # Plot da duração por época
        axes[1].plot(self.data['epoch'], self.data['duration'], marker='o', label='Duration', color='r')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Duration (seconds)')
        axes[1].set_title('Duration per Epoch')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

# Exemplo de uso:
# plotter = TrainingPlotter('training_log.csv')
# plotter.plot_metrics()
