import torch
import torch.nn as nn


def loss_function(x, x_hat, mean, log_var, clamp=(-10, 10), beta=1.1):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    log_var_clamp = torch.clamp(log_var, min=clamp[0], max=clamp[1])
    KLD = -0.5 * torch.sum(1 + log_var_clamp - mean.pow(2) - log_var_clamp.exp())

    return reproduction_loss + beta * KLD


def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    torch.save(checkpoint, path)

    print(f"| OK >> {path}")
    print("")
