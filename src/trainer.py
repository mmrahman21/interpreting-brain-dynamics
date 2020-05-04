import torch
import torch.nn as nn

class Trainer(nn.Module):
    def __init__(self, encoder, wandb, device=torch.device('cpu')):
        super(Trainer, self).__init__()
        self.encoder = encoder
        self.wandb = wandb
        self.device = device

    def generate_batch(self, episodes):
        raise NotImplementedError

    def train(self, episodes):
        raise NotImplementedError

    def log_results(self, epoch_idx, epoch_loss, accuracy):
        raise NotImplementedError

