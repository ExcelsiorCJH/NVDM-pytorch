import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class NVDM(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 500,
        z_dim: int = 200,
        num_sample: int = 1,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.num_sample = num_sample
        self.lr = learning_rate

        # encoder (doc -> vectors)
        self.enc1 = nn.Linear(vocab_size, hidden_dim * 2)
        self.enc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        # mean of Guassian distribution
        self.mean = nn.Linear(hidden_dim, z_dim)
        # log sigma of Guassian distribution
        self.log_sigma = nn.Linear(hidden_dim, z_dim)

        # decoder
        self.dec1 = nn.Linear(z_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.dec3 = nn.Linear(hidden_dim * 2, vocab_size)

    def encoder(self, x):
        x = F.relu(self.enc1(x))
        encoded = F.relu(self.enc2(x))
        mean = self.mean(encoded)
        log_sigma = self.log_sigma(encoded)
        kld = -0.5 * torch.sum(1 - torch.square(mean) + 2 * log_sigma - torch.exp(2 * log_sigma), 1)
        return mean, log_sigma, kld

    def decoder(self, mean, log_sigma, x):
        if self.num_sample == 1:  # single sample
            eps = torch.rand(self.batch_size, self.z_dim).type_as(x)
            z_vec = torch.mul(torch.exp(log_sigma), eps) + mean
            decoded = F.relu(self.dec1(z_vec))
            decoded = F.relu(self.dec2(decoded))
            logits = F.log_softmax(self.dec3(decoded), dim=1)
            recons_loss = -torch.sum(torch.mul(logits, x), 1)
        else:  # multi samples
            eps = torch.rand(self.num_sample * self.batch_size, self.z_dim).type_as(x)
            eps_list = list(eps.reshape(self.num_sample, self.batch_size, self.z_dim))

            recons_loss_list = []
            for idx in range(self.num_sample):
                curr_eps = eps_list[idx]
                z_vec = torch.mul(torch.exp(log_sigma), curr_eps) + mean
                logits = F.log_softmax(self.dec(z_vec))
                recons_loss_list.append(-torch.sum(torch.mul(logits, x), 1))

            recons_loss_list = torch.tensor(recons_loss_list)
            recons_loss = torch.sum(recons_loss_list, dim=1) / self.num_sample

        return z_vec, logits, recons_loss

    def forward(self, bow_docs):
        self.batch_size = len(bow_docs)
        mean, log_sigma, kld = self.encoder(bow_docs)
        z_vec, logits, recons_loss = self.decoder(mean, log_sigma, bow_docs)
        return z_vec, logits, kld, recons_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=4e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self._evaluate(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._evaluate(batch, batch_idx, "val")
        return loss

    def test_step(self, batch, batch_idx):
        self._evaluate(batch, batch_idx, "test")

    def _evaluate(self, batch, batch_idx, stage):
        z_vec, logits, kld, recons_loss = self.forward(batch["doc"])
        loss = kld + recons_loss
        loss = loss.mean()
        self.log(f"{stage}_kld", kld.mean())
        self.log(f"{stage}_recons_loss", recons_loss.mean())
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss
