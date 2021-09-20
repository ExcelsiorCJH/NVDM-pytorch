import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from data_loader import NVDMDataModule
from model import NVDM


def train(args):
    pl.seed_everything(seed=42, workers=True)

    data_dir = args.data_dir
    data_module = NVDMDataModule(data_dir=data_dir, valid_size=0.2, batch_size=16)

    model = NVDM(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        z_dim=args.latent_dim,
        num_sample=args.num_sample,
        learning_rate=args.learning_rate,
    )

    # callbacks
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:02d}-{val_loss:.5f}",
        monitor="val_loss",
        verbose=True,
        save_last=True,
        save_top_k=3,
        mode="min",
    )

    trainer = pl.Trainer(callbacks=[ckpt_callback], max_epochs=args.epochs, gpus=1)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="NVDM")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--vocab_size", type=int, default=2000)
    parser.add_argument("--hidden_dim", type=int, default=500)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--num_sample", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)

    args = parser.parse_args()

    train(args)
