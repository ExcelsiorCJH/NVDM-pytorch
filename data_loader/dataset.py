import os
import copy
import torch
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class NVDMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        valid_size: float,
        batch_size: int,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.valid_size = valid_size
        self.batch_size = batch_size

    def setup(self, stage=None):
        train_docs, train_word_counts, vocab = self._load_data(self.data_dir, stage="train")
        test_docs, test_word_counts, _ = self._load_data(self.data_dir, stage="test")

        train_docs, valid_docs, train_word_counts, valid_word_counts = train_test_split(
            train_docs, train_word_counts, test_size=self.valid_size, shuffle=True
        )

        self.vocab = vocab
        self.trainset = DocDataset(train_docs, train_word_counts, len(vocab))
        self.validset = DocDataset(valid_docs, valid_word_counts, len(vocab))
        self.testset = DocDataset(test_docs, test_word_counts, len(vocab))

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=1, shuffle=False)

    def _load_data(self, data_dir: str, stage: str = "train"):
        docs, word_counts = self._read_docs(os.path.join(data_dir, f"{stage}.feat"))

        index_word = []
        with open(os.path.join(data_dir, "vocab")) as f:
            index_word.extend([line.strip().split()[0] for line in f])

        vocab = Vocab(index_word)
        return docs, word_counts, vocab

    def _read_docs(self, file_path: str):
        docs, word_counts = [], []
        with open(file_path) as f:
            for line in f:
                doc = {}
                cnt = 0
                for id_freq in line.split()[1:]:
                    items = id_freq.split(":")
                    doc[int(items[0]) - 1] = int(items[1])
                    cnt += int(items[1])
                if cnt > 0:
                    docs.append(doc)
                    word_counts.append(cnt)

        return docs, word_counts


class DocDataset(Dataset):
    def __init__(self, docs, word_counts, vocab_size):
        super().__init__()
        self.docs = docs
        self.word_counts = word_counts
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        word_cnt = self.word_counts[idx]

        # BoW representation
        bow = np.zeros(self.vocab_size)
        for w_idx, freq in doc.items():
            bow[w_idx] = freq

        return {"doc": torch.FloatTensor(bow), "word_cnt": word_cnt}


class Vocab:
    def __init__(self, itos):
        self.itos = copy.copy(itos)
        self.stoi = {v: k for k, v in enumerate(itos)}

    def __call__(self, s):
        return self.stoi[s]

    def __len__(self):
        return len(self.itos)
