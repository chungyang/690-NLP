
from torch.utils.data import Dataset
import torch, Tags, numpy as np


class TranslationDataset(Dataset):

    def __init__(self, src_sentences, tgt_sentences):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences

    def __getitem__(self, idx):
        return self.src_sentences[idx], self.tgt_sentences[idx]

    def __len__(self):
        return len(self.src_sentences)


