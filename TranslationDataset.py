
from torch.utils.data import Dataset

class TranslationDataset(Dataset):

    def __init__(self, src_sentences, tgt_sentences = None):
        self._src_sentences = src_sentences
        self._tgt_sentences = tgt_sentences

    def __getitem__(self, idx):
        if self._tgt_sentences:
            return self._src_sentences[idx], self._tgt_sentences[idx]

        return self._src_sentences[idx]

    def __len__(self):
        return len(self.src_sentences)


