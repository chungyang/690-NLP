import argparse, torch, Tags, numpy as np
from transformer import Transformer as t
import TranslationDataset as td


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=64)

    parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    options = parser.parse_args()

    # Set device type
    device = torch.device("cuda" if options.cuda else "cpu")

    # Load preprocessed data
    data = torch.load(options.data)
    training_data, dev_data = prepare_dataloaders(data, options)
    src_embedding = data["glove"]["src"]
    tgt_embedding = data["glove"]["src"]


    transformer = t.Transformer(
        src_embedding = data["glove"]["src"],
        tgt_embedding = data["glove"]["tgt"],
        len_max_seq = data["options"].max_len,
        d_k=options.d_k,
        d_v=options.d_v,
        d_model=options.d_model,
        d_word_vec=options.d_word_vec,
        d_inner=options.d_inner_hid,
        n_layers=options.n_layers,
        n_head=options.n_head,
        dropout=options.dropout).to(device)

    print("hey")


def paired_collate_fn(sentences):
    src_sentences, tgt_sentences = list(zip(*sentences))
    src_sentences, src_pos = collate_fn(src_sentences)
    tgt_sentences, tgt_pos = collate_fn(tgt_sentences)
    return (src_sentences, src_pos, tgt_sentences, tgt_pos)

def collate_fn(sentences):
    """
    Pad all sentences to the max sentence length within a batch and position each word
    with corresponding indices. All pad tokens have position of 0

    :param sentences: lists of word indices
    :return: batch of sentences and batch of
    """
    max_len = max(len(sentence) for sentence in sentences)

    batch_sentences = np.array([
        sentence + [Tags.PAD] * (max_len - len(sentence))
        for sentence in sentences])

    batch_pos = np.array([
        [pos_i+1 if w_i != Tags.PAD else 0
         for pos_i, w_i in enumerate(sentence)] for sentence in batch_sentences])

    batch_sentences = torch.LongTensor(batch_sentences)
    batch_pos = torch.LongTensor(batch_pos)

    return batch_sentences, batch_pos

def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        td.TranslationDataset(
            src_sentences=data['train']['src'],
            tgt_sentences=data['train']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        td.TranslationDataset(
            src_sentences=data['dev']['src'],
            tgt_sentences=data['dev']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)
    return train_loader, valid_loader

if __name__ == "__main__":
    main()