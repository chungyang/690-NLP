import numpy as np
import Tags
import torch

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Tags.PAD_ID).type(torch.float).unsqueeze(-1)

def get_sine_pos_encoding(n_position, d_hid, padding_idx=None):

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    len_q = seq_q.size(1)
    # Element wise comparison to get a mask for padding
    padding_mask = seq_k.eq(Tags.PAD_ID)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


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
        sentence + [Tags.PAD_ID] * (max_len - len(sentence))
        for sentence in sentences])

    batch_pos = np.array([
        [pos_i+1 if w_i != Tags.PAD_ID else 0
         for pos_i, w_i in enumerate(sentence)] for sentence in batch_sentences])

    batch_sentences = torch.LongTensor(batch_sentences)
    batch_pos = torch.LongTensor(batch_pos)

    return batch_sentences, batch_pos

def greedy_decode(pred, idx2word, batch_size, max_seq_length):
    """
    :param pred: prediction probabilities
    :param idx2word: dictionary to map indices to words
    :param batch_size: size of the batch
    :param seq_length: max seqenece length

    :return: predicted words
    """
    predicted_sentences = []
    indices = torch.argmax(pred, dim = 1)
    indices = indices.view(batch_size, max_seq_length).tolist()

    for i in range(batch_size):
        sentence = []
        for j in range(max_seq_length):
            pred_w = idx2word[indices[i][j]]

            if pred_w == Tags.EOS or pred_w == Tags.PAD:
                break

            sentence.append(pred_w)

        predicted_sentences.append(sentence)

    return predicted_sentences


def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Tags.PAD_ID)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Tags.PAD_ID)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Tags.PAD_ID, reduction='sum')

    return loss