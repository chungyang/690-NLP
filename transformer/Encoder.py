
from transformer.utils import *
from transformer.SubLayers import *

class Encoder(nn.Module):

    def __init__(self, src_embedding, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):
        """
        :param src_embedding: word embedding weights
        :param len_max_seq: max sentence length
        :param d_word_vec: embedding dimension
        :param n_layers: number of encoder layer
        :param n_head: number of attention head
        :param d_k: key vector dimension
        :param d_v: value vector dimension
        :param d_model: self-attention output dimension / fully connected layer input dimension
        :param d_inner: hidden dimension for fully connected layer
        :param dropout: drop out rate
        """

        super().__init__()

        self.src_word_emb = nn.Embedding.from_pretrained(src_embedding, d_word_vec, padding_idx = Tags.PAD_ID, freeze = True)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sine_pos_encoding(len_max_seq, d_word_vec, padding_idx=0),freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos):

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        return enc_output


class EncoderLayer(nn.Module):
    ''' Each encoder layer consists of two sublayers. Multihead attention and feedforward '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
