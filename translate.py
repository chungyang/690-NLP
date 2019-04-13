import torch
import argparse
import transformer.Transformer as t
import TranslationDataset as td
from transformer.utils import *
from tqdm import tqdm




def translate(model_path, de_sentences, params, device):
    """
    Translate german to english

    :param model_path: saved model path
    :param de_sentences: input german sentences
    :param params: transformer params
    :param device: pytorch device
    :return:
    """

    # Reconstruct the transformer
    model = t.Transformer(src_embedding = params["src_embedding"],
        tgt_embedding = params["tgt_embedding"],
        len_max_seq = params["len_max_seq"],
        d_k = params["d_k"],
        d_v = params["d_v"],
        d_model = params["d_model"],
        d_word_vec = params["d_word_vec"],
        d_inner = params["d_inner"],
        n_layers = params["n_layers"],
        n_head = params["n_head"],
        dropout = params["dropout"]).to(device)

    # Restore pretrain model state
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load data
    test_data = torch.utils.data.DataLoader(
        td.TranslationDataset(
            src_sentences = de_sentences),
        num_workers = 2,
        batch_size = params.batch_size,
        collate_fn = paired_collate_fn)

    with torch.no_grad():
        for batch in tqdm(
                test_data, mininterval = 2,
                desc='  - (Translating) ', leave=False):

            src_seq, src_pos = map(lambda x: x.to(device), batch)







if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)
    parser.add_argument('-params', type=str, defualt="model/transformer_params")
    parser.add_argument('-model_path', type=str, required=True)
    parser.add_argument('-cuda', action='store_true')

    options = parser.parse_args()

    device = torch.device("cuda" if options.cuda else "cpu")
    params = torch.load(options.params)
    data = torch.load(options.data)

    translate(options.model_path, data, params, device )