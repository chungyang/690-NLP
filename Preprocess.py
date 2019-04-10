import argparse, Tags, torch, numpy as np

def get_sentences(file, max_len):
    """
    Append <BOS> and <EOS> tokens to each sentence and convert them into lists of words

    :param file: file that stores the sentences
    :param max_len: max sentence length
    :return: lists of words
    """
    sentences = []
    n_trimmed = 0

    with open(file) as f:

        for sentence in f:

            words = sentence.split()
            words = words[:max_len]

            if len(words) > max_len:
                n_trimmed += 1

            if words:
                words = [Tags.BOS] + words + [Tags.EOS]
                sentences.append(words)
            else:
                sentences.append([None])

    print('[Info] Get {} sentences from {}'.format(len(sentences), file))

    if n_trimmed:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'.format(n_trimmed, max_len))

    return sentences

def match_size(first_sentences, second_sentences):
    """
    Match the number of sentences if they are mismatched. Pick the smaller set of sentences and
    reduce the size of the bigger to match the smaller one.

    :param first_sentences: first set of instances
    :param second_sentences: second set of instances
    :return: two instances with matched size
    """

    if len(first_sentences) != len(second_sentences):
        print('[Warning] Setences size mismatch')
        smaller_size = min(len(first_sentences), len(second_sentences))
        first_sentences = first_sentences[:smaller_size]
        second_sentences = second_sentences[:smaller_size]

    return first_sentences, second_sentences


def get_idx_seq(sentences, word2idx):
    """
    Replace words in sentences with an ID that's specified in word2idx.

    :param sentences: lists of words
    :param word2idx: dictionary that maps each word to an unique ID
    :return: lists of IDs
    """

    return [[word2idx.get(w.lower(), Tags.UNK_ID) for w in s] for s in sentences]

def get_word2index(en_emb, de_emb, emb_size, train_src, train_tgt):
    """
    Use a pretrained embedding to generate word to index dictionary

    """

    word2idx_en = {Tags.BOS:Tags.BOS_ID, Tags.EOS:Tags.EOS_ID, Tags.UNK:Tags.UNK_ID, Tags.PAD:Tags.PAD_ID}
    word2idx_de = {Tags.BOS:Tags.BOS_ID, Tags.EOS:Tags.EOS_ID, Tags.UNK:Tags.UNK_ID, Tags.PAD:Tags.PAD_ID}

    all_src_vocab = set(w.lower() for sentence in train_src for w in sentence)
    all_tgt_vocab = set(w.lower() for sentence in train_tgt for w in sentence)

    glove_en = {}
    glove_de = {}

    # Randomly Initialize embeddings for <BOS>, <EOS>, <PAD> and <UNK>
    bos_vector = np.random.uniform(low=-0.25, high=0.25, size=emb_size).tolist()
    eos_vector = np.random.uniform(low=-0.25, high=0.25, size=emb_size).tolist()
    pad_vector = np.array([0.0] * emb_size).tolist()
    unk_vector = np.random.uniform(low=-0.25, high=0.25, size=emb_size).tolist()

    glove_en[Tags.BOS] = bos_vector
    glove_de[Tags.BOS] = bos_vector
    glove_en[Tags.PAD] = pad_vector
    glove_de[Tags.PAD] = pad_vector
    glove_en[Tags.EOS] = eos_vector
    glove_de[Tags.EOS] = eos_vector
    glove_en[Tags.UNK] = unk_vector
    glove_de[Tags.UNK] = unk_vector
    
    print('[Info] Getting word2idx and pretrain embeddings')

    with open("./embeddings/" + de_emb) as f:

        for l in f:
            line = l.split()
            word = line[0]
            # Only retrieve vocabs that exist in training src
            if word in all_src_vocab:
                word2idx_de[word] = len(word2idx_de) + 1
                glove_de[word] = line[1:]

    with open("./embeddings/" + en_emb) as f:

        for l in f:
            line = l.split()
            word = line[0]
            # Only retrieve vocabs that exist in training tgt
            if word in all_tgt_vocab:
                word2idx_en[word] = len(word2idx_en) + 1
                glove_en[word] = line[1:]

    return word2idx_de, glove_de, word2idx_en, glove_en

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', required=True)
    parser.add_argument('-train_tgt', required=True)
    parser.add_argument('-dev_src', required=True)
    parser.add_argument('-dev_tgt', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-max_len', type=int, default=50)

    options = parser.parse_args()
    # Include <BOS> <EOS> tokens
    options.max_len += 2

    # Get training sentences
    train_src = get_sentences(options.train_src, options.max_len)
    train_tgt = get_sentences(options.train_tgt, options.max_len)

    # Make sure src and tgt have the same size
    train_src, train_tgt = match_size(train_src, train_tgt)

    # Get dev sentences
    dev_src = get_sentences(options.dev_src, options.max_len)
    dev_tgt = get_sentences(options.dev_tgt, options.max_len)

    # Make sure src and tgt have the same size
    dev_src, dev_tgt = match_size(dev_src, dev_tgt)

    # Get word2idx for training data
    src_word2idx, glove_de, tgt_word2idx, glove_en = get_word2index("glove.6B.300d.txt", "de.300.txt", 300, train_src, train_tgt)
    
    # Convert words to IDs
    print('[Info] Converting words to IDs')
    train_src_idx_seq = get_idx_seq(train_src, src_word2idx)
    train_tgt_idx_seq = get_idx_seq(train_tgt, tgt_word2idx)

    dev_src_idx_seq = get_idx_seq(dev_src, src_word2idx)
    dev_tgt_idx_seq = get_idx_seq(dev_tgt, tgt_word2idx)

    options.src_vocab_size = len(src_word2idx)
    options.tgt_vocab_size = len(tgt_word2idx)


    data = {"word2idx":
                { "src": src_word2idx, "tgt": tgt_word2idx},
            "train":
                {"src": train_src_idx_seq, "tgt": train_tgt_idx_seq},
            "dev":
                {"src": dev_src_idx_seq, "tgt": dev_tgt_idx_seq},
            "glove":
                {"src": glove_de, "tgt": glove_en},
            "options": options
            }

    # Save processed data
    print('[Info] Saving preprocessed data')
    torch.save(data, options.save_data)


if __name__ == "__main__":
    main()
