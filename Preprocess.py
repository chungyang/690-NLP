import argparse
import Tags
import torch

def get_instances(file, max_len):
    """
    Append <BOS> and <EOS> tokens to each sentence and convert them into lists of words

    :param file: file that stores the sentences
    :param max_len: max sentence length
    :return: lists of words
    """
    instances = []
    n_trimmed = 0

    with open(file) as f:

        for sentence in f:

            words = sentence.split()
            words = words[:max_len]

            if len(words) > max_len:
                n_trimmed += 1

            if words:
                words = [Tags.BOS] + words + [Tags.EOS]
                instances.append(words)
            else:
                instances.append([None])

    print('[Info] Get {} instances from {}'.format(len(instances), file))

    if n_trimmed:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'.format(n_trimmed, max_len))

    return instances

def match_size(first_instances, second_instances):
    """
    Match the number of instances if they are mismatched. Pick the smaller set of instances and
    reduce the size of the bigger to match the smaller one.

    :param first_instances: first set of instances
    :param second_instances: second set of instances
    :return: two instances with matched size
    """

    if len(first_instances) != len(second_instances):
        print('[Warning] Instances size mismatch')
        smaller_size = min(len(first_instances), len(second_instances))
        first_instances = first_instances[:smaller_size]
        second_instances = second_instances[:smaller_size]

    return first_instances, second_instances

def word2index(sentences):
    """
    Maps each word in sentences to an unique ID

    :param sentences: lists of words
    :return: dictionary that maps each word in sentences to an unique ID
    """

    all_vocab = set(w for sentence in sentences for w in sentence)

    word2idx = {Tags.BOS:Tags.BOS_ID, Tags.EOS:Tags.EOS_ID, Tags.UNK:Tags.UNK_ID, Tags.PAD:Tags.PAD_ID}

    for vocab in all_vocab:
        word2idx[vocab] = len(word2idx)

    return word2idx

def get_idx_seq(sentences, word2idx):
    """
    Replace words in sentences with an ID that's specified in word2idx.

    :param sentences: lists of words
    :param word2idx: dictionary that maps each word to an unique ID
    :return: lists of IDs
    """

    return [[word2idx.get(w, Tags.UNK) for w in s] for s in sentences]

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

    # Get training instances
    train_src = get_instances(options.train_src, options.max_len)
    train_tgt = get_instances(options.train_tgt, options.max_len)

    # Make sure src and tgt have the same size
    train_src, train_tgt = match_size(train_src, train_tgt)

    # Get dev instances
    dev_src = get_instances(options.dev_src, options.max_len)
    dev_tgt = get_instances(options.dev_tgt, options.max_len)

    # Make sure src and tgt have the same size
    dev_src, dev_tgt = match_size(dev_src, dev_tgt)

    # Get word2idx for training data
    src_word2idx = word2index(train_src)
    tgt_word2idx = word2index(train_tgt)

    # Convert words to IDs
    train_src_idx_seq = get_idx_seq(train_src, src_word2idx)
    train_tgt_idx_seq = get_idx_seq(train_tgt, tgt_word2idx)

    dev_src_idx_seq = get_idx_seq(dev_src, src_word2idx)
    dev_tgt_idx_seq = get_idx_seq(dev_tgt, tgt_word2idx)

    data = {"word2idx":
                { "src": src_word2idx, "tgt": tgt_word2idx},
            "train":
                {"src": train_src_idx_seq, "tgt": train_tgt_idx_seq},
            "dev":
                {"src": dev_src_idx_seq, "tgt": dev_tgt_idx_seq}
            }

    # Save processed data
    torch.save(data, options.save_data)


if __name__ == "__main__":
    main()
