import argparse
import Tags

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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', required=True)
    parser.add_argument('-train_tgt', required=True)
    # parser.add_argument('-valid_src', required=True)
    # parser.add_argument('-valid_tgt', required=True)
    # parser.add_argument('-save_data', required=True)
    parser.add_argument('-max_len', type=int, default=50)
    parser.add_argument('-min_word_count', type=int, default=5)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)

    options = parser.parse_args()
    # Include <BOS> <EOS> tokens
    options.max_len += 2

    # Get training instances
    train_src = get_instances(options.train_src, options.max_len)
    train_tgt = get_instances(options.train_tgt, options.max_len)

    # Make sure src and tgt have the same size
    train_src, train_tgt = match_size(train_src, train_tgt)

    




if __name__ == "__main__":
    main()
