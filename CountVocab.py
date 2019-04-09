import re


def get_vocabs(filename, allvocabs):
    """
    :param filename: name of the file
    :param contents:
    :param allvocabs:
    :return:
    """
    with open("data/" + filename) as f:
        contents = f.readlines()
        vocabs = set()
        contents = [re.sub(r'\W+', " ", content).strip() for content in contents]
        words_lists = [re.split("\s+|,\s+", content) for content in contents]

        for words in words_lists:
            vocabs.update(words)

        print(filename , "has" , len(vocabs) , "unique vocabs")
        allvocabs = allvocabs.union(vocabs)

    return allvocabs


if __name__ == "__main__":

    english_vocabs = set()
    german_vocabs = set()

    german_vocabs = get_vocabs("dev.de", german_vocabs)
    english_vocabs = get_vocabs("dev.en", english_vocabs)
    german_vocabs = get_vocabs("train.de", german_vocabs)
    english_vocabs = get_vocabs("train.en", english_vocabs)

    print(len(english_vocabs), "unique english vocabs")
    print(len(german_vocabs), "unique english vocabs")

