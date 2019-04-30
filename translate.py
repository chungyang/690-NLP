import torch
from Process import *
import argparse
from Models import get_model
from Beam import beam_search
from nltk.corpus import wordnet
import nltk
from torch.autograd import Variable
import re

def get_synonym(word, SRC):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if SRC.vocab.stoi[l.name()] != 0:
                return SRC.vocab.stoi[l.name()]
            
    return 0


def capitalize(sentence):
    """
    Capitalize the first letter and letter after punctuations such as : and .

    :param sentence: a sentence str
    :return: capitalized sentence str
    """
    # First converted the sentence to a list so we can capitalize it based on words and not
    # character
    sentence = sentence.split(" ")
    capitalize_this = False
    capitalized_sentence = []

    # Capitalize the first word no matter what
    first_word = sentence[0].capitalize()
    capitalized_sentence.append(first_word + " ")

    for word in sentence[1:]:

        if word == "i" or capitalize_this:
            word = word.capitalize()
            capitalize_this = False
        elif word == "." or word == ":":
            capitalize_this = True

        capitalized_sentence.append(word)
        capitalized_sentence.append(" ")
    # strip the last white space
    return "".join(capitalized_sentence[:-1])


def multiple_replace(dict, text):
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

def translate_sentence(sentence, model, opt, SRC, TRG):
    
    model.eval()
    indexed = []
    sentence = SRC.preprocess(sentence)
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0 or opt.floyd == True:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(get_synonym(tok, SRC))
    sentence = Variable(torch.LongTensor([indexed]))
    if opt.device == 0:
        sentence = sentence.cuda()
    
    sentence = beam_search(sentence, model, SRC, TRG, opt)
    sentence = capitalize(sentence)

    return  multiple_replace({' ?' : '?',' !':'!',' .':'.','\' ':'\'',' ,':',', " '":"'"}, sentence)

def translate(sentences, opt, model, SRC, TRG):
#     sentences = opt.text.lower().split('.')
    translated = []

    for sentence in sentences:
        trainslate_sentence = translate_sentence(sentence, model, opt, SRC, TRG)
        print(sentence, " " , trainslate_sentence)
        translated.append(trainslate_sentence)

    with open('pred.txt', 'w') as f:
        for item in translated:
            f.write("%s\n" % item)

    return (' '.join(translated))


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', required=True)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=80)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-src_lang', required=True)
    parser.add_argument('-trg_lang', required=True)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-dev_data', default='data/dev.de')
    
    opt = parser.parse_args()

    opt.device = 0 if opt.no_cuda is False else -1
 
    assert opt.k > 0
    assert opt.max_len > 10

    SRC, TRG = create_fields(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    sentences = open(opt.dev_data, encoding='utf-8').read().split('\n')
    
    phrase = translate(sentences, opt, model, SRC, TRG)
    print('> ' + phrase + '\n')

if __name__ == '__main__':
    nltk.download('wordnet')
    main()
