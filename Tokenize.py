import spacy
import re

class tokenize(object):
    
    def __init__(self, lang, keep_punc=True):
        self.nlp = spacy.load(lang)
        self.keep_punc = keep_punc
            
    def tokenizer(self, sentence):
        if not self.keep_punc:
            sentence = re.sub(
            r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]
