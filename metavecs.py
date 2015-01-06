import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import glob
from itertools import groupby
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence

def ngrams(sequence, n):
    count = max(0, len(sequence) - n + 1)
    return (tuple(sequence[i:i+n]) for i in range(count))

def group_parts(text, idx):
    for id, parts, in groupby(text, lambda i: i[idx]):
        yield list(parts)

def load_text(filename):
    with open(filename) as infile:
        return [line.strip().split('\t') for line in infile]

def group_sentences(text): return group_parts(text, 1)

class CorpusIterator(object):
    def __iter__(self):
        filenames = glob.glob("data/international-folktales/*.tok")
        for filename in filenames:
            item_no = 0
            for sentence in group_sentences(load_text(filename)):
                for ngram in ngrams(sentence, 5):
                    transformation = 0 if not any(word[9].lower() == 'transform' for word in ngram) else 1
                    yield LabeledSentence([word[7].lower() for word in sentence], ['%s_SENT_%s-%s' % (filename, transformation, item_no)])
                    item_no += 1

sentences = CorpusIterator()
model = Doc2Vec(sentences, size=100, window=5, min_count=2, workers=12)