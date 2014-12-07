from itertools import groupby
from collections import namedtuple
from alignment import multi_sequence_alignment

def group_parts(text, idx):
    for id, parts, in groupby(text, lambda i: i[idx]):
        yield list(parts)

def group_sentences(text): return group_parts(text, 1)

Word = namedtuple("Word", ["word", "lemma", "pos"])

def get_chunk(chunk):
    return [Word(word[7], word[9], word[10]) for word in chunk]

def find_seed(seed, sentences, n=3, lemma=True):
    idx = 9 if lemma else 7
    for sentence in sentences:
        for i, word in enumerate(sentence):
            if word[idx] == seed:
                yield get_chunk(sentence[i-n:i+n+1])

def score(a, b):
    return sum(a[i] != b[i] for i in range(len(a)))

def align_instances(instances):
    return multi_sequence_alignment(instances, scoring_fn=score)
