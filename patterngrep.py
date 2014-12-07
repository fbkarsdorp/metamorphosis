import codecs
import glob
import sys

from functools import partial
from itertools import groupby
from termcolor import colored

from pattern.search import compile
from pattern.en import Sentence

FIELDS = {field: i for i, field in enumerate(
    ['paragraphId', 'sentenceID', 'tokenId', 'beginOffset', 'endOffset',
     'whitespaceAfter', 'headTokenId', 'word', 'normalizedWord',
     'lemma', 'pos', 'ner', 'deprel', 'inQuotation', 'characterId'])}

Sentence = partial(Sentence, token=['word', 'part-of-speech', 'lemma'])

def get_word_fields(word, features=('word', 'pos', 'lemma')):
    return [word[FIELDS[feature]] for feature in features]

def format_word(word, features=('word', 'pos', 'lemma')):
    return '/'.join(get_word_fields(word, features))

def format_sentence(sentence, features=('word', 'pos', 'lemma')):
    return ' '.join(format_word(word, features) for word in sentence)

def group_parts(text, idx):
    for id, parts in groupby(text, lambda i: i[idx]):
        yield list(parts)

def group_sentences(text):
    return group_parts(text, 1)

def group_paragraphs(text):
    return group_parts(text, 0)

def grep(pattern, sentences):
    pattern = compile(pattern)
    for id, sentence in sentences:
        match = pattern.search(sentence)
        if match:
            yield id, color_hit(sentence, match)

def color_hit(sentence, matches):
    done = set()
    s = [word.string for word in sentence]
    for match in matches:
        for i, word in enumerate(sentence):
            if word.index >= match.start and word.index < match.stop and i not in done:
                done.add(i)
                s[i] = colored(word.string, "red")
    return ' '.join(s)

def load_text(filename):
    lines = [line.strip().split('\t') for line in open(filename)][1:]
    return [Sentence(format_sentence(sentence)) for sentence in group_sentences(lines)]

def load_data(filenames):
    return [(filename, sentence) for filename in filenames
                                 for sentence in load_text(filename)]

if __name__ == '__main__':
    sentences = load_data(glob.glob(sys.argv[1]))
    while True:
        search_pattern = raw_input("Pattern: ").strip()
        for id, match in grep(search_pattern, sentences):
            print colored(id, "blue"), match
