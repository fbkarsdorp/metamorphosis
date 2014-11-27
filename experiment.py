import logging
import os
import sys
from collections import namedtuple, defaultdict

import numpy as np
import scipy.sparse as sp
import sqlite3 as sqlite

from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from gensim.utils import tokenize as normalize_text

from pattern.nl import tokenize

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC, SVC

from phd.datasets import load_transformation_dataset

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# some settings
SIZE = 50
WINDOW = 8
WORKERS = 15
MINCOUNT = 2
SAMPLE = 1e-3

class Struct(object):
    "Simple data structure, similar to namedtuple"
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __cmp__(self, other):
        if isinstance(other, Struct):
            return cmp(self.__dict__, other.__dict__)
        else:
            return cmp(self.__dict__, other)

    def __repr__(self):
        args = ['%s=%s' % (k, repr(v)) for (k, v) in vars(self).iteritems()]
        return 'Struct(%s)' % ', '.join(sorted(args))

def load_transformation_dataset(number=2):
    data, target, target_names, idnumbers = [], [], {}, []
    graph = defaultdict(list)
    last_doc_id, doc_id, sent_id = None, 0, 0
    for line in codecs.open(os.path.join(
            PATH, "data/transformation_annotated-%s.txt" % number), "r", "utf-8"):
        idnumber, _, tag, sentence = line.strip().split('|', 3)
        if last_doc_id is None or last_doc_id != idnumber:
            last_doc_id = idnumber
            doc_id += 1
        if sentence.strip():
            tag = target_names.setdefault(tag, len(target_names)+1)
            target.append(tag)
            data.append(sentence)
            graph[doc_id].append(sent_id)
            idnumbers.append(idnumber.replace(".txt", ""))
            sent_id += 1
    return Struct(data=data, target=target, idnumbers=idnumbers,
                  target_names=target_names, graph=graph)

# start with loading the data set
data = load_transformation_dataset(3)
# we will split the data into a test and training set
train, test = train_test_split(
    range(len(data.target)), test_size=0.1, random_state=None)

X_train, y_train = [data.data[i] for i in train], [data.target[i] for i in train]
X_test, y_test = [data.data[i] for i in test], [data.target[i] for i in test]
annotated_ids = [data.idnumbers[i] for i in test.tolist() + train.tolist()]

class CorpusLoader():
    root = '/vol/bigdata/corpora/TWENTE/'
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def __iter__(self):
        sent_id = 0
        for filename in os.listdir(CorpusLoader.root):
            if filename.endwith(".tok"):
                with codecs.open(os.path.join(CorpusLoader.root, filename),
                                 encoding='utf-8') as infile:
                    for sentence in infile:
                        yield LabeledSentence(
                            list(normalize_text(sentence, lowercase=True)),
                            ["U_background_%s" % sent_id])
                        sent_id += 1
        for sentence in test:
            yield sentence
        for sentence in train:
            yield sentence

def make_label(i, label, phase):
    return "%s_%s_%s" % ("T" if label == 1 else 0 if label != "U" else label, phase, i)

def labeled_sentences(sentences, labels, phase):
    return [LabeledSentence(list(normalize_text(sent, lowercase=True)),
                            [make_label(i, label, phase)])
            for i, (label, sent) in enumerate(zip(labels, sentences))]

# convert data to tokenized Labeled Sentences:
train_sents = labeled_sentences(X_train, y_train, "train")
test_sents = labeled_sentences(X_test, y_test, "test")

# train a Doc2Vec model with dm and cbow
if not os.path.exists("/vol/tensusers/fkarsdorp/model_dm.bin"):
    model_dm = Doc2Vec(CorpusLoader(train_sents, test_sents), size=SIZE,
                       window=8, min_count=MINCOUNT, sample=SAMPLE, workers=8,
                       dm=1, iter=5)
    model_dm.save("/vol/tensusers/fkarsdorp/model_dm.bin")
else:
    model_dm = Doc2Vec.load("/vol/tensusers/fkarsdorp/model_dm.bin")
model_dm.init_sims(replace=True)

if not os.path.exists("/vol/tensusers/fkarsdorp/model_dbow.bin"):
    model_dbow = Doc2Vec(CorpusLoader(train_sents, test_sents), size=SIZE,
                         window=8, min_count=MINCOUNT, sample=SAMPLE, workers=8,
                         dm=0, iter=5)
    model_dbow.save("/vol/tensusers/fkarsdorp/model_dbow.bin")
else:
    model_dbow = Doc2Vec.load("/vol/tensusers/fkarsdorp/model_dbow.bin")
model_dbow.init_sims(replace=True)


# combine dbow and dm models
VecTuple = namedtuple("VecTuple", ["label", "vec"])

def combine_models(phase, *models):
    for key in models[0].vocab.iterkeys():
        if phase in key and key.startswith(('T', '0')):
            yield VecTuple(key, np.concatenate(
                [model.syn0[model.vocab[key].index] for model in models]))

train_tuples = list(combine_models("train", model_dm, model_dbow))
test_tuples = list(combine_models("test", model_dm, model_dbow))

# next we will create a BOW model:
vectorizer = TfidfVectorizer(ngram_range=(1, 1), use_idf=True,
                             analyzer='word', min_df=1, norm='l2')
vectorizer.fit(X_train + X_test)
train_bow = vectorizer.transform(X_train)
test_bow = vectorizer.transform(X_test)
train_bow_indices = {sent.labels[0]: i for i, sent in enumerate(train_sents)}
test_bow_indices = {sent.labels[0]: i for i, sent in enumerate(test_sents)}

def stack_vectors(tuples, bow, bow_indices, stack=True):
    dvectors, svectors, targets = [], [], []
    for key, vector in sorted(tuples):
        dvectors.append(vector)
        svectors.append(bow[bow_indices[key]])
        targets.append(1 if key.startswith("T") else 0)
    dvectors, svectors = np.array(dvectors), sp.vstack(svectors)
    if not stack:
        return (dvectors, svectors), targets
    vectors = sp.hstack([dvectors, svectors], format='csr')
    targets = np.array(targets)
    vectors.asfptype()
    return vectors, targets

train_vec, train_t = stack_vectors(train_tuples, train_bow, train_bow_indices)
test_vec, test_t = stack_vectors(test_tuples, test_bow, test_bow_indices)

# next we'll train a number of classifiers
classifier = LinearSVC
# this is a classifier with all features combined
clf = classifier(C=50, class_weight='auto')
clf.fit(train_vec, train_t)
preds = clf.predict(test_vec)
print classification_report(test_t, preds)

# this is a classifier with only bag of word features
clf = classifier(C=50, class_weight='auto')
(_, train_bow), train_t = stack_vectors(
    train_tuples, train_bow, train_bow_indices, stack=False)
(_, test_bow), test_t = stack_vectors(
    test_tuples, test_bow, test_bow_indices, stack=False)
clf.fit(train_bow, train_t)
preds = clf.predict(test_bow)
print classification_report(np.array([1 if l == 1 else 0 for l in test_t]), preds)

# this is a classifier with only the word2vec feature vectors
train_y, train_vec = zip(*train_tuples)
test_y, test_vec = zip(*test_tuples)
train_y = np.array([1 if l.startswith('T') else 0 for l in train_y])
test_y = np.array([1 if l.startswith('T') else 0 for l in test_y])
clf =  classifier(C=10, class_weight='auto')
clf.fit(train_vec, train_y)
preds = clf.predict(test_vec)
print classification_report(test_y, preds)
