import os
import codecs

from collections import defaultdict
from itertools import groupby

from .utils import *

PATH = os.path.dirname(__file__)

izip = zip

def merge_by_class(documents, filter=None):
    """Document should be an iterable containing pairs of labels and
    documents. This function merges all documents that contain the same
    label into a big document of which the label will become the class
    label. Each document should be a list or tuple of words (or some other
    object).
    """
    merged_documents = defaultdict(list)
    for (labels, document) in documents:
        for label in labels:
            if label != filter:
                merged_documents[label].append(document)
    return [(label, ' '.join(docs)) for label, docs in merged_documents.iteritems()]

def _merge_per_document(data, target, idnumbers, storytypes, concatenate=True):
    X, y, idnumbs, stypes = [], [], [], []
    for i, (_, story_id) in enumerate(groupby(izip(idnumbers, storytypes, target, data),
                                              key=first)):
        ids, storytype, motifs, sentences = unzip(story_id)
        y.append(tuple(motifs) if not concatenate else sum(set(motifs), ()))
        X.append(sentences if not concatenate else ' '.join(sentences))
        idnumbs.append(ids[0])
        stypes.append(storytype[0])
    return Struct(data=X, target=y, idnumbers=idnumbs, storytypes=stypes)

def load_motif_dataset(keep_prefix=True, per_document=False, keep_sentences=False, binary=False,
                       positive=1, negative=-1, encode_labels=True, filter=None):
    # somehow the database contains some zero byte elements. This will mess up all tokenization
    # and splitting operations below. We remove them here.
    remove_zero_bytes(os.path.join(PATH, 'data/annotated_dataset.txt'), 'utf-8')
    lines = [line.strip().split('|', 3) for line in codecs.open(
        os.path.join(PATH, 'data/annotated_dataset.txt'), encoding='utf-8')]
    motif_pos = 0 if keep_prefix else 2
    data, target, idnumbers, storytypes = [], [], [], []
    for _, storylines in groupby(lines, key=second):
        storytype, ids, motifs, sentences = unzip(storylines)
        data.extend(sentences)
        target.extend([tuple([motif[motif_pos:] if motif != 'O' else motif
                                                for motif in motifstring.split('/') if motif !=filter])
                       for motifstring in motifs])
        idnumbers.extend(ids)
        storytypes.extend(storytype)
    if per_document:
        return _merge_per_document(data, target, idnumbers, storytypes, not keep_sentences)
    return Struct(data=data, target=target, idnumbers=idnumbers, storytypes=storytypes)

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

__all__ = ['merge_by_class', 'load_motif_dataset', 'load_transformation_dataset', '_merge_per_document']
