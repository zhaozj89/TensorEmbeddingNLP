import nltk
import collections
import os
import pickle
import sys
import numpy as np

def Idx2Word(doc_idx, vocabulary):
    words = []
    for idx in doc_idx:
        if idx == 0:
            break
        words.append(vocabulary[idx])
    return words

def Path2Sentence(file_path):
    file = open(file_path)
    sentences = file.read().split('\n')
    sentences = sentences[0:-1]

    return sentences

def BuildVocabulary(X):
    max_sentence_len = 0
    word_frequency = collections.Counter()
    for line in X:
        words = nltk.word_tokenize(line.lower())

        if len(words) > max_sentence_len:
            max_sentence_len = len(words)

        for word in words:
            word_frequency[word] += 1
    return word_frequency, max_sentence_len


def Sentence2Index(X, word2index):
    Xout = []

    for line in X:
        words = nltk.word_tokenize(line.lower())
        sequence = []

        for word in words:
            if word in word2index:
                sequence.append(word2index[word])
            else:
                sequence.append(word2index["<UNK>"])

        Xout.append(sequence)

    return Xout

def LoadPretrainedEmbeddings(file_path):
    pretrained_fpath_saved = os.path.expanduser(file_path.format(sys.version_info.major))

    with open(pretrained_fpath_saved, 'rb') as f:
        embedding_weights = pickle.load(f)
    f.close()

    out = np.array(list(embedding_weights.values()))
    print('embedding_weights shape:', out.shape)
    return out
