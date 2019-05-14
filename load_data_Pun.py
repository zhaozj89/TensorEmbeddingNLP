import csv
import os
import numpy as np

from load_data_util import *

def LoadPun(ROOT_PATH):
    X1 = []
    with open(os.path.join(ROOT_PATH, 'puns_of_day.csv'), 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            X1.append(row[1])
    file.close()

    X21 = Path2Sentence(os.path.join(ROOT_PATH, 'proverbs.txt'))
    X22 = Path2Sentence(os.path.join(ROOT_PATH, 'new_select.txt'))
    X2 = X21+X22

    y1 = [1]*len(X1)
    y2 = [0]*len(X2)

    X = X1 + X2
    y = y1 + y2

    word_frequency, max_sentence_len = BuildVocabulary(X)

    max_vocab = len(word_frequency)
    word2index = {x[0]: i + 2 for i, x in enumerate(word_frequency.most_common(max_vocab))}
    word2index["<PAD>"] = 0
    word2index["<UNK>"] = 1
    index2word = {v: k for k, v in word2index.items()}

    X = Sentence2Index(X, word2index)

    vocabulary = [index2word[i] for i in range(max_vocab + 2)]

    Xlist = []
    for x in X:
        if len(x) <= max_sentence_len:
            Xlist.append(list(x) + [0] * (max_sentence_len - len(x)))
        else:
            raise ValueError

    X = np.array(Xlist)
    y = np.array(y)
    vocabulary = np.array(vocabulary)

    return X, y, vocabulary



