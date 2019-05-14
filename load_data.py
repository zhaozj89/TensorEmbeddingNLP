import numpy as np
import joblib
import os
import argparse

from config import *
from load_data_util import *
from load_data_Pun import *

def _Load16000Oneliners(ROOT_PATH):
    file_path_pos = os.path.join(ROOT_PATH, 'Jokes16000-utf8.txt')
    file_path_neg = os.path.join(ROOT_PATH, 'MIX16000-utf8.txt')

    X1 = Path2Sentence(file_path_pos)
    X2 = Path2Sentence(file_path_neg)
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

def LoadData(ROOT_PATH, option):
    if option == "16000oneliners":
        return _Load16000Oneliners(ROOT_PATH)
    if option == "Pun":
        return LoadPun(ROOT_PATH)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", help="use which dataset", default='16000oneliners')
    parser.add_argument('--label_portion', help='proportion of labels', type=float, default=0.1)
    parser.add_argument("--seed", help="reproducible experiment with seeds", type=int, default=666)
    args = parser.parse_args()

    CONFIG = GetConfig(args.option)

    if os.path.isfile(CONFIG['SAVED_RAW_DATA']):
        [X, y, vocabulary] = joblib.load(CONFIG['SAVED_RAW_DATA'])
    else:
        X, y, vocabulary = LoadData(CONFIG['ROOT_PATH'], args.option)
        joblib.dump([X, y, vocabulary], CONFIG['SAVED_RAW_DATA'])

    y = y.astype(np.float)

    N = len(X)
    random_generator = np.random.RandomState(args.seed)
    inds_all = random_generator.permutation(N)

    X = X[inds_all]
    y = y[inds_all]

    inds_all = np.array(range(N))
    train_num = int(args.label_portion * N)
    inds_train = inds_all[:train_num]
    inds_test = inds_all[train_num:]

    X_train = X[inds_train]
    y_train = y[inds_train]
    X_test = X[inds_test]
    y_test = y[inds_test]

    if not os.path.exists(os.path.join(CONFIG['ROOT_PATH'], 'data')):
        os.makedirs(os.path.join(CONFIG['ROOT_PATH'], 'data'))
    joblib.dump([0, vocabulary, X, y, X_train, X_test, y_train, y_test, inds_train, inds_test, inds_all], \
                os.path.join(CONFIG['ROOT_PATH'], 'data/raw.pkl'))