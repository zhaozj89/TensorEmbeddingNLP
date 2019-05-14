import scipy.io
import numpy as np
import collections
import matlab.engine
import gensim
import argparse
import joblib

import codecs
import os
from nltk.tokenize import word_tokenize
from string import punctuation
import re
from scipy.spatial.distance import cdist

def BuildVocabulary(documents):
    max_sentence_len = 0
    word_frequency = collections.Counter()
    for words in documents:
        if len(words) > max_sentence_len:
            max_sentence_len = len(words)

        for word in words:
            word_frequency[word] += 1

    max_vocab = len(word_frequency)
    word2index = {x[0]: i + 2 for i, x in enumerate(word_frequency.most_common(max_vocab))}
    word2index["<PAD>"] = 0
    word2index["<UNK>"] = 1


    return word2index, max_sentence_len

def Sentence2Index(documents, word2index):
    X = []

    for words in documents:
        sequence = []

        for word in words:
            if word in word2index:
                sequence.append(word2index[word])
            else:
                sequence.append(word2index["<UNK>"])

        X.append(sequence)

    return X

class TwoDimDict():
    def __init__(self):
        self.data = {}
    def add(self, i, j, val):
        if i in self.data and j in self.data[i]:
            self.data[i][j] += val
        else:
            if i in self.data:
                self.data[i][j] = val
            else:
                self.data[i] = {}
                self.data[i][j] = val
    def get_item(self):
        for i, i_val in self.data.items():
            for j, j_val in self.data[i].items():
                yield (i, j, j_val)

def decompose_tensors(documents, win_size=5, cp_rank=150):
    """
    Method for computing tensor decompositions based on https://www.cs.ucr.edu/~epapalex/papers/asonam18-fakenews.pdf
    
    :param documents: the documents to be decomposed, tokenized
    :type documents: Iterable[Iterable[str]]
    :param win_size: the window size to use for word cooccurance. Will look win_size to the left and win_size to the right of each token
    :type win_size: int
    :param cp_rank:
    :type cp_rank: int
    """

    vocabulary, sentence_len = BuildVocabulary(documents)
    doc_size = len(documents)
    vocab_size = len(vocabulary)

    X = Sentence2Index(documents, vocabulary)

    coord_list = []
    val_list = []
    for k in range(doc_size):
        word_word_dict = TwoDimDict()

        # by Andrew, April 04, 2019
        # print('build word_word_doc tensor, {:d}/{:d} ...'.format(k, doc_size))
        for i in range(1, len(documents[k])):
            if X[k][i] == 0:
                break
            for j in range(1, win_size + 1):
                left_win_idx = i - j
                right_win_idx = i + j

                for win_idx in [left_win_idx, right_win_idx]:  # check the window both to left and to right
                    if (win_idx >= 0) and (win_idx < len(X[k])) and (X[k][win_idx] != 0):
                        word_word_dict.add(X[k][i], X[k][win_idx], 1.0)

        for item in word_word_dict.get_item():
            coord_list.append((item[0], item[1], k))
            val_list.append(item[2])

    scipy.io.savemat('tmp_tensor_info.mat',
                     dict(coord_list=coord_list, val_list=val_list, vocab_size=vocab_size, doc_size=doc_size))

    eng = matlab.engine.start_matlab()
    eng.TensorDecomposition(cp_rank, nargout=0)
    eng.quit()
    doc2vec = scipy.io.loadmat('tmp_doc2vec_mat.mat')

    os.remove("tmp_tensor_info.mat")
    os.remove("tmp_doc2vec_mat.mat")

    return doc2vec['doc2vec'], vocabulary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", help="root path", default='./data/SemEval/')
    parser.add_argument("--method", help="method", default='tensor')
    args = parser.parse_args()
    
    punc = re.compile(f"[{punctuation}]+")
    midnight = re.compile("@midnight", flags=re.I)
    
    semeval_dir = args.root_path
    eveluation_dir = os.path.join(semeval_dir, "evaluation_dir/evaluation_data")
     
    prediction_dir = os.path.join(semeval_dir, "predictions")
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    for f in os.listdir(eveluation_dir):
        if not f.endswith(".tsv"):
            continue

        filename = os.path.join(eveluation_dir, f)

        print(f)
        hashtag = re.compile(f"#{f[:-4].replace('_','')}", flags=re.I)

        labels = []
        texts = []
        ids = []
        with codecs.open(filename, "r", encoding="utf-8") as file:
            for line in file:
                line = line.split("\t")
                ids.append(line[0])
                texts.append(midnight.sub("", hashtag.sub("", line[1])))
                labels.append(int(line[2]))

        texts = [[word for word in word_tokenize(text) if not punc.fullmatch(word)] for text in texts]

        if args.method == 'word2vec':
            model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
            decomp = np.zeros((len(texts), 300))
            for i, text in enumerate(texts):
                counter = 0
                for word in text:
                    if word in model.vocab:
                        decomp[i] += model.word_vec(word)
                        counter+=1
                decomp[i] /= counter
        if args.method == 'tensor':
            decomp, vocab_map = decompose_tensors(texts)

        center = np.mean(decomp, axis=0)
        distances=cdist(decomp, [center])

        ranked=sorted(list(zip(ids, labels, distances)), key= lambda x : x[2]) #sort from most central to least

        global_predictions = list(zip(*ranked))[0]

        with open(os.path.join(prediction_dir, 'taska', f"{f[:-4]}_PREDICT.tsv"), "w") as of:
            for i, id1 in enumerate(global_predictions[:-1]):
                for id2 in global_predictions[i+1:]:
                    of.write("{}\t{}\t1\n".format(str(id1), str(id2)))

        with open(os.path.join(prediction_dir, 'taskb', f"{f[:-4]}_PREDICT.tsv"), "w") as of:
            of.write("\n".join(global_predictions))
    
    
            
