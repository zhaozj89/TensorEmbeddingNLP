import scipy.io
import joblib
import argparse
import matlab.engine
import os

from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--option", help="use which dataset", default='16000oneliners')
parser.add_argument("--win_size", help="window size", type=int, default=5)
parser.add_argument("--cp_rank", help="rank", type=int, default=10)
args = parser.parse_args()

CONFIG = GetConfig(args.option)

# load data #

[word2idx, vocabulary, X, y, X_train, X_test, y_train, y_test, inds_train, inds_test, inds_all] = \
    joblib.load(CONFIG['RAW_DATA'])


# build doc tensor #

vocab_size = len(vocabulary)
doc_size = X.shape[0]
sentence_size = X.shape[1]

if args.win_size==-1 or args.win_size>sentence_size:
    args.win_size = sentence_size

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

# aspect_idx = word2idx['$t$']

coord_list = []
val_list = []

for k in range(doc_size):
    word_word_dict = TwoDimDict()

    # by Andrew, April 04, 2019
    for i in range(1, sentence_size):
        if X[k][i]==0:
            break
        for j in range(1, args.win_size+1):
            left_win_idx = i-j
            right_win_idx = i+j

            for win_idx in [left_win_idx, right_win_idx]: #check the window both to left and to right
                if (win_idx >= 0) and (win_idx < sentence_size) and (X[k][win_idx] != 0):
                    word_word_dict.add(X[k][i], X[k][win_idx], 1)

    for item in word_word_dict.get_item():
        coord_list.append((item[0], item[1], k))
        val_list.append(item[2])

# matlab tensor cp #

scipy.io.savemat('tmp_tensor_info.mat',
                 dict(coord_list=coord_list, val_list=val_list,
                      vocab_size=vocab_size+2, doc_size=doc_size))

eng = matlab.engine.start_matlab()
eng.TensorDecomposition(args.cp_rank, nargout=0)
eng.quit()
doc2vec = scipy.io.loadmat('tmp_doc2vec_mat.mat')

os.remove("tmp_tensor_info.mat")
os.remove("tmp_doc2vec_mat.mat")

# save #

joblib.dump(doc2vec['doc2vec'], CONFIG['TENSOR_EMBEDDING'])