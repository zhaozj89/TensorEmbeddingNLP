from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import argparse
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from config import *
from load_data_util import *

def MyLabelSpreading(option, neighbor):

    CONFIG = GetConfig(option)

    [word2idx, vocabulary, X, y, X_train, X_test, y_train, y_test, inds_train, inds_test, inds_all] = \
        joblib.load(CONFIG['RAW_DATA'])
    doc2vec = joblib.load(CONFIG['TENSOR_EMBEDDING'])

    # propagation

    classes = np.unique(y)

    n_samples = y.shape[0]
    n_classes = classes.shape[0]

    labels = np.zeros((n_samples, n_classes))

    for i, val in enumerate(y_train):
        labels[i][int(val)] = 1.0

    step = y_train.shape[0]
    for i, val in enumerate(y_test):
        labels[i+step] = -1

    label_prop_model = LabelSpreading(kernel='knn', n_neighbors=neighbor)
    # label_prop_model = LabelSpreading(kernel='rbf', n_neighbors=args.neighbor,\
    #                                   gamma=20, alpha=0.2, max_iter=30, tol=0.001)
    label_prop_model.fit(doc2vec, labels)


    pred_probability = label_prop_model.predict_proba(doc2vec)
    pred_class = classes[np.argmax(pred_probability, axis=1)].ravel()

    accuracy = accuracy_score(y_test, pred_class[inds_test])
    prf = precision_recall_fscore_support(y_test, pred_class[inds_test], average='binary')

    print('Accuracy:%f' % accuracy)
    print('Precision:%f' % prf[0])
    print('Recall:%f' % prf[1])
    print('Fscore:%f' % prf[2])

    return accuracy, prf[0], prf[1], prf[2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", help="use which dataset", default='16000oneliners')
    parser.add_argument("--neighbor", help="number of neighbors", type=int, default=50)
    args = parser.parse_args()

    MyLabelSpreading(args.option, args.neighbor)