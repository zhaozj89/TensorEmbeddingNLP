import os
import joblib
import numpy as np
import argparse

from label_propagation import MyLabelSpreading

parser = argparse.ArgumentParser()
parser.add_argument("--option", help="use which dataset", default='16000oneliners')
parser.add_argument('--label_portion', help='proportion of labels', type=float, default=0.1)
args = parser.parse_args()

option = args.option
label_portion = args.label_portion

if option is '16000oneliners':
    win_size = 5
    cp_rank = 10
    neighbor = 50
elif option is 'Pun':
    win_size = 5
    cp_rank = 10
    neighbor = 50


acc_list = []
pre_list = []
rec_list = []
f1_list = []
for counter, seed in enumerate([345, 543, 789, 987, 567, 765, 123, 321, 456, 654]):

    # load data
    os.system('python load_data.py --option {:s} --label_portion {:f} --seed {:d}' \
              .format(option, label_portion, seed))

    # doc2vec
    os.system('python doc2vec.py --option {:s} --win_size {:d} --cp_rank {:d}' \
              .format(option, win_size, cp_rank))

    acc, pre, rec, f1 = MyLabelSpreading(option, neighbor)
    acc_list.append(acc)
    pre_list.append(pre)
    rec_list.append(rec)
    f1_list.append(f1)

print('accuracy mean: %f' % np.mean(acc_list))
print('accuracy std: %f' % np.std(acc_list))

print('precision mean: %f' % np.mean(pre_list))
print('precision std: %f' % np.std(pre_list))

print('recall mean: %f' % np.mean(rec_list))
print('recall std: %f' % np.std(rec_list))

print('f1 mean: %f' % np.mean(f1_list))
print('f1 std: %f' % np.std(f1_list))

joblib.dump([acc_list, pre_list, rec_list, f1_list], option + str(label_portion) + '.pkl')


