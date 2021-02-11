import sys
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from xclib.data import data_utils
from xclib.evaluation import xc_metrics

dataset = sys.argv[1]

# Read file with features and labels
train_features, train_labels, num_samples, num_features, num_labels = data_utils.read_data('data/' + dataset + '/' + 'train.txt')
test_features, test_labels, _, _, _ = data_utils.read_data('data/' + dataset + '/' + 'test.txt')

if test_features.shape[1] != train_features.shape[1]:
    test_features = csr_matrix(test_features, shape=(test_features.shape[0], train_features.shape[1]))

train_labels = train_labels.astype('int32')
test_labels = test_labels.astype('int32')

if dataset.startswith('wikipedia'):
    a, b = 0.5, 0.4
if dataset.startswith('amazon'):
    a, b = 0.6, 2.6
else:
    a, b = 0.55, 1.5
    
if dataset.startswith('amazon'):
    valid_size = 4000
else:
    valid_size = 200

inv_propen = xc_metrics.compute_inv_propesity(train_labels, a, b)
np.savetxt('data/' + dataset + '/' + 'inv_prop.txt', inv_propen)

data_utils.write_sparse_file(train_labels, 'data/' + dataset + '/' + 'raw_trn_X_Y.txt')
train_features, valid_features, train_labels, valid_labels = train_test_split(train_features, train_labels, test_size=valid_size, random_state=1240)
data_utils.write_sparse_file(train_features, 'data/' + dataset + '/' + 'trn_X_Xf.txt')
data_utils.write_sparse_file(train_labels, 'data/' + dataset + '/' + 'trn_X_Y.txt')
data_utils.write_sparse_file(valid_features, 'data/' + dataset + '/' + 'val_X_Xf.txt')
data_utils.write_sparse_file(valid_labels, 'data/' + dataset + '/' + 'val_X_Y.txt')
data_utils.write_sparse_file(test_features, 'data/' + dataset + '/' + 'tst_X_Xf.txt')
data_utils.write_sparse_file(test_labels, 'data/' + dataset + '/' + 'tst_X_Y.txt')

