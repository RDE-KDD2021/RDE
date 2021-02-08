import os
import sys
import warnings
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import vstack
from tqdm import tqdm
from logzero import logger
from xclib.data import data_utils

from modules import XMLDataset, MLLinear
from models import Model
from configs import get_configs
from evaluation import *


def test_baseline(test_loader, inv_w, mlb, top=5):
    preds = []
    for _, _, sc_col, _ in tqdm(test_loader, desc='Testing baseline', leave=False):
        preds.append(sc_col[:, :top])
    test_preds = np.concatenate(preds)
    test_labels = test_loader.dataset.labels
    
    p, n, psp, psn = [], [], [], []
    for k in (1, 3, 5):
        p.append(get_precision(test_preds, test_labels, mlb, top=k))
        n.append(get_ndcg(test_preds, test_labels, mlb, top=k))
        psp.append(get_psp(test_preds, test_labels, inv_w, mlb, top=k))
        psn.append(get_psndcg(test_preds, test_labels, inv_w, mlb, top=k))
    
    logger.info('P@1,3,5: %.2f, %.2f, %.2f' % tuple(p))
    logger.info('nDCG@1,3,5: %.2f, %.2f, %.2f' % tuple(n))
    logger.info('PSP@1,3,5: %.2f, %.2f, %.2f' % tuple(psp))
    logger.info('PSnDCG@1,3,5: %.2f, %.2f, %.2f' % tuple(psn))

    
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    
    parser = argparse.ArgumentParser('xml-postprocess')
    parser.add_argument('--dataset', '-d', type=str, default='eurlex')
    parser.add_argument('--baseline', '-b', type=str, default='parabel')
    parser.add_argument('--test-baseline', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--test-step', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    cfg = get_configs(args.dataset, args.baseline)
    
    if args.test_baseline or args.train or args.test:
        train_labels = data_utils.read_sparse_file(cfg['trn_lbl_file'], dtype='int32', force_header=True)
        valid_labels = data_utils.read_sparse_file(cfg['val_lbl_file'], dtype='int32', force_header=True)
        inv_w = get_inv_propensity(vstack((train_labels, valid_labels)), cfg['a'], cfg['b'])
        mlb = MultiLabelBinarizer(range(train_labels.shape[1]), sparse_output=True)
        mlb.fit(None)
        
    if args.test_baseline or args.test_step or args.test:
        test_features = data_utils.read_sparse_file(cfg['tst_ft_file'], force_header=True)
        test_scores = data_utils.read_sparse_file(cfg['tst_score_file'], force_header=True)
        test_labels = data_utils.read_sparse_file(cfg['tst_lbl_file'], dtype='int32', force_header=True)
        test_loader = DataLoader(XMLDataset(test_features, test_scores, test_labels, training=False),
                                 batch_size=cfg['test']['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=True)
    else:
        test_loader = None
        
    if args.test_baseline:
        test_baseline(test_loader, inv_w, mlb)
        
    if args.train or args.test:
        model = Model(network=MLLinear, **cfg['data'], **cfg['model'])
    
    if args.train:
        train_features = data_utils.read_sparse_file(cfg['trn_ft_file'], force_header=True)
        train_scores = data_utils.read_sparse_file(cfg['trn_score_file'], force_header=True)
        valid_features = data_utils.read_sparse_file(cfg['val_ft_file'], force_header=True)
        valid_scores = data_utils.read_sparse_file(cfg['val_score_file'], force_header=True)
        train_loader = DataLoader(XMLDataset(train_features, train_scores, train_labels, training=True),
                                  batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=True)
        valid_loader = DataLoader(XMLDataset(valid_features, valid_scores, valid_labels, training=False),
                                  batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=True)
        label_freq = np.asarray(train_labels.sum(axis=0)).squeeze()
        
        if args.test_step:
            model.add_test_step(test_loader)
        if args.reload:
            model.load_model(cfg['model_dir'])
        model.train(train_loader, valid_loader, label_freq, inv_w, mlb, cfg['model_dir'], **cfg['data'], **cfg['train'])
        
    if args.test:
        model.test(test_loader, inv_w, mlb, cfg['model_dir'], **cfg['data'], **cfg['train'])
    
    