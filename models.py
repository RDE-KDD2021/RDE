import os
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import csr_matrix, vstack, coo_matrix
from tqdm import tqdm
from logzero import logger

from evaluation import *


def arr2coo(cols, values, shape):
    rows = torch.LongTensor(np.arange(0, cols.shape[0]).repeat(cols.shape[1]))
    cols = cols.reshape(-1)
    indices = torch.stack((rows, cols), dim=0)
    values = values.reshape(-1)
    return torch.sparse.FloatTensor(indices, values, torch.Size(shape))

def fill_zeros(a):
    a = a.masked_fill_(a == 0, float('inf'))
    min_a = a.min(dim=1, keepdim=True)[0]
    a = a.where(a != float('inf'), min_a)
    return a
    

class Model:
    def __init__(self, network, num_experts, **kwargs):
        self.num_experts = num_experts
        self.devices, self.output_device = self.assign_devices()
        self.experts = nn.ModuleList( network(**kwargs).cuda(self.devices[i]) for i in range(self.num_experts) )
        self.test_step = False
        
    def assign_devices(self):
        visible_devices = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',') ))
        num_visible_devices = len(visible_devices)
        devices = [visible_devices[::-1][i % num_visible_devices] for i in range(self.num_experts)]
        output_device = visible_devices[0]
        return devices, output_device
        
    def replicate(self, item):
        return [item.cuda(device) for device in self.devices]
        
    def gather(self, items):
        return [item.cuda(self.output_device) for item in items]
    
    def train(self, train_loader, valid_loader, label_freq, inv_w, mlb, model_dir,
              feature_size, label_size, clf_flood, div_flood, div_factor,
              lr=0.01, cost_mode='log', temp_mode='log', num_epochs=100, step=100, top=5, **kwargs):
        
        self.optimizer = optim.Adam(self.experts.parameters(), lr)
        
        if cost_mode == 'none':
            cost = None
        elif cost_mode == 'log':
            cost = torch.Tensor( 1 / (1 + np.log(label_freq.clip(1))) ).cuda(self.output_device)
        elif cost_mode == 'loglog':
            cost = torch.Tensor( 1 / (1 + np.log(1 + np.log(label_freq.clip(1)))) ).cuda(self.output_device)
        
        self.clf_loss_fn = nn.BCELoss(weight=cost)
        self.div_loss_fn = nn.KLDivLoss(reduction='mean')
        
        if temp_mode == 'none':
            temperature = torch.ones( len(label_freq) )
        elif temp_mode == 'log':
            temperature = torch.Tensor( 1 + np.log(label_freq.clip(1)) )
        elif temp_mode == 'loglog':
            temperature = torch.Tensor( 1 + np.log(1 + np.log(label_freq.clip(1))) )
        temperature = temperature.cuda(self.output_device)
        
        global_step, avr_clf_loss, avr_div_loss, best_psn5 = 0, 0, 0, 0
        
        for epoch_idx in range(num_epochs):
            for batch_idx, (ft_col, ft_value, sc_col, sc_value, lbl_col, lbl_value) in enumerate(train_loader, 1):
                global_step += 1
                
                feature = arr2coo(ft_col, ft_value, (len(ft_col), feature_size))
                score = arr2coo(sc_col, sc_value, (len(sc_col), label_size))
                label = arr2coo(lbl_col, lbl_value, (len(lbl_col), label_size))
            
                self.experts.train()
                logits, mean_logit = self.forward(feature, score)

                probs = [torch.sigmoid(logits[i])                      for i in range(self.num_experts)]
                dists = [F.log_softmax(logits[i] / temperature, dim=1) for i in range(self.num_experts)]
                with torch.no_grad():
                    mean_dist = F.softmax(mean_logit / temperature, dim=1)
                
                label = label.cuda(self.output_device).to_dense()
                
                total_clf_loss, total_div_loss = 0, 0
                for i in range(self.num_experts):
                    clf_loss =  self.clf_loss_fn(probs[i], label)
                    div_loss = -self.div_loss_fn(dists[i], mean_dist)
                    clf_loss = (clf_loss - clf_flood).abs() + clf_flood
                    div_loss = (div_loss - div_flood).abs() + div_flood
                    total_clf_loss += clf_loss
                    total_div_loss += div_loss

                self.optimizer.zero_grad()
                loss = total_clf_loss + div_factor * total_div_loss
                loss.backward()
                self.optimizer.step()
                
                
                avr_clf_loss += total_clf_loss.item() / self.num_experts / step
                avr_div_loss += total_div_loss.item() / self.num_experts / step
                if global_step % step == 0:
                    p5, n5, psp5, psn5 = self.validate(valid_loader, inv_w, mlb, feature_size, label_size, top)
                    if psn5 > best_psn5:
                        best_psn5 = psn5
                        self.save_model(model_dir)
                        color = '36'
                    else:
                        color = '0'
                    logger.info('epoch %d %d | clf loss: %.2e  div loss: %.2e | p@5: %.2f  n@5: %.2f  psp@5: %.2f  psn@5: \033[%sm%.2f\033[0m'
                                % (epoch_idx, batch_idx * train_loader.batch_size, avr_clf_loss, avr_div_loss, p5, n5, psp5, color, psn5))
                    avr_clf_loss, avr_div_loss = 0, 0
                    
                    if self.test_step and color == '36':
                        self.test(self.test_loader, inv_w, mlb, model_dir, feature_size, label_size, **kwargs)
                
    def forward(self, feature, score):
        features, scores = self.replicate(feature), self.replicate(score)
        
        logits = []
        
        lock = threading.Lock()
        results = {}
        def _worker(i, device, expert, feature, score):
            try:
                with torch.cuda.device(device):
                    feature = feature.to_dense()
                    score = fill_zeros(score.to_dense())
                    logit = expert(feature, score)
                with lock:
                    results[i] = logit
            except Exception as e:
                with lock:
                    results[i] = e
        
        threads = [threading.Thread(target=_worker,
                                    args=(i, device, expert, feature, score))
                  for i, (device, expert, feature, score) in
                  enumerate(zip(self.devices, self.experts, features, scores))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        for i in range(self.num_experts):
            result = results[i]
            if isinstance(result, Exception):
                raise result
            logits.append(result)
            
        logits = self.gather(logits)
        mean_logit = sum(logits) / len(logits)
        return logits, mean_logit
        
    def predict(self, data_loader, feature_size, label_size, top):
        preds = []
        
        for ft_col, ft_value, sc_col, sc_value in data_loader:
            feature = arr2coo(ft_col, ft_value, (len(ft_col), feature_size))
            score = arr2coo(sc_col, sc_value, (len(sc_col), label_size))
            
            self.experts.eval()
            with torch.no_grad():
                _, mean_logit = self.forward(feature, score)
                prob = torch.sigmoid(mean_logit)
                _, pred = torch.topk(prob, k=top)
            preds.append(pred.cpu())
        return np.concatenate(preds)
    
    def validate(self, valid_loader, inv_w, mlb, feature_size, label_size, top):
        valid_preds = self.predict(valid_loader, feature_size, label_size, top)
        valid_labels = valid_loader.dataset.labels
        
        p5 = get_p_5(valid_preds, valid_labels, mlb)
        n5 = get_n_5(valid_preds, valid_labels, mlb)
        psp5 = get_psp_5(valid_preds, valid_labels, inv_w, mlb)
        psn5 = get_psndcg_5(valid_preds, valid_labels, inv_w, mlb)
        return p5, n5, psp5, psn5
    
    def test(self, test_loader, inv_w, mlb, model_dir, feature_size, label_size, top=5, **kwargs):
        self.load_model(model_dir)
        test_preds = self.predict(tqdm(test_loader, desc='Testing', leave=False), feature_size, label_size, top)
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
        
    def add_test_step(self, test_loader):
        self.test_step = True
        self.test_loader = test_loader
    
    def save_model(self, model_dir):
        if model_dir != None:
            os.makedirs(model_dir, exist_ok=True)
            for i in range(self.num_experts):
                torch.save(self.experts[i].state_dict(), os.path.join(model_dir, f'{i}.pth'))
                
    def load_model(self, model_dir):
        if model_dir != None:
            for i in range(self.num_experts):
                self.experts[i].load_state_dict(torch.load(os.path.join(model_dir, f'{i}.pth')))
    
