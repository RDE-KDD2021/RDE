
def get_configs(dataset, baseline):
    
    if dataset.startswith('wikipedia'):
        a, b = 0.5, 0.4
    if dataset.startswith('amazon'):
        a, b = 0.6, 2.6
    else:
        a, b = 0.55, 1.5
    
#--------------------------eurlex---------------------------
    
    if dataset == 'eurlex':
        data = {
            'feature_size': 5000,
            'label_size'  : 3993,
        }
        model = {
            'num_experts' : 3,
            'hidden_size' : (1024,),
            'input_mode'  : 'feature',
            'output_mode' : 'residual',
            'use_norm'    : True,
            'drop_prob'   : 0.7,
        }
        train = {
            'batch_size': 40,
            'num_epochs': 200,
            'lr'        : 0.01,
            'cost_mode' : 'log',
            'temp_mode' : 'log',
            'clf_flood' : 1e-5,
            'div_flood' : -5e-4,
            'div_factor': 0.3,
        }
        valid = {
            'batch_size': 100,
        }
        test = {
            'batch_size': 100,
        }
        
#--------------------------wiki10---------------------------

    elif dataset == 'wiki10':
        data = {
            'feature_size': 101938,
            'label_size'  : 30938,
        }
        model = {
            'num_experts' : 3,
            'hidden_size' : (1024,),
            'input_mode'  : 'feature',
            'output_mode' : 'residual',
            'use_norm'    : True,
            'drop_prob'   : 0.7,
        }
        train = {
            'batch_size': 40,
            'num_epochs': 50,
            'lr'        : 0.01,
            'cost_mode' : 'log',
            'temp_mode' : 'log',
            'clf_flood' : 1e-4,
            'div_flood' : -1e-5,
            'div_factor': 0.5,
        }
        valid = {
            'batch_size': 100,
        }
        test = {
            'batch_size': 100,
        }
        
#------------------------amazon670k-------------------------
        
    elif dataset == 'amazon670k':
        data = {
            'feature_size': 135909,
            'label_size'  : 670091,
        }
        model = {
            'num_experts' : 3,
            'hidden_size' : (1024,),
            'input_mode'  : 'feature',
            'output_mode' : 'residual',
            'use_norm'    : True,
            'drop_prob'   : 0.7,
        }
        train = {
            'batch_size': 100,
            'num_epochs': 30,
            'lr'        : 0.01,
            'cost_mode' : 'log',
            'temp_mode' : 'log',
            'clf_flood' : 0,
            'div_flood' : -1e-5,
            'div_factor': 0.5,
        }
        valid = {
            'batch_size': 100,
        }
        test = {
            'batch_size': 100,
        }
        
#-----------------------------------------------------------
    
    cfg = {
        'trn_ft_file'   : f'data/{dataset}/trn_X_Xf.txt',
        'trn_lbl_file'  : f'data/{dataset}/trn_X_Y.txt',
        'val_ft_file'   : f'data/{dataset}/val_X_Xf.txt',
        'val_lbl_file'  : f'data/{dataset}/val_X_Y.txt',
        'tst_ft_file'   : f'data/{dataset}/tst_X_Xf.txt',
        'tst_lbl_file'  : f'data/{dataset}/tst_X_Y.txt',
        'trn_score_file': f'data/{dataset}/{baseline}/trn_score_mat.txt',
        'val_score_file': f'data/{dataset}/{baseline}/val_score_mat.txt',
        'tst_score_file': f'data/{dataset}/{baseline}/tst_score_mat.txt',

        'model_dir'     : f'models/{dataset}/{baseline}',

        'num_workers': 8,

        'a': a,
        'b': b,
        
        'data' : data,
        'model': model,
        'train': train,
        'valid': valid,
        'test' : test,
    }
    
    return cfg
