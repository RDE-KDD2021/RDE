# RDE
### Requirements

#### Packages

- python == 3.6.12
- numpy == 1.16.2
- pytorch == 1.6.0
- scipy == 1.5.2
- tqdm == 4.54.0
- logzero == 1.6.2
- scikit-learn == 0.23.2
- pyxclib (available in https://github.com/kunaldahiya/pyxclib)

#### Hardware requirements

4 GPUs with 32G GPU RAM

### Data Preparation

Please put data in the same location as below

```
data
├── eurlex
│   ├── trn_X_Xf.txt
│   ├── trn_X_Y.txt
│   ├── val_X_Xf.txt
│   ├── val_X_Y.txt
│   ├── tst_X_Xf.txt
│   ├── tst_X_Y.txt
│   ├── parabel
│   │   ├── trn_score_mat.txt
│   │   ├── val_score_mat.txt
│   │   └── tst_score_mat.txt
│   ├── fastxml
│   │   └── ...
│   └── pfastrexml
│       └── ...
├── wiki10
│   └── ...
└── amazon670k
    └── ...
```



### Train and Test

Train and test as follows:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py -d eurlex -b parabel --train --test
```
