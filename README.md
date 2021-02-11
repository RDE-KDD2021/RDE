# RDE
## Requirements

### Packages

- python == 3.6.12
- pytorch == 1.6.0
- scikit-learn == 0.23.2
- tqdm == 4.56.0
- logzero == 1.6.3
- numpy == 1.19.2
- scipy == 1.5.2
- pyxclib (available in https://github.com/kunaldahiya/pyxclib)

### Hardware requirements

4 GPUs with 32G GPU RAM

## Data Preparation

Please put data in the same location as below

```
data
├── eurlex
│   ├── train.txt
│   ├── test.txt
├── wiki10
│   ├── train.txt
│   ├── test.txt
└── amazon670k
    ├── train.txt
    └── test.txt
```



## Train and Test

Split train and test data:

```bash
python split_data.py eurlex
python split_data.py wiki10
python split_data.py amazon670k
```

Makefile for baseline methods:

```bash
make -C baseline/parabel/
make -C baseline/fastxml/
make -C baseline/pfastrexml/
```

Run baseline methods:

```bash
sh baseline/run_parabel.sh eurlex
sh baseline/run_fastxml.sh eurlex
sh baseline/run_pfastrexml.sh eurlex

sh baseline/run_parabel.sh wiki10
sh baseline/run_fastxml.sh wiki10
sh baseline/run_pfastrexml.sh wiki10

sh baseline/run_parabel.sh amazon670k
sh baseline/run_fastxml.sh amazon670k
sh baseline/run_pfastrexml.sh amazon670k
```

Train and test as follows:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python main.py -d eurlex -b parabel --train --test-step
python main.py -d eurlex -b fastxml --train --test-step
python main.py -d eurlex -b pfastrexml --train --test-step

python main.py -d wiki10 -b parabel --train --test-step
python main.py -d wiki10 -b fastxml --train --test-step
python main.py -d wiki10 -b pfastrexml --train --test-step

python main.py -d amazon670k -b parabel --train --test-step
python main.py -d amazon670k -b fastxml --train --test-step
python main.py -d amazon670k -b pfastrexml --train --test-step
```
