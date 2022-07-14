# dependency-pytorch
This is pytorch implementation of paper APPLYING CROSS-VIEW TRAINING FOR DEPENDENCY PARSING IN VIETNAMESE, 2022

hehe

## package
- torch (https://pytorch.org/)
- matplotlib
- transformers
- graphviz
- nltk

## Usage
- set options in file config/default_config.py
- run python main.py

### Some notable config:

- mode: train, evaluate or annotate
- cross_view: True or False (train supervised or semi-supervised)
- file location: place to define file name
- word level, sentence level, dropout, encoder: hyper-parameters for the parser
- train: set number of batch, batch size, printing