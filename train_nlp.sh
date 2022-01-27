#!/bin/bash

python train_nlp.py --model plain --dataset 20ng
python train_nlp.py --model de --dataset 20ng
python train_nlp.py --model oe --dataset 20ng
python train_nlp.py --model noneclass --dataset 20ng
python train_nlp.py --model dirlik --dataset 20ng
python train_nlp.py --model mixed --dataset 20ng
python train_nlp.py --model vb --dataset 20ng
python train_nlp.py --model vbood --dataset 20ng


python train_nlp.py --model plain --dataset sst
python train_nlp.py --model de --dataset sst
python train_nlp.py --model oe --dataset sst
python train_nlp.py --model noneclass --dataset sst
python train_nlp.py --model dirlik --dataset sst
python train_nlp.py --model mixed --dataset sst
python train_nlp.py --model vb --dataset sst
python train_nlp.py --model vbood --dataset sst


python train_nlp.py --model plain --dataset trec
python train_nlp.py --model de --dataset trec
python train_nlp.py --model oe --dataset trec
python train_nlp.py --model noneclass --dataset trec
python train_nlp.py --model dirlik --dataset trec
python train_nlp.py --model mixed --dataset trec
python train_nlp.py --model vb --dataset trec
python train_nlp.py --model vbood --dataset trec
