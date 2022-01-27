#!/bin/bash

python eval_nlp.py --compute_hessian --dataset sst
python eval_nlp.py --compute_hessian --dataset 20ng
python eval_nlp.py --compute_hessian --dataset trec

python eval_nlp.py --repeat 5 --dataset sst
python eval_nlp.py --repeat 5 --dataset 20ng
python eval_nlp.py --repeat 5 --dataset trec
