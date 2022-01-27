# Bayesian OOD Training

Code for the AISTATS 2022 paper titled "Being a Bit Frequentist Improves Bayesian Neural Networks" by Agustinus Kristiadi, Matthias Hein, and Philipp Hennig.

## Setting up:

1. Run: `conda create --name ENV_NAME --file conda_env.txt`.
2. Then: `conda activate ENV_NAME`.
3. Install PyTorch and TorchVision (<https://pytorch.org/get-started/locally/>).
4. Set the dataset path in `util/dataloaders.py`, line 30 (`path = os.path.expanduser('~/Datasets')`).
5. Follow the instruction here to obtain the NLP datasets: <https://github.com/hendrycks/outlier-exposure/tree/master/NLP_classification>.


## Reproducing the paper's results:

1. Model training: run `train.sh`, `train_nlp.sh`, and `train_aux.sh`.
2. Run `eval.sh` and `eval_nlp.sh` to gather experiments data.
3. Run `aggregate_*.py` to create the tables in the paper based on the previous data.
4. Run `plot_*.py` to create figures for dataset shift experiments.

## Citing the paper:

```
@inproceedings{kristiadi2021being,
  title={Being a Bit Frequentist Improves {B}ayesian Neural Networks},
  author={Kristiadi, Agustinus and Hein, Matthias and Hennig, Philipp},
  booktitle={AISTATS},
  year={2022}
}
```
