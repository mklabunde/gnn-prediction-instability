# On the Prediction Instability of Graph Neural Networks
Implementation and supplementary material of the paper "On the Prediction Instability of Graph Neural Networks".

[[arxiv]](https://arxiv.org/abs/2205.10070)


## Installation
Clone the repository and move into the root directory.

Then:
```shell
conda env create -f environment.yml
conda activate gnn-pred-stab
python -m pip install -e .
```

For plotting, latex is required. 
You may need to run:
```shell
apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super
pip install latex
```

Some parts require updated paths. 
Ctrl+F for `CHANGE` over the whole project to find them.

## Reproducing the results
### Training
We use [Hydra](https://hydra.cc/) to run the experiments.
The following command runs 50 times repeated training on GPU 0 over the listed datasets:

```shell
python scripts/layer_identification.py cuda=0 n_repeat=50 dataset=computers,photo,cs,physics,wikics model=pubmed_gat2017 -m
```

The `-m` flag allows you to specify multiple values for arguments, which are then sweeped over.
The results are collected in a subdirectory of the `multirun` directory.

Further options for the configuration can be seen in the `config` directory.
We use RTX3090 GPUs and most experiments complete within a couple of hours.

To simply rerun all experiments, use `bash run_experiments.sh`.

### Preparing results
After an experiment is run, the results need to be aggregated into a pandas DataFrame for plotting.
First, open `notebooks/21.1-collate_results.py` and the required paths (You can find where with Ctrl+F: CHANGE).
Afterwards:

```shell
python notebooks/21.1-collate_results.py
python notebooks/29.1-compute_nodeprops.py
```

### Plots
The paper plots and tables can be created with notebooks 22 to 31.


## Contact
Please contact max.klabunde@uni-passau.de in case you have any questions.

## Cite
If you find this work helpful in your research, please consider citing our work.

```
@article{klabunde_prediction_2022,
	title = {On the {Prediction} {Instability} of {Graph} {Neural} {Networks}},
	journal = {arXiv preprint arXiv:2205:10070},
	author = {Klabunde, Max and Lemmerich, Florian},
	year = {2022}
}
```
