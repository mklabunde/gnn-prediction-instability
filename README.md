# On the Prediction Instability of Graph Neural Networks
Implementation and supplementary material of the paper "On the Prediction Instability of Graph Neural Networks", accepted at ECML PKDD 2022.
Note that we made some small corrections for the published version compared to the arxiv version (that we may not update). 

[[Springer]](https://doi.org/10.1007/978-3-031-26409-2_12)
[[ECML PKDD]](https://2022.ecmlpkdd.org/wp-content/uploads/2022/09/sub_955.pdf)
[[arxiv]](https://arxiv.org/abs/2205.10070)
[[Supplementary]](supplementary_material.pdf)


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
The code from [this commit](https://github.com/mklabunde/gnn-prediction-instability/commit/969e84c4ed147fbd242d3080cc6e7c26e5057472) was used to produce the final results.
### Training
To simply rerun all experiments, use `bash run_experiments.sh` (Takes >24h on a single RTX3090).
If you want to reproduce the results, we recommend splitting the commands in the file into multiple scripts and executing them individually on different GPUs.

We use [Hydra](https://hydra.cc/) to run the experiments.
The following command runs 50 times repeated training on GPU 0 over the listed datasets:

```shell
python scripts/run.py cuda=0 n_repeat=50 dataset=computers,photo,cs,physics,wikics model=pubmed_gat2017 -m
```

The `-m` flag allows you to specify multiple values for arguments, which are then sweeped over.
The results are collected in a subdirectory of the `multirun` directory.

Further options for the configuration can be seen in the `config` directory.
We use RTX3090 GPUs and most experiments complete within a couple of hours.

### Preparing results
After an experiment is run, the results need to be aggregated into a pandas DataFrame for plotting.
First, open `notebooks/21.1-collate_results.py` and add the required paths (You can find where with Ctrl+F: CHANGE).
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
