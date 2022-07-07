#!/bin/bash

tmux new -s run -d
tmux send-keys "conda activate gnn-pred-stab" C-m
tmux send-keys "conda env config vars set CUBLAS_WORKSPACE_CONFIG=:4096:8" Enter
tmux send-keys "conda activate gnn-pred-stab" Enter
cuda=3  # CHANGE: can be an integer or "cpu". Recommended: run parts of this script on different GPUs to get results much more quickly.
reps=50  # 50 for full experiment, 2 for testing

# Baseline
tmux send-keys "
python scripts/run.py dataset=citeseer,pubmed,computers,photo,cs,physics,wikics model=pubmed_gat2017,pubmed_gcn2017 cuda=$cuda n_repeat=$reps cka.use_masks=[] -m &
wait" Enter

# Training size
tmux send-keys "
python scripts/run.py dataset=citeseer,pubmed,computers,photo,cs,physics,wikics model=pubmed_gat2017,pubmed_gcn2017 cuda=$cuda n_repeat=$reps proportional_split=true public_split=false part_test=0.84,0.8,0.75,0.65,0.45,0.25 cka.use_masks=[] datasplit_seed=0,1,2,3,4,5,6,7,8,9 -m &
wait" Enter

# Optimizer
tmux send-keys "
python scripts/run.py dataset=citeseer,pubmed,computers,photo,cs,physics,wikics model=pubmed_gat2017,pubmed_gcn2017 cuda=$cuda n_repeat=$reps optim=adam,sgd,sgd_momentum cka.use_masks=[] -m &
wait" Enter

# L2
tmux send-keys "
python scripts/run.py dataset=citeseer,pubmed,computers,photo,cs,physics,wikics model=pubmed_gat2017,pubmed_gcn2017 cuda=$cuda n_repeat=$reps optim.weight_decay=0.0,1e-6,1e-5,1e-4,1e-3 cka.use_masks=[] -m &
wait" Enter

# Dropout
tmux send-keys "
python scripts/run.py dataset=citeseer,pubmed,computers,photo,cs,physics,wikics model=pubmed_gat2017,pubmed_gcn2017 cuda=$cuda n_repeat=$reps model.dropout_p=0.0,0.2,0.4,0.6,0.8 cka.use_masks=[] -m &
wait" Enter

# Width
tmux send-keys "
python scripts/run.py dataset=citeseer,pubmed,computers,photo,cs,physics,wikics model=pubmed_gat2017 cuda=$cuda n_repeat=$reps model.hidden_dim=1,2,4,8,16,32 cka.use_masks=[] -m &
wait" Enter
tmux send-keys "
python scripts/run.py dataset=citeseer,pubmed,computers,photo,cs,physics,wikics model=pubmed_gcn2017 cuda=$cuda n_repeat=$reps model.hidden_dim=8,16,32,64,128,256 cka.use_masks=[] -m &
wait" Enter

# Depth
tmux send-keys "
python scripts/run.py dataset=citeseer,pubmed,computers,photo,cs,physics,wikics model=pubmed_gat2017,pubmed_gcn2017 cuda=$cuda n_repeat=$reps model.n_layers=2,3,4,5,6 -m &
wait" Enter

# Combination
tmux send-keys "
python scripts/run.py dataset=citeseer,pubmed,computers,photo,cs,wikics model=pubmed_gat2017 cuda=$cuda n_repeat=$reps model.hidden_dim=32 model.dropout_p=0.2 optim.weight_decay=1e-4 cka.use_masks=[] -m &
wait" Enter
tmux send-keys "
python scripts/run.py dataset=physics model=pubmed_gat2017 cuda=$cuda n_repeat=$reps model.hidden_dim=25 model.dropout_p=0.2 optim.weight_decay=1e-4 cka.use_masks=[] -m &
wait" Enter
tmux send-keys "
python scripts/run.py dataset=citeseer,pubmed,computers,photo,cs,physics,wikics model=pubmed_gcn2017 cuda=$cuda n_repeat=$reps model.hidden_dim=256 model.dropout_p=0.2 optim.weight_decay=1e-4 cka.use_masks=[] -m &
wait" Enter
