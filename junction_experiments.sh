#!/bin/sh

# data filtering experiments
python3 experiments.py --dataset junction --seq_only --experiment "label-and-filter"
python3 experiments.py --dataset junction --seq_only --experiment "label-and-filter (off-target)"

# SHAP experiments
python3 experiments.py --experiment SHAP --dataset off-target --pm_only
python3 experiments.py --experiment SHAP --dataset junction
python3 experiments.py --experiment SHAP --dataset junction-splice-sites --context 0

# predictions
python3 junction_predictions_tiger.py --seed 13
python3 junction_predictions_seabass.py --seed 13 --dataset junction --loss mse
python3 junction_predictions_seabass.py --seed 13 --dataset junction --loss mse --use_lfc

# generate plots
python3 analysis.py --dataset junction --seq_only
python3 analysis.py --dataset junction
python3 junction_figures.py
