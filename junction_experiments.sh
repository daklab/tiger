#!/bin/sh

SEED=853211

# data filtering experiments
python3 experiments.py --dataset junction --experiment label-and-filter --normalization No --context 0 --seq_only --seed $SEED
python3 experiments.py --dataset junction --experiment "label-and-filter (off-target)" --normalization No --context 0 --seq_only --seed $SEED

# data normalization experiments
python3 experiments.py --dataset junction --experiment normalization --context 0 --seq_only --seed $SEED

# configuration experiments
python3 experiments.py --dataset junction --experiment model --context 0 --seed $SEED
python3 experiments.py --dataset junction --experiment context --seed $SEED

# non-sequence feature importance
python3 experiments.py --dataset junction --experiment feature-groups-individual
python3 experiments.py --dataset junction --experiment feature-groups-cumulative

# SHAP experiments
python3 experiments.py --dataset junction --experiment SHAP

# predictions
python3 junction_predictions.py --training_set junction --seed $SEED
python3 junction_predictions.py --training_set off-target --seed $SEED
python3 junction_predictions.py --training_set off-target --correct --seed $SEED
python3 junction_predictions.py --training_set combined --seed $SEED
python3 junction_predictions.py --training_set combined --correct --seed $SEED

# RBP experiments
python3 junction_rbp_model.py --mode junction --seed 112358
python3 junction_rbp_model.py --mode junction --rbp_junc --seed 112358
python3 junction_rbp_model.py --mode junction --rbp_nt --seed 112358
python3 junction_rbp_model.py --mode junction --rbp_nt --sum_peaks --seed 112358
python3 junction_rbp_model.py --mode junction --rbp_nt_relaxed --seed 112358
python3 junction_rbp_model.py --mode junction --rbp_nt_relaxed --sum_peaks --seed 112358
python3 junction_rbp_model.py --mode target --seed 112358
python3 junction_rbp_model.py --mode target --rbp_junc --seed 112358
python3 junction_rbp_model.py --mode target --rbp_nt --seed 112358
python3 junction_rbp_model.py --mode target --rbp_nt --sum_peaks --seed 112358
python3 junction_rbp_model.py --mode target --rbp_nt_relaxed --seed 112358
python3 junction_rbp_model.py --mode target --rbp_nt_relaxed --sum_peaks --seed 112358

# generate plots
python3 analysis.py --dataset junction
python3 junction_figures.py
