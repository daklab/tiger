#!/bin/sh

# configuration experiments
python3 experiments.py --dataset off-target --holdout targets --experiment model --context 0
python3 experiments.py --dataset off-target --holdout targets --experiment context

# learning curve experiments
python3 experiments.py --dataset off-target --holdout genes --experiment learning-curve
python3 experiments.py --dataset off-target --holdout targets --experiment learning-curve

# non-sequence feature importance
python3 experiments.py --dataset off-target --holdout targets --experiment feature-groups-individual
python3 experiments.py --dataset off-target --holdout targets --experiment feature-groups-cumulative

# SHAP experiments
python3 experiments.py --dataset off-target --holdout targets --experiment SHAP

# lucky Cas13
SEED=13

# on-target predictions
python3 predictor_validation.py --seed $SEED --dataset off-target --pm_only --holdout genes
python3 predictor_validation.py --seed $SEED --dataset off-target --pm_only --holdout guides
python3 predictor_validation.py --seed $SEED --dataset off-target --pm_only --holdout targets
python3 train_and_test.py --seed $SEED --dataset off-target --pm_only --test_dataset flow-cytometry

# off-target predictions
python3 predictor_validation.py --seed $SEED --dataset off-target --mm_only --holdout genes
python3 predictor_validation.py --seed $SEED --dataset off-target --mm_only --holdout guides
python3 predictor_validation.py --seed $SEED --dataset off-target --mm_only --holdout targets
python3 train_and_test.py --seed $SEED --dataset off-target --mm_only --test_dataset flow-cytometry

# combined predictions
python3 predictor_validation.py --seed $SEED --dataset off-target --holdout genes
python3 predictor_validation.py --seed $SEED --dataset off-target --holdout guides
python3 predictor_validation.py --seed $SEED --dataset off-target --holdout targets
python3 train_and_test.py --seed $SEED --dataset off-target --test_dataset flow-cytometry

# titration validation
python3 train_and_test.py --seed $SEED --dataset off-target --test_dataset hap-titration

# benefit of new, larger dataset
python3 train_and_test.py --seed $SEED --dataset flow-cytometry --test_dataset off-target

# generate plots
python3 analysis.py --dataset off-target --holdout genes
python3 analysis.py --dataset off-target --holdout targets
python3 tiger_figures.py

# web tool normalization experiments
python3 experiments.py --dataset off-target --experiment normalization --seq_only
python3 analysis.py --dataset off-target --holdout targets --seq_only
