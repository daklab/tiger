#!/bin/sh

# configuration experiments
python3 experiments.py --dataset off-target --experiment model --filter_method NoFilter --holdout targets
python3 experiments.py --dataset off-target --experiment context --filter_method NoFilter --holdout targets --model Tiger2D

# learning curve experiments
python3 experiments.py --context 1 --dataset off-target --experiment learning-curve --filter_method NoFilter --holdout genes --model Tiger2D
python3 experiments.py --context 1 --dataset off-target --experiment learning-curve --filter_method NoFilter --holdout targets --model Tiger2D

# non-sequence feature importance
python3 experiments.py --context 1 --dataset off-target --experiment feature-groups-individual --filter_method NoFilter --holdout targets --model Tiger2D
python3 experiments.py --context 1 --dataset off-target --experiment feature-groups-cumulative --filter_method NoFilter --holdout targets --model Tiger2D

# SHAP experiments
python3 experiments.py --context 1 --dataset off-target --experiment SHAP --filter_method NoFilter --holdout guides --model Tiger2D
python3 experiments.py --context 1 --dataset off-target --experiment SHAP --filter_method NoFilter --holdout targets --model Tiger2D

# on-target predictions
python3 predictor_validation.py --context 1 --dataset off-target --filter_method NoFilter --holdout genes --model Tiger2D --pm_only --seed 112358
python3 predictor_validation.py --context 1 --dataset off-target --filter_method NoFilter --holdout guides --model Tiger2D --pm_only --seed 112358
python3 predictor_validation.py --context 1 --dataset off-target --filter_method NoFilter --holdout targets --model Tiger2D --pm_only --seed 13
python3 train_and_test.py --context 1 --dataset off-target --filter_method NoFilter --test_dataset flow-cytometry --model Tiger2D --pm_only --seed 112358

# off-target predictions
python3 predictor_validation.py --context 1 --dataset off-target --filter_method NoFilter --holdout genes --model Tiger2D --mm_only --seed 112358
python3 predictor_validation.py --context 1 --dataset off-target --filter_method NoFilter --holdout guides --model Tiger2D --mm_only --seed 112358
python3 predictor_validation.py --context 1 --dataset off-target --filter_method NoFilter --holdout targets --model Tiger2D --mm_only --seed 13
python3 train_and_test.py --context 1 --dataset off-target --filter_method NoFilter --test_dataset flow-cytometry --model Tiger2D --mm_only --seed 112358

# combined predictions
python3 predictor_validation.py --context 1 --dataset off-target --filter_method NoFilter --holdout genes --model Tiger2D --seed 112358
python3 predictor_validation.py --context 1 --dataset off-target --filter_method NoFilter --holdout guides --model Tiger2D --seed 112358
python3 predictor_validation.py --context 1 --dataset off-target --filter_method NoFilter --holdout targets --model Tiger2D --seed 13
python3 train_and_test.py --context 1 --dataset off-target --filter_method NoFilter --test_dataset flow-cytometry --model Tiger2D --seed 112358

# titration validation
python3 train_and_test.py --context 1 --dataset off-target --filter_method NoFilter --test_dataset hap-titration --model Tiger2D --seed 13

# benefit of new, larger dataset
python3 train_and_test.py --context 1 --dataset flow-cytometry --filter_method NoFilter --model Tiger2D --test_dataset off-target --seed 112358

# generate plots
python3 analysis.py --dataset off-target --holdout genes
python3 analysis.py --dataset off-target --holdout guides
python3 analysis.py --dataset off-target --holdout targets
python3 tiger_figures.py