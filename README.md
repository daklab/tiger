### A deep learning framework to predict on-target and off-target activity of CRISPR-Cas13d guide RNAs

For more information please refer to our manuscript.

### Requirements

This repository utilizes TensorFlow as its deep learning framework.
We utilize TensorFlow's recommended [Docker installation](https://www.tensorflow.org/install/docker).
Our environment is based upon TensorFlow's published Docker tag for version 2.11.0 with GPU support:
`tensorflow:2.11.0-gpu`.
Our `Dockerfile` utilizes this image to create our Docker container, which also depends on the python packages listed in
`requirements.txt`.

Our repository contains two submodules:
- [Open source code](https://github.com/yandexdataschool/roc_comparison/) for ROC analysis. We submodule their code and license in the `roc_comparison` directory.
- [Our hugging face code](https://huggingface.co/spaces/Knowles-Lab/tiger) has saved model weights and a script to generate predictions locally. We submodule this code in the `hugging-face` directory.
When cloning are repository be sure to use the `--recursive option`. For example:
```
git clone --recursive https://github.com/daklab/tiger.git
```

### Repository Overview

#### Data
We provide all data and fold assignments required reproduce figures in our manuscript in the following directories:
- `/data-processed/off-target.bz2` is the primary dataset, a survival screen for HEK293T cells.
- `/data-processed/off-target-nt.bz2` contains non-targeting data for the item above.
- `/data-processed/hap-titration.bz2` is the titration validation dataset, a survival screen for HAP1 cells.
- `/data-processed/hap-titration-nt.bz2` contains non-targeting data for the item above.
- `/data-processed/flow-cytometry.bz2` contains data from [_Wessels, Méndez-Mancilla 2020_].

#### Other Models' Predictions

The directory `predictions (other models)` contains the predictions made by other Cas13d models.
We uploaded the full transcript for each of the sixteen genes targeted in our HEK293 experiment to the web portals of [_Cheng, Li 2021_] and [_Wei 2021_] for their predictions.
Our code only analyzes their performance for the perfect match guides in our HEK293T dataset.
Please see our methods section for additional collection details.

#### Experiments
To reproduce our manuscript's figures please run `tiger_experiments.sh`.
There, we provide the random number seeds.
While we enable TensorFlow's GPU determinism, exact reproducibility may vary depending on computing resources (e.g. GPU model/resources, CPU instead of GPU, etc...).
We used an NVIDIA 3090 RTX card for all experiments.
