# Targeted Inhibition of Gene Expression via gRNA Design (TIGER)

This code repository accompanies [our Nature Biotechnology manuscript](http://sanjanalab.org/reprints/WesselsStirn_NBT_2023.pdf) as well as [our recent preprint](https://www.biorxiv.org/content/10.1101/2023.09.12.557474v1).
Please consider citing us:
> **[Prediction of on-target and off-target activity of CRISPR–Cas13d guide RNAs using deep learning](http://sanjanalab.org/reprints/WesselsStirn_NBT_2023.pdf).** Wessels, H.-H.<sup>\*</sup>, Stirn, A.<sup>\*</sup>, Méndez-Mancilla, A., Kim, E. J., Hart, S. K., Knowles, D. A.<sup>#</sup>, & Sanjana, N. E.<sup>#</sup> *Nature Biotechnology* (2023).  [https://doi.org/10.1038/s41587-023-01830-8](https://doi.org/10.1038/s41587-023-01830-8)

> **[Cas13d-mediated isoform-specific RNA knockdown with a unified computational and experimental toolbox](https://www.biorxiv.org/content/10.1101/2023.09.12.557474v1).** Schertzer, M. D., Stirn, A., Isaev, K., Pereira, L., Das, A., Harbison, C., Park, S. H., Wessels, H.-H., Sanjana, N. E., & Knowles, D. A. *Bioarxiv* (2023).

## Requirements

This repository utilizes TensorFlow as its deep learning framework.
We utilize TensorFlow's recommended [Docker installation](https://www.tensorflow.org/install/docker).
Our environment is based upon TensorFlow's published Docker tag for version 2.11.0 with GPU support:
`tensorflow:2.11.0-gpu`.
Our `Dockerfile` utilizes this image and additionally installs the python packages listed in `requirements.txt`.

Our repository contains two submodules:
- [Open source code](https://github.com/yandexdataschool/roc_comparison/) for ROC analysis. We submodule their code and license in the `roc_comparison` directory.
- [Our online TIGER tool on hugging face](https://huggingface.co/spaces/Knowles-Lab/tiger) has saved model weights and a script to generate predictions locally. We submodule this code in the `hugging_face` directory.

The `hugging_face` repository uses [Git LFS](https://git-lfs.com/) to store model weights.
One must have Git LFS properly installed to correctly pull model parameters.
When cloning our repository be sure to use the `--recursive` option to pull both submodules:
```
git clone --recursive https://github.com/daklab/tiger.git
```

## Generating Predictions

[Our online TIGER tool](https://huggingface.co/spaces/Knowles-Lab/tiger) is the simplest way to generate predictions.
There, one can enter a single transcript manually or upload a fasta file with multiple transcripts and select one of three modes:
1) Score all possible gRNAs per transcript.
2) Identify and score the top ten guides per transcript. This mode supports off-target checking for up to three nucleotide substitutions against all gencode (v19) coding and lncRNA transcripts.
3) Titration mode identifies and scores the top ten guides per transcript and reports and scores all single-mismatch titration candidates for each transcripts' top ten guides.

For faster performance, one can fork our hugging face repository and pay for upgraded compute.
Alternatively, for those with local GPU resources, one can call `tiger.py` (located in the `hugging_face` submodule directory of this github repository) locally:
```
python tiger.py --mode <all, top_guides, or titration> --fasta_path <path to a directory of fasta files> [--check_off_targets]
```
The required mode argument has three valid options:`all`, `top_guides`, or `titration`, which correspond to the modes described above.
The required `--fasta_path` argument must be a directory of fasta files, where each file has one or more transcripts.
Upon completion, `on_target.csv` (located in the `hugging_face` submodule directory of this GitHub repository) will contain either all or the ten most effective guides per transcript depending on the selected mode.
If titration mode was selected, then `titration.csv` will contain all single mismatches for the guides in `on_target.csv`.
If the optional `--check_off_targets` was enabled, `off_target.csv` will contain potential off-target effects (up to three nucleotide substitutions) for all guides in `on_target.csv`.

### Versioning and Training of Online TIGER Tool

The `tiger_trainer.py` script herein trains [our online TIGER tool](https://huggingface.co/spaces/Knowles-Lab/tiger).
We mark all versions of this tool with a vX.x tag.
This GitHub repository has corresponding vX.x tags to identify the revision of `tiger_trainer.py` used to train the online tool.

## Repository Overview

### Data
We provide all data and fold assignments required reproduce figures in our manuscript in the following directories:
- `/data-processed/off-target.bz2` is the primary dataset, a survival screen for HEK293T cells.
- `/data-processed/off-target-nt.bz2` contains non-targeting data for the item above.
- `/data-processed/hap-titration.bz2` is the titration validation dataset, a survival screen for HAP1 cells.
- `/data-processed/hap-titration-nt.bz2` contains non-targeting data for the item above.
- `/data-processed/flow-cytometry.bz2` contains data from [_Wessels, Méndez-Mancilla 2020_].

### Other Models' Predictions

The directory `predictions (other models)` contains the predictions made by other Cas13d models.
We uploaded the full transcript for each of the sixteen genes targeted in our HEK293 experiment to the web portals of [_Wei et al. 2021_] and [_Cheng et al. 2023_] for their predictions.
Our code only analyzes their performance for the perfect match guides in our HEK293T dataset.
Please see our methods section for additional collection details.

## Reproducing Experiments

While we enable TensorFlow's GPU determinism, exact reproducibility may vary depending on computing resources (e.g. GPU model/resources, CPU instead of GPU, etc...).
We used an NVIDIA 3090 RTX card for all experiments.

To reproduce results from [our Nature Biotechnology manuscript](http://sanjanalab.org/reprints/WesselsStirn_NBT_2023.pdf), please see and run `tiger_experiments.sh`, which has our utilized random number seeds.
To reproduce results from [our recent preprint](https://www.biorxiv.org/content/10.1101/2023.09.12.557474v1), please see and run `junction_experiments.sh`, which also has our utilized random number seeds.
