# EEG Brain Age Estimation

A deep learning project for estimating age groups from resting-state EEG
(Electroencephalography) recordings using a CNN-LSTM architecture.

## Overview

This project explores EEG-based age-group classification using the LEMON
dataset. EEG recordings are converted into Mel spectrogram representations,
which are processed by a convolutional neural network and a bidirectional
LSTM.

The task is formulated as classification into nine age-related classes,
following the label mapping defined in the project code.

## Pipeline

1. Load participant metadata and construct age labels.
2. Discover EEG recordings and identify common channels.
3. Divide recordings into fixed-duration EEG blocks.
4. Convert EEG blocks into Mel spectrogram representations.
5. Split samples by participant to avoid subject overlap between data splits.
6. Train a CNN-LSTM classifier.
7. Evaluate the model using classification metrics and confusion matrices.

## Model

The model combines:

- **Shared CNN:** extracts spatial and spectral patterns from EEG
  spectrograms.
- **Bidirectional LSTM:** models relationships across the sequence of EEG
  blocks.
- **Classification layer:** predicts one of the age-related classes.

The exact architecture and hyperparameters are defined in `src/eeg_brain_age/`.

## Repository Structure

```text
eeg-brain-age/
├── README.md
├── requirements.txt
├── .gitignore
├── checkpoints/
├── notebooks/
│   ├── brain_age_experiment.ipynb
│   └── brain_age_experiment_refactored.ipynb
├── reports/
│   └── figures/
└── src/
    └── eeg_brain_age/
        ├── data.py
        ├── dataloaders.py
        ├── dataset.py
        ├── evaluation.py
        ├── features.py
        ├── models.py
        ├── splits.py
        ├── training.py
        └── visualization.py
```

## Requirements

- Python 3.10 or newer is recommended.
- Dependencies are listed in `requirements.txt`.
- PyTorch should be installed with a build appropriate for your hardware
  and CUDA requirements.

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, follow the official PyTorch installation instructions
to select a compatible CUDA build.

## Dataset and Configuration

This project uses EEG recordings and participant metadata from the LEMON
dataset. Access the dataset through its official distribution and comply
with its access conditions and data-use requirements.

The raw EEG recordings and metadata are not included in this repository.
Configure the data and metadata paths in the notebook for your local
environment before running the experiment.

The current notebook was developed in Google Colab and uses Google Drive
paths. Those paths must be adjusted when running elsewhere.

## Running the Experiment

Open:

`notebooks/brain_age_experiment_refactored.ipynb`

Run the notebook cells in order, after configuring the dataset paths and
mounting or accessing the storage containing the data.

The notebook uses the modules in `src/eeg_brain_age/` for data preparation,
model definition, training, evaluation, and visualization.

## Outputs

Depending on the experiment configuration, the workflow can produce:

- A trained model checkpoint.
- Training and validation curves.
- Classification metrics.
- Confusion matrices and class-level evaluation reports.

Generated artifacts, cached features, raw EEG files, and model checkpoints
are excluded from Git by the current `.gitignore` configuration.

## Reproducibility Notes

Results can depend on the dataset version, participant selection, split
configuration, software versions, and training settings. Consult the
notebook and source modules for the exact configuration used in a run.

The repository structure and documentation do not, by themselves, imply
that the experiment has been independently reproduced.

## Limitations

This is a research and portfolio project. Age-group classification from
EEG should not be interpreted as a clinical diagnostic tool.

## License

A license has not yet been specified. Check the applicable dataset terms
and decide on an appropriate code license before redistributing the project.
