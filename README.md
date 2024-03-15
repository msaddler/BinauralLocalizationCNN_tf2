# Tensorflow2 implementation of BinauralLocalizationCNN

This repository contains a minimal, more recent `tensorflow-2.x` implementation of the [Francl and McDermott (2022)](https://www.nature.com/articles/s41562-021-01244-z) sound localization CNN model. The original `tensorflow-1.x` implementation can be found in the [BinauralLocalizationCNN](https://github.com/afrancl/BinauralLocalizationCNN) repository.

## Setup

- Clone this repository
- Download the trained model weights (checkpoint files for both implementations) and test stimuli from our [Google Drive](https://drive.google.com/drive/folders/1s9UC4DoksC7mTlxL2ZQFrn8XI0uJcsoV?usp=share_link).

#### Python packages
```
Package                      Version
---------------------------- ---------
h5py                         3.9.0
keras                        2.13.1
matplotlib                   3.8.2
numpy                        1.24.3
pandas                       1.5.3
scipy                        1.10.1
tensorflow                   2.13.0
tensorflow-io                0.33.0
tqdm                         4.65.0
```

## Contents

- [DEMO_tf2_model.ipynb](DEMO_tf2_model.ipynb) :
    - Minimal code example of building the model, loading trained weights, and evaluating on example sound stimuli
- [compare_tf1_and_tf2_model_weights_and_outputs.ipynb](compare_tf1_and_tf2_model_weights_and_outputs.ipynb) :
    - Code tests comparing model weights and outputs between the `tensorflow-2.x` and `tensorflow-1.x` model implementations
- [run_model_optimization.sh](run_model_optimization.sh) and [run_model.py](run_model.py) :
    - Example code for how to train or evaluate the model at scale on `tfrecords` files
- [models/tensorflow2](models/tensorflow2) and [models/tensorflow2](models/tensorflow1) :
    - Model directories containing config files with cochlear model parameters and optimization hyperparameters
    - The 10 subdirectories contain the 10 convolutional neural network architectures
    - Trained model weights can be downloaded from our [Google Drive](https://drive.google.com/drive/folders/1s9UC4DoksC7mTlxL2ZQFrn8XI0uJcsoV?usp=share_link)


## Contact

Mark R. Saddler (msaddler@mit.edu)
