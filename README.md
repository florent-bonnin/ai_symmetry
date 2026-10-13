# AI Symmetry

A deep learning experiment in **detecting reflection symmetry in circular binary sequences**, built with TensorFlow and Keras.

The central idea is to make a neural network recognize a symmetric pattern regardless of where the sequence starts. The project explores both conventional dense networks and a custom architecture that explicitly handles circular rotations.

## The problem

Given a binary sequence of a fixed length, the model predicts whether it belongs to the symmetric class (`1`) or the non-symmetric class (`0`).

A symmetric example is constructed by creating a palindrome and optionally rotating it. For instance:

```text
Palindrome:  0110
Rotated:     1100
```

Although `1100` is not a palindrome when read linearly, it is a circular rotation of one. The network therefore needs to recognize patterns independently of their starting position.

The dataset generator also supports harder training examples, including near-symmetric sequences and noisy positives. In these modes, labels identify **how examples were generated**; they are not always a guarantee of exact mathematical symmetry.

## Model architecture

The training script defines three experimental architectures:

- **Simple dense model:** a ReLU hidden layer followed by a sigmoid output.
- **Deep dense model:** multiple dense layers with input concatenations, dropout, and an optional random circular-shift augmentation layer.
- **Advanced model (currently selected):** a custom `AdvancedLayer` designed to make the prediction invariant to circular shifts.

### The advanced model

For a sequence of length `N`, `AdvancedLayer`:

1. Evaluates all `N` circular rotations of the input.
2. Passes each rotation through the **same** pair of dense layers: ReLU, then sigmoid.
3. Takes the **maximum** prediction across rotations.

```text
                 Input: N bits
                       |
              Circular rotations
               /      |      \
           shift 0  shift 1  ... shift N-1
               \      |      /
          Shared Dense (ReLU)
          Shared Dense (sigmoid)
                       |
             Maximum across shifts
                       |
                 Output: 0–1
```

Because the weights are shared and every rotation is considered, circularly shifting the input does not change the resulting score (apart from numerical effects). The default hidden-layer width is `N` units.

Training uses **binary cross-entropy**, the **Adam** optimizer, and **accuracy** as a metric. Learning-rate reduction on a plateau and early stopping are enabled.

> **Note:** The advanced model is the active architecture in `symmetry_learn.py`. The other models are retained as experiments; the deep model's architecture image is generated, but the deep model itself is not trained by default.

## Repository contents

```text
ai_symmetry/
├── README.md
├── symmetry_dataset.py   # Generate labeled binary sequences
├── symmetry_learn.py     # Build, train, and evaluate the models
└── symmetry_predict.py   # Example inference script (see note below)
```

The scripts create or use these local folders:

```text
datasets/                  # Generated CSV datasets
results/                   # Saved models, plots, and model diagram
```

## Getting started

### 1. Install dependencies

Clone the repository and create a Python environment with a TensorFlow-compatible Python version:

```bash
git clone https://github.com/florent-bonnin/ai_symmetry.git
cd ai_symmetry
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install tensorflow keras matplotlib pydot
```

[Graphviz](https://graphviz.org/download/) must also be installed and available on your system for `keras.utils.plot_model()`.

There is currently no pinned dependency file, so exact library versions are not specified by the repository.

### 2. Generate a dataset

Open `symmetry_dataset.py` and select a generation mode using the `MODE` constant (the committed default is `"noisy"`). Then run:

```bash
python symmetry_dataset.py
```

The script prompts for:

```text
nb_bits = 100
nb_examples = 10000
```

This creates `datasets/symmetry_100_10000.csv`. Each row contains a binary string and a label, with no header:

```csv
0110,1
0100,0
```

The two rows above illustrate the **file format**; they are not a sample of the 100-bit dataset.

### 3. Train and evaluate

Create the output folder **before** launching training: the script writes the model diagram before its own directory-creation step.

```bash
mkdir -p results
python symmetry_learn.py
```

When prompted, enter the same bit length and example count used for dataset generation, then choose the maximum number of epochs:

```text
nb_bits = 100
nb_examples = 10000
nb_epochs = 100
```

The script:

1. Reads and converts the CSV strings into floating-point bit vectors.
2. Splits the data into **80% training, 10% validation, and 10% testing**.
3. Trains the selected network with shuffled training batches of 32.
4. Restores the best weights when early stopping is triggered.
5. Saves the trained model and evaluates it on the test split.
6. Saves training/validation loss and accuracy plots.

With the default width multiplier (`NN_WIDTH = 1`), a 100-bit model is saved to:

```text
results/symmetry_100_100.keras
```

Other generated files include `results/loss.png`, `results/accuracy.png`, and `results/deep_model.png`.

### 4. Run predictions (adaptation required)

`symmetry_predict.py` demonstrates the intended inference workflow: load a Keras model, convert binary strings into numeric tensors, and call `model.predict()`.

**It is not synchronized with the current training script:** it loads the hard-coded path `results/model_100_1000000.keras`, while training saves a file named `results/symmetry_<nb_bits>_<nn_width>.keras`. It also contains fixed 100-bit inputs.

To reuse a saved advanced model, you must update the path and make the custom `AdvancedLayer` available during deserialization (for example, by registering the layer and implementing an appropriate serialization configuration). As committed, this script should be treated as an inference example rather than a ready-to-run command.

## Dataset generation modes

Change `MODE` in `symmetry_dataset.py`:

| Mode | Generation strategy |
| --- | --- |
| `normal` | Random negative examples and positive examples built from rotated palindromes. |
| `easy` | Random negatives and palindromic positives whose rotation is limited by `ROTATION` (`0` by default). |
| `trap` | Negative examples produced by swapping two adjacent, different bits in a symmetric sequence; positives are generated normally. |
| `noisy` | Random negatives and symmetric positives with some positions replaced by randomly chosen bits. **Default mode.** |
| `all_random` | Random bit strings on both sides of the label; useful as a control, not a meaningful symmetry-labeling mode. |

The generator chooses between the two label branches with approximately equal probability.

For `noisy` mode, the number of selected positions is `int(nb_bits * NOISE)`, with `NOISE = 0.01` by default. Replacement is random and does **not** necessarily flip a bit; some replacements leave the sequence unchanged.

**Label caveat:** The generator does not use `is_symmetrical()` to verify or correct labels. Random negatives can occasionally be symmetric, and noisy positives or trap examples may not match their intended geometric class. Interpret the resulting accuracy as performance on the generated labels.

## Configuration

The main settings are constants near the top of the scripts:

| File | Setting | Committed value | Purpose |
| --- | --- | --- | --- |
| `symmetry_dataset.py` | `MODE` | `"noisy"` | Dataset generation strategy |
| `symmetry_dataset.py` | `NOISE` | `0.01` | Fraction of positions selected for random replacement |
| `symmetry_dataset.py` | `ROTATION` | `0` | Maximum rotation in `easy` mode |
| `symmetry_learn.py` | `TRAIN_RATIO` | `0.8` | Proportion used for training |
| `symmetry_learn.py` | `BATCH_SIZE` | `32` | Training and evaluation batch size |
| `symmetry_learn.py` | `NN_WIDTH` | `1` | Hidden-layer width multiplier relative to sequence length |
| `symmetry_learn.py` | `NN_DEPTH` | `9` | Number of repeated blocks in the inactive deep architecture |
| `symmetry_learn.py` | `FREEZE` | `False` | Enables an additional experimental training stage |

## Scope and limitations

This repository is an experimental learning project, not a packaged production classifier. In particular:

- Training and generation are controlled by interactive prompts and source-code constants rather than CLI arguments.
- The data-generation modes intentionally explore different difficulty levels, but some introduce label ambiguity.
- The active model operates on a **fixed input length** chosen when it is built.
- The repository does not include pretrained weights, datasets, a dependency lockfile, or documented benchmark results.
- Model reloading with the custom layer and the inference script require additional integration work.

The main purpose is to investigate how neural-network architecture and dataset construction affect the learning of **symmetry and rotation invariance**.
