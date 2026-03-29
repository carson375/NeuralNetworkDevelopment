# Constellation Classification Neural Network

This project implements a Multi-Layer Perceptron (MLP) to classify constellations based on star data. Originally developed as a Jupyter Notebook project for a Machine Learning class, it has been refactored into a modular Python application.

## Project Overview
The model uses celestial coordinates and star properties to determine which constellation a given set of data points belongs to. The training data was originally sourced from star catalogs and processed using PyTorch for the neural network implementation.

## New Project Structure
The project has been restructured for better maintainability and scalability:

```text
NeuralNetworkDevelopment/
├── data/
│   ├── raw/                # Original star data (Xtr.csv, Xts.csv, ytr.csv)
│   └── processed/          # Standardized data for model training
├── src/
│   ├── __init__.py         # Makes src a Python package
│   ├── model.py            # Neural Network architecture (ConstellationNet)
│   ├── dataset.py          # Data loading and RobustScaler preprocessing
│   ├── train.py            # Model training and validation loops
│   └── predict.py          # Inference and AUC calculation
├── models/
│   └── saved_models/       # Persistent storage for .pt model files
├── results/                # Output directory for prediction CSVs
├── main.py                 # Central entry point for the pipeline
├── neural_net.ipynb        # Original notebook (preserved for plots/reference)
├── requirements.txt        # Project dependencies
└── .gitignore              # Prevents tracking of venv and temporary files
```

## Getting Started

### Prerequisites
- Python 3.8+ (Homebrew version recommended)
- Virtual Environment (`venv`)

### Installation
1. Activate your virtual environment:
   ```bash
   source venv/bin/activate
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The project uses `main.py` as a centralized command-line interface.

### Run Inference (Default)
If a trained model exists in `models/saved_models/`, this will load it, evaluate it on the validation set, and generate test predictions:
```bash
python main.py
```

### Train a New Model
To force the model to re-train from scratch:
```bash
python main.py --train --epochs 20 --lr 0.001
```

### Command Line Arguments
- `--train`: Force the model to train even if a saved model exists.
- `--epochs`: Number of training iterations (default: 10).
- `--lr`: Learning rate for the Adam optimizer (default: 1e-3).
- `--batch_size`: Size of data batches for training (default: 10).

## Model Architecture
The `ConstellationNet` is a symmetric encoder-decoder style MLP:
- **Input Layer**: 8 features (standardized via `RobustScaler`).
- **Hidden Layers**: Sequential expansion from 16 to 512 units, followed by a symmetric reduction back to 16 units.
- **Activation**: `LeakyReLU` for hidden layers and `Sigmoid` for the final output.
- **Regularization**: Includes `Dropout` (20%) and is optimized using `BCE Loss`.

## Evaluation
The project uses **ROC AUC Score** as the primary metric for classification performance, ensuring robust evaluation beyond simple accuracy. Results are saved to `results/yts_hat_neural.csv` in a format suitable for Kaggle-style submissions.
