# TRAGIC (Time seRies Analysis using Graph attentIon networks Classifier)

A  deep learning framework that uses Graph Attention Networks (GAT) for time series classification. This project implements a novel approach to transform time series data into graph structures, enabling  geometric deep learning techniques for analysis and classification.

## Core Architecture

### Graph Representation
- Time series are converted into graphs where:
  - Each timestep becomes a node
  - Edges connect neighboring timesteps within a window
  - Node features include the time series value and normalized position
  - Self-loops are added to allow nodes to maintain their own information

### Main Model Components

The `GNNTimeSeriesClassifier` architecture includes:
- Input projection layer
- Two GAT layers with attention mechanisms
- Layer normalization for stability
- Global mean pooling for graph-level representations
- Final classification layers

## Key Features

### Robust Training
- K-fold cross validation for reliable evaluation
- Early stopping to prevent overfitting
- Class weighting for imbalanced datasets
- Per-fold data normalization to prevent leakage

### Visualization Capabilities
- Attention weights visualization
- Saliency maps
- ROC curves with confidence intervals
- Confusion matrices
- Time series examples

### Interpretability Tools
- Attention visualization for important timesteps
- Saliency analysis for feature influence
- Attention weight distributions
- Average attention profiles across samples

### Comprehensive Evaluation
- Metrics:
  - Accuracy
  - Balanced accuracy
  - Matthews Correlation Coefficient (MCC)
  - AUC-ROC (for binary classification)
  - Per-class metrics
  - Bootstrap confidence intervals

### Data Processing
- Support for UCR/UEA time series datasets
- Graph structure conversion
- Proper train/validation/test splitting
- Leakage-free data normalization

### Output & Logging
Saves detailed results including:
- Metrics with confidence intervals
- Visualizations
- Per-fold performance
- Classification reports
- Summary statistics (CSV/JSON)

## Project Structure
- `*.ipynb`: Jupyter notebooks containing experiments
- `plots/`: Generated visualizations
- `results/`: Experimental results and metrics

## Requirements
See `requirements.txt` for detailed package dependencies.

## Usage
1. Clone the repository
```bash
git clone https://github.com/salilp42/TRAGIC.git
cd TRAGIC
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run experiments through Jupyter notebooks

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
