
# Titanic Survival Prediction Project

This project contains a structured workflow for predicting Titanic passenger survival using machine learning models, along with comprehensive data analysis and result visualization components.

## 📂 Project Structure

```bash
Titanic-Survival-Prediction/
├── Analysis/                   # Data exploration and result visualization
│   ├── data.csv                # Raw Titanic dataset
│   ├── Pre-analysis-EDA.ipynb  # Exploratory Data Analysis (EDA) on raw data
│   └── Result-analysis.ipynb   # Model performance analysis and visualization
│
├── Train/                      # Model training components
│   ├── data.csv                # Raw Titanic dataset
│   ├── decision_tree.py        # Decision Tree model implementation
│   ├── logistic_regression.py  # Logistic Regression model implementation
│   ├── train.ipynb             # Unified notebook for both models
│
├──.ipynb_checkpoints/         # Jupyter auto-save files (ignore)
├── README.md                  # Training documentation
└── requirements.txt           # Dependency list

```

## Key Features

### Model Training (`Train/`)
- **Two Implementations**  
  - `decision_tree.py`: Decision Tree classifier with hyperparameter tuning
  - `logistic_regression.py`: Logistic Regression model with feature normalization
- **Unified Notebook** (`train.ipynb`):  
  Combines both models with interactive visualizations and comparative analysis

### Data Analysis (`Analysis/`)
- **Pre-analysis-EDA.ipynb**:  
  - Missing value handling
  - Feature distributions
  - Survival correlations (age/class/gender)
- **Result-analysis.ipynb**:  
  - Feature importance visualization
  - Model accuracy metrics

## Installation

1. Clone repository:
   ```bash
   git clone https://github.com/yourusername/titanic-survival-prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter for analysis:
   ```bash
   jupyter notebook
   ```

## Usage

### Training Models
```bash
# Run Decision Tree
python Train/decision_tree.py

# Run Logistic Regression
python Train/logistic_regression.py
```

### Interactive Exploration
1. Open `train.ipynb` for model comparison
2. Execute cells to regenerate results

---


