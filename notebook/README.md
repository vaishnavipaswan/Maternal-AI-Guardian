# Maternal AI Guardian -  Model Notebook

This repository contains the Jupyter notebook `maternal_ai_guardian_final_model.ipynb`, which demonstrates the end-to-end workflow for maternal health risk prediction and disease diagnosis using machine learning.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Notebook Features](#notebook-features)
- [Getting Started](#getting-started)
- [How to Use](#how-to-use)
- [Results & Outputs](#results--outputs)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The notebook guides you through:
- Loading and exploring maternal health data
- Understanding and preprocessing features
- Building, training, and evaluating risk prediction models
- Identifying possible maternal diseases based on input features

The ultimate goal is to enable AI-assisted assessment of maternal health risks, supporting better clinical decision-making and early intervention.

## Dataset

The dataset used is `maternal_risk_with_disease_diagnosis.csv`. It contains 1,014 samples with the following features:

- **Demographics & Vitals**: Age, Heart Rate, Body Temperature, Blood Pressure (Systolic/Diastolic), Blood Sugar (BS)
- **Symptoms & Conditions**: Swelling, Nausea, Abdominal Pain, Vaginal Bleeding, Stress Level, Depression Symptoms, Sleep Quality, Diabetes, Anemia, Prior Miscarriage
- **Labels**: 
    - `RiskLevelBinary` (High/Low risk)
    - `PossibleDiseases` (Comma-separated possible maternal diseases)

## Notebook Features

- **Data Loading & Exploration**: View head of the dataset, analyze data types, missing values, and descriptive statistics.
- **Data Cleaning**: Handles missing values and ensures data quality.
- **Feature Engineering**: Prepares data for modeling.
- **Model Training**: Implements and evaluates machine learning models for risk prediction.
- **Disease Diagnosis**: Suggests possible diseases based on clinical features.
- **Visualization**: (Optional) Plots and charts to understand data distribution and model performance.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook or Google Colab
- Required Python libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, etc.

### Installation

Clone the repository:
```bash
git clone https://github.com/vaishnavipaswan/Maternal-AI-Guardian.git
cd Maternal-AI-Guardian/notebook
```

Install dependencies:
```bash
pip install -r requirements.txt
```
*(or install packages manually as needed)*

### Running the Notebook

Open the notebook in Jupyter or Google Colab:
```bash
jupyter notebook maternal_ai_guardian_final_model.ipynb
```
Or upload directly to [Google Colab](https://colab.research.google.com/).

## How to Use

1. Download or clone this repository.
2. Place the dataset CSV file (`maternal_risk_with_disease_diagnosis.csv`) in the same directory as the notebook or update the notebook path as needed.
3. Run the notebook cells sequentially, following the instructions and comments.
4. Review the results, visualizations, and model outputs.

## Results & Outputs

- **Risk Classification**: Predicts maternal health risk as High or Low.
- **Disease Prediction**: Suggests possible diseases based on patient features.
- **Model Evaluation**: Shows accuracy, confusion matrix, and other metrics.

## Requirements

- pandas
- numpy
- scikit-learn
- matplotlib
- (Optional) seaborn

Install them with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Contributing

Contributions, suggestions, and improvements are welcome! Please open issues or pull requests for any changes.

## License

This project is licensed under the MIT License.

---

*For questions, contact [vaishnavipaswan](https://github.com/vaishnavipaswan).*
