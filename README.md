![ml](https://github.com/user-attachments/assets/3b625437-0ab1-4afb-8437-2cc26824a022)
# Modular Machine Learning Pipeline

Welcome to my first project! This repository demonstrates a **modular machine learning pipeline** built in Python. It is designed to simplify the process of ingesting, training, and evaluating machine learning models, while providing flexibility to tune hyperparameters using YAML configuration files.

---

## Features

- **Modular Design**: Each stage of the ML process (data ingestion, training, evaluation) is encapsulated in its own module.
- **Hyperparameter Tuning**: Easily configure and experiment with hyperparameters.
- **Custom Exception Handling**: Robust error management ensures smooth execution and helpful debugging.
- **Automation**: Everything is triggered through the `data_injection.py` file, which acts as the entry point.

---

## Project Structure

```
project_root/
|-- src
| |- components/
|   |-- data_ingestion_reading.py   # Handles data loading and preprocessing
|   |-- data_transformation.py    # Trains machine learning models
|   |-- model_training.py # Evaluates model performance
|   |- utils/
|   |-- exception.py        # Custom exception handling
|   |-- logger.py           # Logging utility
|-- artifacts/             # Stores processed data and model artifacts
```

## Usage

1. **Trigger the pipeline**:
   Run the `data_injection_reading.py` file to start the pipeline:
   ```bash
   python data_injection.py
   ```

2. **Modify Hyperparameters**:
   Modify the Search_space Parameter in model_trainer.py

3. **Logs and Artifacts**:
   - Logs are stored in the `logs/` folder for debugging and tracking.
   - Processed data and trained model artifacts are saved in the `artifacts/` folder.

---

## Modules Overview

### Data Ingestion
- Reads raw data from CSV, JSON, or other sources.
- Performs preprocessing, such as missing value handling and feature encoding.

### Model Trainer
- Trains various machine learning models (e.g., Linear Regression, Random Forest).
- Supports hyperparameter tuning via YAML configuration.

### Model Evaluation
- Computes metrics such as accuracy, precision, recall, and F1-score.
- Provides insights into model performance.

### Exception Handling
- Custom `Exception` class ensures meaningful error messages for smooth debugging.

---
### This is just my first try to making a modular project to run a machine learning model. This served as my reference for other projects. Here I revised python, solved issues of imports and applying python exports, learned about creating artifacts to store model (though not commited here).
