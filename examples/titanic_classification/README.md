# Example: Titanic (classification, mixed types, missing data)

Predicts `Survived` (binary) from passenger attributes.

- **Data**: `data/titanic.csv` — the standard Titanic passenger manifest (891 rows), vendored from the public [datasciencedojo/datasets](https://github.com/datasciencedojo/datasets) repository.
- **Run it**: `python examples/titanic_classification/run_example.py`
- **What the pipeline does automatically here**: median-imputes `Age`, mode-imputes `Embarked`, frequency-encodes the high-cardinality `Cabin`/`Ticket` columns, drops the free-text `Name` column, one-hot encodes `Sex`, then compares 3 classification models and runs a Logit diagnostic baseline (this is the multi-type/messy-real-world-data example: mixed numeric, categorical, near-unique-text and heavily-missing columns all handled without dataset-specific code).

## Results (best of 3 compared models)

| Model | Accuracy | F1 (weighted) | ROC-AUC |
|---|---|---|---|
| **Logistic Regression (best)** | **0.805** | **0.804** | **0.838** |
