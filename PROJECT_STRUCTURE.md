# Project Structure Guide

*How I organized this mess (and why future me will thank present me)*

## Folder Layout

```
kaggle-calorie-prediction/
│
├── README.md                    # The main story (start here!)
├── Predict_Calories_Expenditure.ipynb  # Complete authentic competition notebook
├── requirements.txt             # All the Python packages needed
├── config.yaml                  # Project configuration and settings  
├── PROJECT_STRUCTURE.md         # This file (organization guide)
│
├── src/                         # Clean, reusable code
│   ├── __init__.py             # Package initialization  
│   ├── data_preprocessing.py   # Data loading, cleaning, train/test split
│   ├── feature_engineering.py  # BMI, heart rate ratios, all feature creation
│   ├── models.py              # All ML models, training & evaluation
│   ├── utils.py               # Helper functions (rmsle, evaluation, submission)
│   └── visualization.py       # Plotting functions and correlation analysis
│
├── results/                     # Competition results and outputs
│   ├── kaggle_ranking.png      # Screenshot of 1605/4318 ranking!
│   └── submission_history.png  # Progression through different approaches
│
└── data/                       # Data information
    └── README.md              # How to obtain the competition data
```
