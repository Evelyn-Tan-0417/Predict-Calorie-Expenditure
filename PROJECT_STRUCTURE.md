# Project Structure Guide 📁

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

## File Naming Conventions

### Notebooks
- **Numbers first:** `01_`, `02_`, etc. so they stay in order
- **Descriptive names:** You'll forget what "notebook1.ipynb" does in a week
- **Underscores not spaces:** Because command line and GitHub URLs

### Data Files
- **`_raw`** suffix for original data
- **`_processed`** for cleaned data
- **`_final`** for the version used in final model
- **Date stamps:** `model_results_2025-06-30.csv` when you have multiple versions

### Model Files
- Include model type: `xgboost_v1.pkl`
- Include performance: `rf_0.85_rmse.pkl`
- Version numbers when iterating: `ensemble_v3_final.pkl`

## How I Actually Used This Structure

### Phase 1: Exploration (Week 1)
- Started in `notebooks/01_data_exploration.ipynb`
- Saved interesting findings in `docs/data_exploration_notes.md`
- Raw data stayed untouched in `data/raw/`

### Phase 2: Feature Engineering (Week 2)  
- Built features in `notebooks/02_feature_engineering.ipynb`
- Moved reusable functions to `src/feature_engineering.py`
- Saved processed data to `data/processed/`

### Phase 3: Modeling (Week 3-4)
- Tried different models in separate notebooks (`03_`, `04_`)
- Clean model code went into `src/models.py`
- Saved best models in `models/` folder

### Phase 4: Final Push (Competition deadline!)
- Combined everything in `notebooks/06_final_submission.ipynb`
- Generated submission file in `results/submissions/`
- Updated README with final results

## Pro Tips I Learned the Hard Way

1. **Version control your data processing:** If you mess up your processed data, you want to recreate it easily
2. **Comment your notebook cells:** "Why did I do this transformation?" - future you will ask
3. **Save model objects:** Training XGBoost takes forever, don't do it twice
4. **Backup submissions:** Submit early and often, keep all versions
5. **Document weird decisions:** "Why did I drop this feature?" Better write it down!

## What I'd Do Differently Next Time

- Start with the clean `src/` structure earlier instead of only notebooks
- Use better logging instead of print statements everywhere  
- Set up automated testing for my data processing functions
- Create a proper pipeline script instead of running notebooks manually

But hey, this was my first rodeo! 🤠