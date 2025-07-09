# Predicting Calorie Burn - My First Kaggle Competition! 

**Competition:** Playground Series S5E5 - Predict Calorie Expenditure  
**My Result:** **Ranked 1605 out of 4318 (Top 37%)** with final score of 0.05951  
**What I learned:** SO much about machine learning and why my Apple Watch sometimes seems way off

![Competition Results](results/kaggle_ranking.png)

## What's This All About?

Ever wondered how your fitness tracker calculates those calories you burned? This Kaggle competition challenged us to predict how many calories people burn during exercise based on their personal info and workout details.

Basically, I got to play data scientist and figure out the math behind those numbers on your phone!

## The Challenge

Given data about people's:
- Age, gender, height, weight  
- Exercise intensity (how hard they're working out)
- Duration of exercise
- Heart rate info

**Goal:** Predict exactly how many calories they burned 

## My Approach (The Journey)

### 1. **Getting to Know the Data** 
- First thing I did was make a bunch of charts to see what the data actually looked like
- Found some interesting patterns! Like obviously taller/heavier people burn more calories
- Also discovered that exercise intensity matters WAY more than I expected

### 2. **Feature Engineering** (Fancy term for "making the data better")
- **BMI Calculation:** `BMI = weight / (height/100)²` - turned out to be super important!
- **Heart Rate Ratios:** Created `Heart_Rate / Duration` to capture exercise intensity
- **Interaction Features:** `Heart_Rate × Duration` squared for non-linear relationships
- **Body Temperature Ratios:** `Body_Temp / Heart_Rate` (though this was less useful)

### 3. **Model Progression** 
My journey from simple to sophisticated:
1. **Linear Regression:** Great baseline, achieved decent results with feature scaling
2. **Decision Tree:** Better performance, could capture non-linear patterns
3. **Random Forest:** Big improvement! This was my breakthrough model
4. **Grid Search:** Fine-tuned Random Forest hyperparameters (max_features optimization)
5. **Ensemble Learning:** Combined Ridge, Random Forest, and SVR with VotingRegressor

### 4. **What Actually Worked Best**
- **Random Forest with engineered features** gave me the winning score
- **BMI was crucial** - much more predictive than height/weight separately
- **Heart rate intensity ratios** captured the exercise effort better than raw heart rate
- **Feature scaling** helped the linear models but wasn't as critical for tree-based models

## Key Insights 

1. **BMI is a game-changer** - Much more predictive than using height and weight separately
2. **Heart rate intensity matters most** - Not just the raw heart rate, but how it relates to exercise duration
3. **Feature engineering beats model complexity** - Simple Random Forest with good features outperformed complex ensembles
4. **Non-linear relationships exist** - Tree-based models significantly outperformed linear approaches
5. **Data quality is evident** - The clean correlations suggested this was well-designed synthetic data for learning

## Technical Stuff

- **Best Model:** Random Forest with engineered features
- **Final Score:** 0.05951 (RMSE)
- **Evaluation Metrics:** RMSE and RMSLE for robust performance measurement
- **Cross-validation:** 10-fold CV to ensure model stability
- **Feature Engineering:** 4 new features (BMI, ratios, interactions)
- **Key Libraries:** scikit-learn, pandas, numpy, matplotlib
- **Ensemble Approach:** VotingRegressor combining Ridge, Random Forest, and SVR

## What I Learned

This was my first "real" machine learning project and honestly? It was harder than I thought but way more fun! 

**Technical skills:**
- How to engineer meaningful features (BMI was my breakthrough moment!)
- Why ensemble methods can be powerful but aren't always necessary
- Proper evaluation with multiple metrics (RMSE vs RMSLE)
- Grid search for hyperparameter optimization without overfitting
- The importance of data exploration and correlation analysis

**Life lessons:**
- Ranking in top 37% on my first competition feels amazing! 🎉
- Sometimes simple solutions (Random Forest) work better than complex ones (ensemble)
- Feature engineering is where the real magic happens
- My fitness tracker probably uses similar algorithms to what I just built
- Data science is like solving puzzles, but with numbers

## Files in This Repo

- **`Predict_Calories_Expenditure.ipynb`** - My complete competition notebook (5967 lines of authentic ML work!)
- **`src/`** - Clean, reusable code modules extracted from the exploration:
  - `feature_engineering.py` - BMI calculation and heart rate features that made the difference
  - `models.py` - Random Forest training and ensemble experiments  
  - `data_preprocessing.py` - Data loading and preprocessing pipeline
  - `utils.py` - Evaluation functions and submission generation
  - `visualization.py` - All the plotting and correlation analysis
- **`results/`** - My competition results and ranking screenshots
  - Final ranking: **1605 out of 4318 (Top 37%)**
  - Best score: **0.05951 RMSE**
- **`data/`** - Information about obtaining the competition data

## Next Steps

- Want to try some deep learning approaches (neural networks!)
- Curious about time-series data - like predicting calories for ongoing workouts
- Maybe collect my own data from friends/family to see how the model performs IRL

## Shoutouts

Thanks to the Kaggle community for all the shared notebooks and ideas! Also big thanks to my dad for helping me debug when I was stuck 🙏

---

*P.S. - Still not sure why my Apple Watch thinks I burned 500 calories walking to the kitchen, but at least now I understand the math behind the madness!*
