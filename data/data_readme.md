# Competition Data

## How to Get the Data

The dataset for this project comes from **Kaggle Playground Series S5E5 - Predict Calorie Expenditure**.

### Option 1: Download from Kaggle (Recommended)
1. Visit the competition page: https://www.kaggle.com/competitions/playground-series-s5e5
2. Download the following files:
   - `train.csv` - Training data with features and target calories
   - `test.csv` - Test data for predictions  
   - `sample_submission.csv` - Submission format

### Option 2: Google Drive (As used in notebook)
The original notebook downloads data from Google Drive using:
```python
file_id = '1S8YoUUNWUWXebWfssg9wHRNB-y1s_ZNh'
!gdown --id {file_id} -O downloaded_file.zip
```

## Data Description

### Training Data (`train.csv`)
- **Rows:** ~15,000 samples
- **Features:** 8 columns
  - `Age` - Age of person (years)
  - `Sex` - Gender (Male/Female) 
  - `Height` - Height in centimeters
  - `Weight` - Weight in kilograms
  - `Duration` - Exercise duration (minutes)
  - `Heart_Rate` - Heart rate during exercise (BPM)
  - `Body_Temp` - Body temperature during exercise (°C)
  - `Calories` - **TARGET** - Calories burned during exercise

### Test Data (`test.csv`) 
- **Rows:** ~10,000 samples
- **Features:** Same as training data except no `Calories` column
- Includes `id` column for submission

### Sample Submission (`sample_submission.csv`)
- Format for Kaggle submission
- Columns: `id`, `Calories`

## Data Quality Notes

From my exploration, this dataset is:
- Very clean (no missing values)
- Well-structured for learning
- Perfect for practicing ML techniques
