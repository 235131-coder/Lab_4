import pandas as pd
import numpy as np
from scipy import stats

# Input data
scores = pd.Series([72, 65, 88, np.nan, 54, 91, 76, np.nan, 83, 69])

# (a) Display describe statistics
print("--- Descriptive Statistics ---")
print(scores.describe())

# (b) Skewness & Kurtosis (after dropping NaN)
valid_scores = scores.dropna()
print(f"\nSkewness: {stats.skew(valid_scores):.4f}")
print(f"Kurtosis: {stats.kurtosis(valid_scores):.4f}")

# (c) Missing values count
missing_count = scores.isnull().sum()
print(f"\nMissing values: {missing_count}")

# (d) Fill missing values with column mean
mean_val = scores.mean()
filled_scores = scores.fillna(mean_val)
print("\nFilled scores:")
print(filled_scores)