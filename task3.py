import pandas as pd
import numpy as np
from scipy import stats

# Input Data (15 readings, 3 consecutive missing, 1 outlier at 58)
data = {
    "Hour": range(1, 16),
    "Temperature": [22, 23, np.nan, np.nan, np.nan, 25, 26, 58, 24, 23, 22, 21, 23, 24, 25]
}
df = pd.DataFrame(data)
df["Temp_raw"] = df["Temperature"]

# (a) Stats Before Cleaning
valid_temp = df["Temperature"].dropna()
print("Stats BEFORE cleaning:")
print(f"Mean: {valid_temp.mean():.2f}  Std: {valid_temp.std():.2f}")
print(f"Skewness: {stats.skew(valid_temp):.2f}  Kurtosis: {stats.kurtosis(valid_temp):.2f}")

# (b) Linear Interpolation for Missing Values
df["Temperature"] = df["Temperature"].interpolate(method="linear")
print(f"\nAfter interpolation, missing values: {df['Temperature'].isnull().sum()}")

# (c) Outlier Detection via Z-score (>2) & Replace with Median
z_scores = np.abs(stats.zscore(df["Temperature"]))
outlier_idx = np.where(z_scores > 2)[0]

clean_median = df["Temperature"].median()
for idx in outlier_idx:
    print(f"Z-score(>2) outlier detected at Hour: {df.loc[idx, 'Hour']} (value: {df.loc[idx, 'Temperature']})")
    print(f"Replaced with column median: {clean_median}")
    df.loc[idx, "Temperature"] = clean_median

df["Temp_clean"] = df["Temperature"]

# (d) Z-score Standardization on Cleaned Data
df["Temp_Zscore"] = stats.zscore(df["Temp_clean"])

# Output Comparison
print("\nBefore/After comparison (first 8 rows):")
print(df[["Hour", "Temp_raw", "Temp_clean", "Temp_Zscore"]].head(8))

print("\nStats AFTER cleaning:")
print(f"Mean: {df['Temp_clean'].mean():.2f}  Std: {df['Temp_clean'].std():.2f}")