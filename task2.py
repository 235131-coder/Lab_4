import pandas as pd
import numpy as np
from scipy.stats import zscore

# Input Data
data = {
    "Day": range(1, 11),
    "Revenue": [45, 52, np.nan, 48, 390, 55, np.nan, 50, 47, 53]
}
df = pd.DataFrame(data)

# (a) Missing values check & fill with Median
print("Missing values in Revenue:", df["Revenue"].isnull().sum())
median_revenue = df["Revenue"].median()
df["Revenue"] = df["Revenue"].fillna(median_revenue)

# (b) Min-Max Normalization & Z-Score Standardization
df["Revenue_MinMax"] = (df["Revenue"] - df["Revenue"].min()) / (df["Revenue"].max() - df["Revenue"].min())
df["Revenue_Zscore"] = zscore(df["Revenue"])

# (c) IQR Outlier Detection
Q1 = df["Revenue"].quantile(0.25)
Q3 = df["Revenue"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df["Outlier"] = (df["Revenue"] < lower_bound) | (df["Revenue"] > upper_bound)
outlier_days = df[df["Outlier"]]["Day"].tolist()

# (d) Display Final Output
print("\n--- Processed Retail Sales Table ---")
print(df[["Day", "Revenue", "Revenue_MinMax", "Revenue_Zscore", "Outlier"]])
print(f"\nIQR outliers detected on day(s): {outlier_days}")