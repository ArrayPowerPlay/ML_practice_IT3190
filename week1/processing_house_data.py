import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data
path = 'week1/Bengaluru_House_Data.csv'
df_raw = pd.read_csv(path, delimiter=',')
print("Data dimension:")
print(df_raw.shape)

# review first 5 samples
print()
print(df_raw.head())

# Exploratory Data Analysis (EDA)
print()
df = df_raw.copy()
print(df.info())
print()
print(df.describe())

# counts the unique values ​​of each field and their occurrences
def value_count(df):
    for var in df.columns:
        print(df[var].value_counts())
        print('------------------------')

print()
value_count(df)

# .........
sns.pairplot(df)