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

# consider the correlation of values ​​of pairs of numeric field
print()
sns.pairplot(df)
plt.savefig('week1/correlation_matrix.png')
plt.show()

num_vars = ['bath', 'balcony', 'price']
sns.heatmap(df[num_vars].corr(), cmap='coolwarm', annot=True)
plt.savefig('week1/correlation_map.png')
plt.show()

# preprocessing data
# processing null/nan values
print()
print("Probability of null-rows in each field")
print(df.isnull().mean() * 100)

# eliminate 'society' field (too many null rows)
df2 = df.drop(columns='society')
# replace null-value in 'balcony' field by its mean
df2['balcony'] = df2['balcony'].fillna(df['balcony'].mean())
# drop rows which have nan values
df3 = df2.dropna()

print()
print("Probability of null values in each field")
print(df3.isnull().sum())

# analyze 'total_sqft' field
print()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df3['total_sqft'].value_counts())

print()
total_sqft_float = []
for value in df3['total_sqft']:
    try:
        total_sqft_float.append(float(value))
    except:
        try:
            tmp = []
            tmp = value.split('-')
            num = (float(tmp[0]) + float(tmp[-1]))/2
            total_sqft_float.append(num)
        except:
            # add null values
            total_sqft_float.append(np.nan)

df4 = df3.join(pd.DataFrame({'total_sqft_float': total_sqft_float}))
df4 = df4.drop(columns='total_sqft')
df4 = df4.dropna()
df4 = df4.reset_index(drop=True)
# check information of each row after analyze 'total_sqft_float' field
print(df4.info())

# analyze feature 'size'
print()
print(df4['size'].value_counts())
# convert field's name value from category to numberic
size_int = []
for size in df4['size']:
    try:
        tmp = []
        tmp = size.split(' ')
        size_int.append(int(tmp[0]))
    except:
        size_int.append(np.nan)

df5 = df4.join(pd.DataFrame({'bhk': size_int}))
df5 = df5.reset_index(drop=True)

# detect outliers and eliminate them
df5_tmp = df5[df5['total_sqft_float'] < 3500]
sns.boxplot(x=df5_tmp['total_sqft_float'])
plt.show()

# eliminate outliers
df6 = df5[(df5['total_sqft_float'] > 500) & (df5['total_sqft_float'] < 2000)]
sns.boxplot(x=df6['total_sqft_float'])
plt.show()

# add 'price_per_sqft' field
df7 = df6.reset_index(drop=True)
df7['price_per_sqft'] = df7['price'] * 100000 / df7['total_sqft_float']
print()
print(df7.head())

# use boxplot to contemplate distribution of data, detect outliers
vars = ['price', ]