import pandas as pd
import numpy as np

# load data
df = pd.read_csv('homework/week2/elantra.csv')
print(df.tail(10))
print()
print("Data dimension:")
print(df.shape)
# rearrange the order of data rows by month/year
df = df.sort_values(by=['Year', 'Month']).reset_index(drop=True)

# visualize the sales from time to time
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 6))
plt.plot(df['ElantraSales'].values)
plt.xlabel('Time Index')
plt.ylabel('Sales')
# plt.show()

# create features for our model
print()
numeric_features = df.columns.drop(['Month', 'Year', 'ElantraSales'])
print("Numeric features:")
print(numeric_features)

# build data for training and testing
df_train = df[df['Year'] < 2013]
df_test = df[df['Year'] >= 2013]

y_train = df_train['ElantraSales'].values
y_test = df_test['ElantraSales'].values

# feature scaling (mean = 0 and standard deviation = 1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(df_train[numeric_features])

X_train = scaler.transform(df_train[numeric_features])
X_test = scaler.transform(df_test[numeric_features])

# build model
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# evaluate model
from sklearn.metrics import mean_squared_error

# build relative error for business task evaluation
def relative_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred).astype(float) / y_true) * 100

y_pred = model.predict(X_test)
print()
print('RMSE loss: {:.2f}'.format(np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))))
print('Mean relative error: {:.2f}%'.format(relative_error(y_true=y_test, y_pred=y_pred)))

# compare y_pred and y_test
import matplotlib.pyplot as plt
print()
compare_data = pd.DataFrame({'Predicted': y_pred,
                             'True values': y_test,
                             'Difference': np.abs(y_test - y_pred)})
print(compare_data)
print()

# draw plot to compare y_pred and y_test
plt.figure(figsize=(9, 6))
plt.plot(y_test, label=' values', color='blue')
plt.plot(y_pred, label='Predicted values', color='red')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
# plt.savefig('homework/week2/compare_before.png')
# plt.show()

# use non numeric feature for our model using one-hot encoding 
# (here we only use month feature)
month_feature_train = pd.get_dummies(df_train['Month'], prefix='Month')

month_feature_test = pd.get_dummies(df_test['Month'], prefix='Month')

# synchronize between training set and test set
month_feature_test = month_feature_test.reindex(columns=month_feature_train.columns, fill_value=0)

# convert to numpy array
month_feature_train = month_feature_train.values
month_feature_test = month_feature_test.values

X_train = np.concatenate((X_train, month_feature_train), axis=1)
X_test = np.concatenate((X_test, month_feature_test), axis=1)

# evaluate fixed model
print()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('RMSE: {:.2f}'.format(np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))))
print('Mean relative error: {:.2f}%'.format(relative_error(y_true=y_test, y_pred=y_pred)))

# # plot difference between predicted and true sales with fixed model
print()
plt.figure(figsize=(9, 6))
plt.plot(y_test, label='True values', color='blue')
plt.plot(y_pred, label='Predicted values', color='red')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.savefig('homework/week2/compare_after.png')
plt.show()