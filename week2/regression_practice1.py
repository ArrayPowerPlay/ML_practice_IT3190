import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# load data of diabetes dataset
diabetes = datasets.load_diabetes()
# data exploration
print("Số chiều dữ liệu input:\n", diabetes.data.shape)
print("Kiểu dữ liệu input:\n", type(datasets.data))
print("Số chiều dữ liệu target:\n", diabetes.target.shape)
print("Kiểu dữ liệu target:\n", type(diabetes.target))

print("5 mẫu dữ liệu đầu tiên:")
print("Input:\n", diabetes.data[:5])
print("Output:\n", diabetes.target[:5])

# training: 362 examples
# test: 80 examples
X = diabetes.data
y = diabetes.target
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

Xtrain = X[:362]
ytrain = y[:362]

Xtest = X[362:]
ytest = y[362:]

# build regression model using sklearn
# setting model with alpha=0.1
regr = linear_model.Ridge(alpha=0.1)

# training model
print()
regr.fit(Xtrain, ytrain)
print("Tham số mô hình tìm được:")
print(regr.coef_)
print(regr.intercept_)

# predict result by model
y_pred = regr.predict(Xtest)

w = regr.coef_
b = regr.intercept_

# predict for the first observation
print()
y_pred_first = np.dot(w, Xtest[0]) + b
print("Giá trị dự đoán cho trường hợp đầu tiên:")
print(y_pred_first)
print("Giá trị thư viện cho trường hợp đầu tiên:")
print(ytest[0])
# predict the first observation using model's prediction function
print("Giá trị dự đoán của model cho trường hợp đầu tiên:")
print(y_pred[0])
print()

# save results
compare_data = pd.DataFrame(data=np.array([
    ytest, y_pred, abs(y_pred - ytest)
]).T, columns=["Thực tế", "Dự đoán", "Lệch"])
print(compare_data)
print()

# evaluate model using RMSE
model_error_first = math.sqrt(mean_squared_error(y_true=ytest, y_pred=y_pred))
print(model_error_first)

# evaluate multiple models with different lambda
_lambda = [0, 0.0001, 0.01, 0.04, 0.05, 0.06, 0.1, 0.5, 1, 5, 10, 20]
model_list = []
error = []

for alpha in _lambda:
    model = linear_model.Ridge(alpha)
    model_list.append(model)
    model.fit(Xtrain, ytrain)
    y_pred = model.predict(Xtest)
    model_error = math.sqrt(mean_squared_error(y_true=ytest, y_pred=y_pred))
    error.append(model_error)

for i in range(len(error)):
    print(f"Đánh giá loss của mô hình sử dụng lambda = {_lambda[i]}")
    print(error[i])

# choose the best model
final_model = model_list[np.argmin(error)]
final_lamba = _lambda[np.argmin(error)]
print()
print("Giá trị lamba tốt nhất: lambda = ", final_lamba)
# prediction of the best model
y_pred_final = final_model.predict(Xtest)

# plot the distribution chart for the result predicted by the linear regression model
print()
import seaborn as sns

df_pred = pd.DataFrame({'Predicted': y_pred_final})
print(df_pred)
print()
print(df_pred.describe())
print()

plt.figure(figsize=(8, 6))
# kde=True means show additional “density estimate” curve on the histogram
sns.histplot(df_pred['Predicted'], kde=True, color='skyblue')
plt.xlabel('Giá trị dự đoán')
plt.ylabel('Tần suất')
plt.savefig('week2/density_prediction.png')
plt.show()


