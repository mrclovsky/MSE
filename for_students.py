import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 andtheta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: standardization

x_train_copy = x_train
y_train_copy = y_train
x_train = (x_train - np.average(x_train))/np.std(x_train)
y_train = (y_train - np.average(y_train))/np.std(y_train)
x_test = (x_test - np.average(x_train_copy))/np.std(x_train_copy)
y_test = (y_test - np.average(y_train_copy))/np.std(y_train_copy)

# TODO: calculate closed-form solution

x_dotted = np.c_[np.ones(len(x_train)), x_train]
inverted = np.linalg.inv(x_dotted.T.dot(x_dotted))
theta_best = inverted.dot(x_dotted.T).dot(y_train)
print("Theta best:", theta_best)

# TODO: calculate error

x_test_dotted = np.c_[np.ones(len(x_test)), x_test]
MSE = 0
for i in range(len(x_test_dotted)):
    prediction = theta_best.dot(x_test_dotted[i])
    MSE += (prediction - y_test[i])**2

MSE /= len(x_test)
print("Mse: ", MSE)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()


# TODO: calculate theta using Batch Gradient Descent

x_dotted = np.c_[np.ones(len(x_train)), x_train]
y_column = np.c_[y_train]
theta_best = np.random.rand(2,1)
learning_rate = 0.1

for i in range(100):
    compare = x_dotted.dot(theta_best) - y_column
    matrix = x_dotted.T.dot(compare)
    gradient_MSE = 2 / len(x_dotted) * matrix
    theta_best = theta_best - learning_rate*gradient_MSE

print("Theta best:", theta_best)

# TODO: calculate error

x_test_dotted = np.c_[np.ones(len(x_test)), x_test]
MSE = 0
for i in range(len(x_test_dotted)):
    prediction = theta_best.T.dot(x_test_dotted[i])
    MSE += (prediction - y_test[i])**2

MSE /= len(x_test)
print("Mse: ", MSE)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()