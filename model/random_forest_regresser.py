from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from data_bakery import bake_train_Xy
import random

# Obtain data for our random forest regressor
X, y = bake_train_Xy()
print("X.shape =", X.shape)
print("y.shape =", y.shape)

#The reason for having a seed is to control the randomness in our models
seed = random.randint(0,2**32)
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=seed)

regressor = RandomForestRegressor(n_estimators=10)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_validate)
print("Validation r2 score = ", r2_score(y_validate, y_pred))

#Plotting to visualize the accuracy of our model.
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(y_validate, y_pred, '.')
plt.plot(range(2,14), range(2,14), '--')
fig.savefig('temp.png', dpi=fig.dpi)

fig = plt.figure()
plt.plot(y_train, regressor.predict(X_train), '.')
plt.plot(range(2,14), range(2,14), '--')
fig.savefig('temp_train.png', dpi=fig.dpi)
