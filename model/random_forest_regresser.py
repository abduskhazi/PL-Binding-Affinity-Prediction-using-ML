from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from data_bakery import bake_train_Xy
import reproducibility
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np

# Firstly ...
reproducibility.make_program_reproducible()

# Obtain data for our random forest regressor.
X, y = bake_train_Xy()
print("X.shape =", X.shape)
print("y.shape =", y.shape)

X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2)

regressor = RandomForestRegressor(n_estimators=100, oob_score = True, n_jobs=-1)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_validate)
print("oob score = ", regressor.oob_score_, ", Validation r2 score = ", r2_score(y_validate, y_pred))

#Plotting to visualize the accuracy of our model.
fig = plt.figure()
plt.plot(y_validate, y_pred, '.')
plt.plot(range(2,14), range(2,14), '--')
fig.savefig('temp_validate.png', dpi=fig.dpi)

fig = plt.figure()
plt.plot(y_train, regressor.predict(X_train), '.')
plt.plot(range(2,14), range(2,14), '--')
fig.savefig('temp_train.png', dpi=fig.dpi)

# TO check the importance protein features in our model -
# We shuffle all protein feature columns one by one. Then we check the R^2 score.
# Similar process is repeated for the ligand features.
X_validate_backup = np.copy(X_validate)

for i in range(55):
    np.random.shuffle(X_validate[:,i])
print("r2 score (shuffled protein columns) =", r2_score(y_validate, regressor.predict(X_validate)))

X_validate = np.copy(X_validate_backup)

for i in range(55, X.shape[1]):
    np.random.shuffle(X_validate[:,i])
print("r2 score (shuffled ligand columns) =", r2_score(y_validate, regressor.predict(X_validate)))

sorted_indices = np.argsort(regressor.feature_importances_)
print(sorted_indices)

from sklearn.inspection import permutation_importance
X_validate = np.copy(X_validate_backup)
impt = permutation_importance(regressor, X_validate, y_validate, n_repeats=30, n_jobs=-1)
print(impt.importances_mean.argsort()[::-1])
