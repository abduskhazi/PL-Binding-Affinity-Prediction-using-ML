from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from data_bakery import bake_train_Xy
import random
import numpy as np

# Obtain data for our random forest regressor
X, y = bake_train_Xy()
print("X.shape =", X.shape)
print("y.shape =", y.shape)

#The reason for having a seed is to control the randomness in our models
seed = random.randint(0,2**32)
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=seed)

regressor = RandomForestRegressor(n_estimators=100, oob_score = True, n_jobs=-1)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_validate)
print("oob score = ", regressor.oob_score_, ", Validation r2 score = ", r2_score(y_validate, y_pred))

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

# TO check how much importance prtein features have in our trained model -
# We shuffle all the columns of proteins
X_validate_backup = np.copy(X_validate)

print("Shuffling protein columns")
for i in range(55):
    np.random.shuffle(X_validate[:,i])
print("r2 score (shuffled protein columsn) =", r2_score(y_validate, regressor.predict(X_validate)))

X_validate = np.copy(X_validate_backup)
print("r2 score (backup value) =", r2_score(y_validate, regressor.predict(X_validate)))

print("Shuffling ligand columns")
for i in range(55, 457):
    np.random.shuffle(X_validate[:,i])
print("r2 score (shuffled ligand columns) =", r2_score(y_validate, regressor.predict(X_validate)))

sorted_indices = np.argsort(regressor.feature_importances_)
print(sorted_indices)

from sklearn.inspection import permutation_importance
X_validate = np.copy(X_validate_backup)
impt = permutation_importance(regressor, X_validate, y_validate, n_repeats=30, n_jobs=-1, random_state=seed)
print(impt.importances_mean.argsort()[::-1])
