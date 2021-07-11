from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import data_bakery as bakery
import reproducibility
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
import sys

# Firstly ...
ExecutionID = None
if len(sys.argv) > 1:
    ExecutionID = int(sys.argv[1])
ExecutionID = reproducibility.reproduce(ExecutionID)


# Obtain data for our random forest regressor.
X, y, features = bakery.bake_train_Xy()
print("X.shape =", X.shape)
print("y.shape =", y.shape)

X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2)

print("Fitting the Random Forest Regressor...")
regressor = RandomForestRegressor(n_estimators=100, oob_score = True, n_jobs=-1)
regressor.fit(X_train, y_train)
print("Fitting completed.")

y_pred = regressor.predict(X_validate)
print("oob score = ", regressor.oob_score_, ", Validation r2 score = ", r2_score(y_validate, y_pred))

#Plotting to visualize the accuracy of our model.
fig = plt.figure()
plt.scatter(y_validate, y_pred, 1)
plt.xlabel("y_validate")
plt.ylabel("y_validate_pred")
plt.title("Execution ID = " + str(ExecutionID))
plt.plot(range(2,14), range(2,14), '--')
fig.savefig('accuracy_validate.png', dpi=fig.dpi)

fig = plt.figure()
plt.scatter(y_train, regressor.predict(X_train), 1)
plt.xlabel("y_train")
plt.ylabel("y_train_pred")
plt.title("Execution ID = " + str(ExecutionID))
plt.plot(range(2,14), range(2,14), '--')
fig.savefig('accuracy_train.png', dpi=fig.dpi)

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

# Getting the original X_validate for further checking
X_validate = np.copy(X_validate_backup)

# Gini Importance (Sorting in the decreasing order of importance)
impt_indices = regressor.feature_importances_.argsort()[::-1]

print("Gini impt ->", [regressor.feature_importances_[i] for i in impt_indices[:5]])

fig = plt.figure()
fig.subplots_adjust(left=0.35)
ax = fig.add_subplot(111) #fig.add_axes([0,0,1,1])
feature_names = [features[i] for i in impt_indices[:15]]
importances = [regressor.feature_importances_[i] for i in impt_indices[:15]]
y_pos = np.arange(len(importances))
ax.barh(y_pos, importances, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(feature_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel("Execution ID = " + str(ExecutionID))
ax.set_title("Gini Importances of features.")
fig.savefig('Gini_importance.png', dpi=fig.dpi)

print("Important features based on decrease in entropy (Gini Importance)")
for i in impt_indices[:30]:
    print(features[i])

print("Calculating importance of features using permuation importance...")
# Permutation Importance
from sklearn.inspection import permutation_importance
impt = permutation_importance(regressor, X_validate, y_validate, n_repeats=30, n_jobs=-1)
impt_indices = impt.importances_mean.argsort()[::-1]

print("Permuation impt ->", [impt.importances_mean[i] for i in impt_indices[:5]])

fig = plt.figure()
fig.subplots_adjust(left=0.35)
ax = fig.add_subplot(111) #fig.add_axes([0,0,1,1])
feature_names = [features[i] for i in impt_indices[:15]]
importances = [impt.importances_mean[i] for i in impt_indices[:15]]
y_pos = np.arange(len(importances))
ax.barh(y_pos, importances, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(feature_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel("Execution ID = " + str(ExecutionID))
ax.set_title("Permutation Importances of features.")
fig.savefig('Permutation_importance.png', dpi=fig.dpi)

print("Important features based on permutation")
for i in impt_indices[:30]:
    print(features[i])

print("Program finished.")
