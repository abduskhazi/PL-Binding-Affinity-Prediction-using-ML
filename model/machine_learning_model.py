import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from data_bakery import bake_train_Xy
import reproducibility

# Firstly ...
ExecutionID = None
if len(sys.argv) > 1:
    ExecutionID = int(sys.argv[1])
ExecutionID = reproducibility.reproduce(ExecutionID)


X, y, features = bake_train_Xy()

print("X.shape =", X.shape)
print("y.shape =", y.shape)

##################################################################################
# Reporting Linear Regression accuracy with all features included (R^2 score)
##################################################################################

import random
seed = random.randint(0,2**32)

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.10, random_state=seed)

reg = LinearRegression().fit(X_train, y_train)
print("With all features, linear regression score = ", -reg.score(X_train, y_train))

##################################################################################

def linear_regression_score(population, X, y):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures

    # Converting index to actual bit strings
    temp = []
    for p in population:
        p_bin = [0] * X.shape[1]
        for index in p:
            p_bin[index] = 1
        temp += [p_bin]
    population = temp

    population = np.asarray(population)
    population = population[:, np.newaxis, :]

    # Use the same train set for population

    score_list = []
    i = 1
    for feature_selection in population:
        # Use broadcasting for the selection of columns
        X_local = X * feature_selection

        idx = np.argwhere(np.all(X_local[..., :] == 0, axis=0))
        X_local = np.delete(X_local, idx, axis=1)

        print("Finished - ", i, ", Features = ", np.sum(feature_selection), "X_local.shape = ", X_local.shape, end="\r")
        i = i + 1

        X_train, X_test, y_train, y_test = train_test_split(X_local, y, test_size=0.10, random_state=seed)

        reg = LinearRegression().fit(X_train, y_train)
        score_list += [-reg.score(X_train, y_train)]
        
        # This is too slow for running GA
        # from sklearn.svm import SVR
        # regressor = SVR(kernel = 'rbf')
        # regressor.fit(X_train, y_train)
        # score_list += [-regressor.score(X_train, y_train)]

    return score_list

from genetic_model import genetic_algorithm, onemax

# define the total iterations
n_iter = 200
# bits
n_bits = X.shape[1] #20
# define the population size
n_pop = n_bits * 6 #100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1 # THis is differnt for the index and the bit string version # 1.0 / float(n_bits)
# perform the genetic algorithm search
best, score = genetic_algorithm(linear_regression_score, X, y, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
print('f(%s) = %f' % (best, score))

p_bin = [0] * X.shape[1]
for index in best:
    p_bin[index] = 1
best = p_bin

print("Number of selected features = ", sum(best))
#Printing the best with column names - 
for f in range(len(best)):
    if best[f] != 0:
        #print(f)
        print(features[f])

best = np.asarray(best)
best_features = best[np.newaxis, :]

# Use broadcasting for the selection of columns
X_local = X * best_features

idx = np.argwhere(np.all(X_local[..., :] == 0, axis=0))
X_local = np.delete(X_local, idx, axis=1)

X_train, X_validate, y_train, y_validate = train_test_split(X_local, y, test_size=0.10) #, random_state=42)
reg = LinearRegression().fit(X_train, y_train)

y_pred = reg.predict(X_validate)

###################################################################################
# PLOTTING 
###################################################################################

import matplotlib
import matplotlib.pyplot as plt

#Plotting to visualize the accuracy of our model.
fig = plt.figure()
plt.plot(y_validate, y_pred, '.')
plt.xlabel("y_validate")
plt.ylabel("y_validate_pred")
plt.plot(range(2,14), range(2,14), '--')
fig.savefig('accuracy_validate.png', dpi=fig.dpi)

fig = plt.figure()
plt.plot(y_train, reg.predict(X_train), '.')
plt.xlabel("y_train")
plt.ylabel("y_train_pred")
plt.plot(range(2,14), range(2,14), '--')
fig.savefig('accuracy_train.png', dpi=fig.dpi)
