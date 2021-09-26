import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import data_bakery as bakery
import reproducibility

# Firstly ...
ExecutionID = None
if len(sys.argv) > 1:
    ExecutionID = int(sys.argv[1])
ExecutionID = reproducibility.reproduce(ExecutionID)


###################################################################################
# PLOTTING 
###################################################################################

import matplotlib
import matplotlib.pyplot as plt

def plot_figures(X_validate, y_validate, X_test, y_test):
    print("Plotting to visualize the accuracy of our model.")
    fig = plt.figure()
    plt.plot(y_validate, reg.predict(X_validate), '.')
    plt.xlabel("y_validate")
    plt.ylabel("y_validate_pred")
    plt.title("Execution ID = " + str(ExecutionID))
    plt.plot(range(2,14), range(2,14), '--')
    fig.savefig('accuracy_validate.png', dpi=fig.dpi)

    fig = plt.figure()
    plt.plot(y_test, reg.predict(X_test), '.')
    plt.xlabel("y_test")
    plt.ylabel("y_test_pred")
    plt.title("Execution ID = " + str(ExecutionID))
    plt.plot(range(2,14), range(2,14), '--')
    fig.savefig('accuracy_test.png', dpi=fig.dpi)

best = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0    , 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0,     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0    , 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0    , 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,     1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0    , 0, 1]

#X, y, features, weights = bakery.bake_train_Xy_with_given_features(best) #_correlated_feature_selection(pearson = True)
#X_test, y_test, _, w_test = bakery.bake_test_Xy_with_given_features(best) #_correlated_feature_selection(pearson = True)

X, y, features, weights = bakery.bake_train_Xy()
X_test, y_test, _, w_test = bakery.bake_test_Xy()

#X, y, features, weights = bakery.bake_train_Xy_correlated_feature_selection(spearman = True)
#X_test, y_test, _, w_test = bakery.bake_test_Xy_correlated_feature_selection(spearman = True)


#X, y, features, weights = bakery.bake_train_Xy_manual_feature_selection()
#X_test, y_test, _, w_test = bakery.bake_test_Xy_manual_feature_selection()

#X, y, features, weights = bakery.bake_train_Xy_manual_feature_selection()
print("X.shape =", X.shape)
print("y.shape =", y.shape)

# This is to control the randomness when selecting train-validation set
# If you change the training set, the function fitted to the data changes. We want to avoid this.
# import random
# seed = random.randint(0,2**32)

# Reporting Linear Regression accuracy with all features included (R^2 score)
X_train, X_validate, y_train, y_validate, w_train, w_validate = bakery.test_train_split(X, y, weights, test_size=0.2)
#X_train, y_train, w_train = bakery.duplicate_data(X_train, y_train, w_train)

import time
start = time.time()

reg = LinearRegression().fit(X_train, y_train, w_train)

end = time.time()
print("Fitting time for Linear regression: " + str(end - start))



plot_figures(X_validate, y_validate, X_test, y_test)

training_r2_score = reg.score(X_train, y_train)
validation_r2_score = reg.score(X_validate, y_validate)
testing_r2_score = reg.score(X_test, y_test)

print("Training R2 score = ", training_r2_score)
print("Validation R2 score = ", validation_r2_score)
print("Testing R2 score = ", testing_r2_score)

# Running the genetic algorithms.
score = -reg.score(X_train, y_train)
print("Linear regression score (raw) = ", score)

# Our optimization function = score * (X.shape[1] - num_selected_features)
print("Linear regression score (Used for optimization)= ", score * (X.shape[1] - X.shape[1]))


def linear_regression_score(population, X, y):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

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

        X_train, _, y_train, _ = train_test_split(X_local, y, test_size=0.10, random_state=seed)

        reg = LinearRegression(n_jobs=-1).fit(X_train, y_train)
        score = -reg.score(X_train, y_train)
        score_list += [ score * (X.shape[1] - np.sum(feature_selection))]

    print()
    return score_list

from genetic_model import genetic_algorithm, onemax

# define the total iterations
n_iter = 500
# bits
n_bits = X.shape[1]
# define the population size
n_pop = n_bits * 6 #100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / float(n_bits) #1 # THis is differnt for the index and the bit string version # 1.0 / float(n_bits)
# perform the genetic algorithm search
best, score = genetic_algorithm(linear_regression_score, X, y, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
print('f(%s) = %f' % (best, score))

print("Number of selected features = ", sum(best))
for f in range(len(best)):
    if best[f] != 0:
        print(features[f])

best = np.asarray(best)
best_features = best[np.newaxis, :]

# Use broadcasting for the selection of columns
X_local = X * best_features

idx = np.argwhere(np.all(X_local[..., :] == 0, axis=0))
X_local = np.delete(X_local, idx, axis=1)

X_train, X_validate, y_train, y_validate = train_test_split(X_local, y, test_size=0.20) #, random_state=42)
reg = LinearRegression().fit(X_train, y_train)

plot_figures(X_train, y_train, X_validate, y_validate)
