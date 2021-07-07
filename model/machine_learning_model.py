import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# First get the output variable value
output_var_file = "regression_var.data"

print("Enter input variable file name : ", file=sys.stderr)
input_var_file = input()

with open(output_var_file) as regression_f:
    regression_list = [r.strip() for r in regression_f.readlines()]

regression = {}

for r in regression_list:
    if r[0] != "#":
        key = r.split()[0]
        regression[key] = float(r.split()[1])

X = []
y = []

with open(input_var_file) as input_f:
    input_list = [r.strip() for r in input_f.readlines()]

###
# Get rid of strings from the rows.
###

for row in input_list:
    row_values = row.split()
    complex_name = row_values[0][:4]
    row_protein = row_values[:57]
    row_ligand = row_values[57:]
    # since IPC has huge values we have to use a logorithmic scale. Otherwise Value 235 should be removed
    row_ligand[235] = np.log(float(row_ligand[235]))
    # Get rid of the string columns
    x_i = [row_protein[2:] + row_ligand[1:]]
    X += x_i
    y += [regression[complex_name]]

X = np.asarray(X, dtype='float64')
y = np.asarray(y, dtype='float64')
print("Len of X = ", np.shape(X))
print("Len of y = ", np.shape(y))


def linear_regression_score(population, X, y):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures

    population = np.asarray(population)
    population = population[:, np.newaxis, :]

    # Use the same train set for population
    import random
    seed = random.randint(0,2**32)

    score_list = []
    i = 1
    for feature_selection in population:
        # Use broadcasting for the selection of columns
        X_local = X * feature_selection

        # Not working have to fix this.
        #idx = np.argwhere(np.all(X_local[..., :] == 0, axis=1))
        #X_local = np.delete(X_local, idx, axis=1)

        print("Finished - ", i, end="\r")
        i = i + 1

        X_train, X_test, y_train, y_test = train_test_split(X_local, y, test_size=0.10, random_state=seed)

        reg = LinearRegression().fit(X_train, y_train)
        score_list += [-reg.score(X_test, y_test)]

    return score_list

from genetic_model import genetic_algorithm, onemax

# define the total iterations
n_iter = 100
# bits
n_bits = X.shape[1] #20
# define the population size
n_pop = 100 # n_bits * 6 #100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / float(n_bits)
# perform the genetic algorithm search
best, score = genetic_algorithm(linear_regression_score, X, y, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
print('f(%s) = %f' % (best, score))

with open("names_protein_descriptors.txt") as names_f:
    names_desc_prot = [n.strip() for n in names_f.readlines()]

with open("names_ligand_descriptors.txt") as c:
    ligand_columns = c.readlines()

combined_columns = names_desc_prot[2:] + ligand_columns[1:]

print("Number of selected features = ", sum(best))
#Printing the best with column names - 
for f in range(len(best)):
    if best[f] != 0:
        #print(f)
        print(combined_columns[f])

best = np.asarray(best)
best_features = best[np.newaxis, :]

# Use broadcasting for the selection of columns
X_local = X * best_features

idx = np.argwhere(np.all(X_local[..., :] == 0, axis=0))
X_local = np.delete(X_local, idx, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_local, y, test_size=0.10) #, random_state=42)
reg = LinearRegression().fit(X_train, y_train)

y_pred = reg.predict(X_test)

###################################################################################
# PLOTTING 
###################################################################################

import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(y_test, y_pred, '.')
plt.plot(range(2,14), range(2,14), '--')
fig.savefig('temp.png', dpi=fig.dpi)

# Ending the script here.
exit()

###################################################################################
# MACHINE LEARNING MODEL BEGINS
###################################################################################
### Model 1 --> ordinary least squares
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

print(X_train.shape)
print(X_test.shape)

if False:
    degree = 2
    poly = PolynomialFeatures(degree, include_bias=False)
    X = poly.fit_transform(X)
    #y = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    reg = LinearRegression().fit(X_train, y_train)
    print("Ordinary LSQ Regression score (train)    = ", reg.score(X_train, y_train))
    print("Ordinary LSQ Regression score (Validate) = ", reg.score(X_test, y_test))
    #y_pred = reg.predict(X_test)[:15]
    #y_test = y_test[:15]

# Model 1 ends ... We would like to test the model for our test set.
# For this we create a graph of predicted vs actual values to see how it performs.

if False:
    # Model 2 -> Lasso 
    from sklearn import linear_model
    lasso_reg = linear_model.Lasso(alpha=0.8)
    lasso_reg.fit(X_train,y_train)
    y_pred = lasso_reg.predict(X_test)
    print("Lasso Regression score = ", lasso_reg.score(X_train, y_train))

# Since it takes a lot of time.
if False:
    # Next = support vector machines, decision trees, random forest, and neural network
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X_train, y_train)
    print("Support Vector Regession score = ", regressor.score(X_train, y_train))
    y_pred = regressor.predict(X_test)

if False:
    from sklearn.linear_model import Ridge
    regression_ridge = Ridge(alpha=1.0).fit(X_train, y_train)
    y_pred = regression_ridge.predict(X_test)
    print("Ridge Regession score = ", regression_ridge.score(X_train, y_train))

if False:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            # Having 2 legs is not good for a dichotomous problem
            # self.fc_1_protein = nn.Linear(20, 10)
            # self.fc_1_ligand = nn.Linear(20, 10)
            # self.fc_2_protein = nn.Linear(10, 4)
            # self.fc_2_ligand = nn.Linear(10, 4)
            self.fc_1 = nn.Linear(40,10)
            self.fc_2 = nn.Linear(10,4)
            self.fc_3 = nn.Linear(4, 1)
            #########

        def forward(self, x):
            # # Split our x to protein and ligand
            # x_protein, x_ligand = torch.split(x, (20, 20), dim=-1)
            # # Pass x_protein through the protein leg
            # x_protein = F.relu(self.fc_1_protein(x_protein))
            # x_protein = F.relu(self.fc_2_protein(x_protein))
            # # Pass x_ligand through the ligand leg
            # x_ligand = F.relu(self.fc_1_ligand(x_ligand))
            # x_ligand = F.relu(self.fc_2_ligand(x_ligand))
            # # Join the output of the legs
            # x = torch.cat((x_protein, x_ligand), -1)
            # # Continue the left out stretch.
            x = self.fc_1(x)
            x = F.relu(x)
            x = self.fc_2(x)
            x = F.relu(x)
            x = self.fc_3(x)
            return x


    deepNN = Net().double()
    loss_function = nn.MSELoss()

    import torch.optim as optim
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(deepNN.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(deepNN.parameters(), lr=0.01)

    X_train_torch = torch.from_numpy(X_train)
    y_train_torch = torch.from_numpy(y_train).view(-1, 1)
    batch_size = 100

    n = X_train_torch.shape[0]

    for epoch in range(500):
        for i in range((n-1)//batch_size + 1):
            start = i * batch_size
            end = start + batch_size
            x_bs = X_train_torch[start : end]
            y_bs = y_train_torch[start : end]
            
            optimizer.zero_grad()
            output = deepNN(x_bs)
            loss = loss_function(output, y_bs)
            loss.backward()
            optimizer.step()

        print("Epoch : ", epoch)

    X_test_torch = torch.from_numpy(X_test)
    y_pred = deepNN(X_test_torch)
    y_pred = y_pred.detach().numpy()


if False:
    import matplotlib
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(y_test, y_pred, '.')
    plt.plot(range(2,14), range(2,14), '--')
    fig.savefig('temp.png', dpi=fig.dpi)

