import sys
import numpy as np

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
# Do direct feature selection here as this will get rid of strings as well.
###

for row in input_list:
    row_values = row.split()
    complex_name = row_values[0][:4]
    # Feature selection. According to spearman coeffecient
    protein_indexes = sorted([17, 30, 18, 15, 16, 10, 13, 25, 47, 56, 11, 3, 31, 6, 49, 5, 9, 26, 28, 19])
    ligand_indexes = sorted([121, 2, 120, 208, 112, 111, 182, 210, 235, 79, 90, 249, 205, 239, 233, 206, 1, 209, 212, 207])
    row_protein = row_values[:57]
    row_ligand = row_values[57:]
    row_protein = [row_protein[i] for i in protein_indexes]
    temp = []
    for i in ligand_indexes:
        # since IPC has huge values we have to use a logorithmic scale. Otherwise Value 235 should be removed
        if i == 235:
            temp += [np.log(float(row_ligand[i]))]
        else:
            temp += [row_ligand[i]]
    row_ligand = temp
    x_i = [row_protein + row_ligand]
    X += x_i
    y += [regression[complex_name]]

X = np.asarray(X, dtype='float64')
y = np.asarray(y, dtype='float64')
print("Len of X = ", np.shape(X))
print("Len of y = ", np.shape(y))


def linear_regression_score():
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    reg = LinearRegression().fit(X_train, y_train)
    return reg.score(X_test, y_test)


from genetic_model import genetic_algorithm, onemax

# define the total iterations
n_iter = 100
# bits
n_bits = 500 #20
# define the population size
n_pop = n_bits * 5 #100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / float(n_bits)
# perform the genetic algorithm search
best, score = genetic_algorithm(onemax, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
print('f(%s) = %f' % (best, score))

exit()
###################################################################################
# PLOTTING 
###################################################################################

import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(y_test, y_pred, '.')
plt.plot(range(2,14), range(2,14), '--')
fig.savefig('temp.png', dpi=fig.dpi)


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

