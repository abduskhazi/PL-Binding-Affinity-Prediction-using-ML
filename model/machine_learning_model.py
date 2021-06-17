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

X_protein = []
X_ligand = []

for row in input_list:
    row_values = row.split()
    complex_name = row_values[0][:4]
    p_i = row_values[:57]
    l_i = row_values[57:]
    # since IPC has huge values we have to use a logorithmic scale. Otherwise Value 235 should be removed
    l_i[235] = np.log(float(l_i[235]))
    # Removing the first strings in both descriptors
    p_i = p_i[2:]
    l_i = l_i[1:]
    X_protein += [p_i]
    X_ligand += [l_i]
    y += [regression[complex_name]]


X_protein = np.asarray(X_protein, dtype='float64')
X_ligand = np.asarray(X_ligand, dtype='float64')

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression

print("X_protein shape : ", X_protein.shape)
print("X_ligand shape : ", X_ligand.shape)

X_protein = SelectKBest(score_func=mutual_info_regression, k=30).fit_transform(X_protein, y)
X_ligand = SelectKBest(score_func=mutual_info_regression, k=30).fit_transform(X_ligand, y) 

print("X_protein shape (after feature selection) : ", X_protein.shape)
print("X_ligand shape (after feature selection) : ", X_ligand.shape)

X = np.concatenate((X_protein, X_ligand), axis=1)

print("X shape = ", X.shape)

X = np.asarray(X, dtype='float64')
y = np.asarray(y, dtype='float64')
print("Len of X = ", np.shape(X))
print("Len of y = ", np.shape(y))

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

if True:
    #degree = 2
    #poly = PolynomialFeatures(degree, include_bias=False)
    #X = poly.fit_transform(X)
    #y = y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("Ordinary LSQ Regression score = ", reg.score(X_train, y_train))

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


import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(y_test, y_pred, '.')
plt.plot(range(2,14), range(2,14), '--')
fig.savefig('temp.png', dpi=fig.dpi)

