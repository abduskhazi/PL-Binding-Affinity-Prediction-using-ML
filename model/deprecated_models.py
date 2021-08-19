# This file contains code that was used for testing different models.
# Future work - Create a seperate script for each of these models

import sys
import data_bakery as bakery
import reproducibility
from sklearn.metrics import r2_score

# Firstly ...
ExecutionID = None
if len(sys.argv) > 1:
    ExecutionID = int(sys.argv[1])
ExecutionID = reproducibility.reproduce(ExecutionID)

X, y, features = bakery.bake_train_Xy()

print("X.shape =", X.shape)
print("y.shape =", y.shape)

##################################################################################
### Model 1 --> ordinary least squares
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20)

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

if False:
    # Statistics of SVR (Execution ID = 438302616)
    # When all features are used.
    #     Support Vector Regession R^2 training score =  0.31437544671737994
    #     Support Vector Regession R^2 validation score =  0.28133698851672306
    # Time taken - 6m27.786s
    from sklearn.svm import SVR
    regressor = SVR(kernel = "poly")
    print("Fitting an SVR...")
    regressor.fit(X_train, y_train)
    print("Fitting finished")
    print("Support Vector Regession R^2 training score = ", regressor.score(X_train, y_train))
    print("Support Vector Regession R^2 validation score = ", regressor.score(X_validate, y_validate))

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
            self.fc_1 = nn.Linear(457,10)
            self.fc_2 = nn.Linear(10,7)
            self.fc_3 = nn.Linear(7, 4)
            self.fc_4 = nn.Linear(4, 2)
            self.fc_5 = nn.Linear(2, 1)
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
            x = F.relu(x)
            x = self.fc_4(x)
            x = F.relu(x)
            x = self.fc_5(x)
            return x


    deepNN = Net().double()
    loss_function = nn.MSELoss()

    import torch.optim as optim
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(deepNN.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(deepNN.parameters(), lr=0.01)

    X_train_torch = torch.from_numpy(X_train)
    y_train_torch = torch.from_numpy(y_train).view(-1, 1)
    batch_size = 700

    n = X_train_torch.shape[0]

    score = 0.0
    epoch = 0
    #while(score < 0.95): # --> Use this when you want to check the best local minima.
    for epoch in range(1000):
        #Shuffling data for better regularization
        #from sklearn.utils import shuffle
        #X_train, y_train = shuffle(X_train, y_train)

        epoch += 1

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

        X_train_torch = torch.from_numpy(X_train)
        y_train_pred = deepNN(X_train_torch)
        y_train_pred = y_train_pred.detach().numpy()
        X_validate_torch = torch.from_numpy(X_validate)
        y_validate_pred = deepNN(X_validate_torch)
        y_validate_pred = y_validate_pred.detach().numpy()
        score_validate = r2_score(y_validate, y_validate_pred)
        score = r2_score(y_train, y_train_pred)
        print("Epoch : ", epoch, "R^2 (Train)", score, "R^2 (Validate)", score_validate, end='\r')

    X_train_torch = torch.from_numpy(X_train)
    y_train_pred = deepNN(X_train_torch)
    y_train_pred = y_train_pred.detach().numpy()
    X_validate_torch = torch.from_numpy(X_validate)
    y_validate_pred = deepNN(X_validate_torch)
    y_validate_pred = y_validate_pred.detach().numpy()
    print("Neural Network R^2 score (Train) = ", r2_score(y_train, y_train_pred))
    print("Neural Network R^2 score (Validation) = ", r2_score(y_validate, y_validate_pred))


if False:
    import matplotlib
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(y_test, y_pred, '.')
    plt.plot(range(2,14), range(2,14), '--')
    fig.savefig('temp.png', dpi=fig.dpi)

