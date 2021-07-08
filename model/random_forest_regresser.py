from data_bakery import bake_Xy

# Obtain data for our random forest regressor
X, y = bake_Xy()
print("X.shape =", X.shape)
print("y.shape =", y.shape)
