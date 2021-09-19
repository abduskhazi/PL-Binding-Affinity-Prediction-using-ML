import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from data_bakery import bake_train_Xy
import reproducibility

#With random Initization (500 generations, population = 2400)

random_init_best = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#Number of features selected = 50

from data_bakery import bake_train_Xy
import numpy as np

X, y, features = bake_train_Xy()
print("X.shape =", X.shape)
print("y.shape =", y.shape)

best = np.asarray(random_init_best)
best_features = best[np.newaxis, :]

# Use broadcasting for the selection of columns
X_local = X * best_features

idx = np.argwhere(np.all(X_local[..., :] == 0, axis=0))
X_local = np.delete(X_local, idx, axis=1)

X_train, X_validate, y_train, y_validate = train_test_split(X_local, y, test_size=0.10) #, random_state=42)
reg = LinearRegression().fit(X_train, y_train)

training_r2_score = reg.score(X_train, y_train)
validation_r2_score = reg.score(X_validate, y_validate)

print("Training R2 score = ", training_r2_score)
print("Validation R2 score = ", validation_r2_score)

"""
protein.nb_AS
protein.surf_apol_vdw14
protein.surf_apol_vdw22
protein.n_abpa
protein.ASP
protein.GLN
protein.TRP
protein.TYR
protein.VAL
ligand.AUTOCORR2D_11
ligand.AUTOCORR2D_137
ligand.AUTOCORR2D_159
ligand.AUTOCORR2D_163
ligand.AUTOCORR2D_164
ligand.AUTOCORR2D_166
ligand.AUTOCORR2D_175
ligand.AUTOCORR2D_21
ligand.AUTOCORR2D_3
ligand.AUTOCORR2D_58
ligand.AUTOCORR2D_61
ligand.AUTOCORR2D_66
ligand.AUTOCORR2D_69
ligand.AUTOCORR2D_77
ligand.AUTOCORR2D_8
ligand.BertzCT
ligand.EState_VSA9
ligand.FractionCSP3
ligand.MaxAbsEStateIndex
ligand.NOCount
ligand.NumSaturatedHeterocycles
ligand.PEOE_VSA13
ligand.PEOE_VSA2
ligand.PEOE_VSA9
ligand.SlogP_VSA12
ligand.SlogP_VSA3
ligand.TPSA
ligand.VSA_EState10
ligand.VSA_EState3
ligand.VSA_EState5
ligand.fr_Al_COO
ligand.fr_Al_OH
ligand.fr_Ar_OH
ligand.fr_COO2
ligand.fr_N_O
ligand.fr_alkyl_carbamate
ligand.fr_benzene
ligand.fr_pyridine
ligand.fr_quatN
ligand.fr_sulfonamd
ligand.qed
"""
