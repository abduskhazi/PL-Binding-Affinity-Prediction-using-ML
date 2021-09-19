import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from data_bakery import bake_train_Xy
import reproducibility

#With specific initialization (2000 generations, population = 457

specific_init_best = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]

#Number of features selection = 49


from data_bakery import bake_train_Xy
import numpy as np

X, y, features = bake_train_Xy()
print("X.shape =", X.shape)
print("y.shape =", y.shape)

best = np.asarray(specific_init_best)
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
protein.crit6_continue
protein.mean_loc_hyd_dens
protein.polarity_score
protein.n_abpa
protein.GLN
ligand.AUTOCORR2D_106
ligand.AUTOCORR2D_116
ligand.AUTOCORR2D_13
ligand.AUTOCORR2D_137
ligand.AUTOCORR2D_141
ligand.AUTOCORR2D_164
ligand.AUTOCORR2D_170
ligand.AUTOCORR2D_183
ligand.AUTOCORR2D_190
ligand.AUTOCORR2D_191
ligand.AUTOCORR2D_21
ligand.AUTOCORR2D_3
ligand.AUTOCORR2D_37
ligand.AUTOCORR2D_56
ligand.AUTOCORR2D_61
ligand.AUTOCORR2D_66
ligand.AUTOCORR2D_69
ligand.AUTOCORR2D_74
ligand.AUTOCORR2D_8
ligand.BCUT2D_LOGPHI
ligand.Chi0
ligand.FpDensityMorgan2
ligand.FractionCSP3
ligand.MaxPartialCharge
ligand.MolMR
ligand.NumSaturatedHeterocycles
ligand.PEOE_VSA2
ligand.PEOE_VSA9
ligand.RingCount
ligand.SMR_VSA1
ligand.SMR_VSA6
ligand.SlogP_VSA10
ligand.SlogP_VSA2
ligand.SlogP_VSA3
ligand.VSA_EState2
ligand.fr_Al_OH
ligand.fr_Ar_COO
ligand.fr_HOCCN
ligand.fr_NH0
ligand.fr_N_O
ligand.fr_imidazole
ligand.fr_pyridine
ligand.fr_sulfonamd
ligand.qed
"""
