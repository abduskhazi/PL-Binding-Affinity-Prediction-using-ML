import numpy as np

def bake_Xy(output_variable_file, input_variable_file):

    with open(output_variable_file) as regression_file:
        regression_list = [r.strip() for r in regression_file.readlines()]

    regression = {}

    for r in regression_list:
        # Not including the comments in the file.
        if r[0] != "#":
            key = r.split()[0]
            value = r.split()[1]
            regression[key] = float(value)

    with open(input_variable_file) as input_var_f:
        input_list = [r.strip() for r in input_var_f.readlines()]

    X = []
    y = []

    protein_start = 2
    ligand_start = 1 # Use 193 for ignoring auto curr 2d descriptors

    for row in input_list:
        row_values = row.split()
        complex_name = row_values[0][:4]
        row_protein = row_values[:57]
        row_ligand = row_values[57:]
        # since IPC has huge values we have to use a logorithmic scale. Otherwise Value 235 should be removed
        row_ligand[235] = np.log(float(row_ligand[235]))
        # Get rid of the string columns
        x_i = [row_protein[protein_start:] + row_ligand[ligand_start:]]
        X += x_i
        y += [regression[complex_name]]

    X = np.asarray(X, dtype='float64')
    y = np.asarray(y, dtype='float64')

    return X, y, get_feature_names(protein_start, ligand_start)

def get_feature_names(protein_start, ligand_start):
    with open("../data/protein_descriptors.txt") as names_f:
        names_desc_prot = ["protein." + n.strip() for n in names_f.readlines()]

    with open("../data/ligand_descriptors.txt") as c:
        ligand_columns = ["ligand." + i.strip() for i in c.readlines()]

    return names_desc_prot[protein_start:] + ligand_columns[ligand_start:]

def bake_train_Xy():
    output_variable_file = "../data/regression_varible.data"
    input_variable_file = "../data/train/train_model_input_all_proteins_mol2_fp_no_nan.data"
    return bake_Xy(output_variable_file, input_variable_file)

def bake_test_Xy():
    output_variable_file = "../data/regression_varible.data"
    input_variable_file = "../data/test/test_model_input_all_proteins_mol2_fp_no_nan.data"
    return bake_Xy(output_variable_file, input_variable_file)

def bake_train_Xy_manual_feature_selection():
    X, y, feature_names = bake_train_Xy()

    selected_features_list = []
    with open("manual_ligand_features.csv") as f:
        for l in f.readlines():
            name, flag = l.strip().split(';')
            name = "ligand." + name
            if flag == '1': # Include the feature
                selected_features_list += [name]

    # Use all protein features and select the ligand features
    best = [1] * len([ f for f in feature_names if "protein." in f])
    best += [0] * len([ f for f in feature_names if "ligand." in f])
    for i in range(len(feature_names)):
        if feature_names[i] in selected_features_list:
            best[i] = 1

    best = np.asarray(best)
    best_features = best[np.newaxis, :]

    # Use broadcasting for zeroing all the unwanted columns
    # Deleting the columns is a little complicated so not done here.
    #    here are some selected columns that are zero by default.
    X_selected = X * best_features

    # Removing the columns that are zeros
    idx = np.argwhere(np.all(X_selected[..., :] == 0, axis=0))
    X_selected = np.delete(X_selected, idx, axis=1)
    idx = list(idx.flatten())

    features_selected = []
    for i in range(len(feature_names)):
        if i not in idx:
            features_selected += [feature_names[i]]

    return X_selected, y, features_selected

def bake_train_Xy_exclude_features_families(exclusion_list):
    X, y, feature_names = bake_train_Xy()

    # Get all the ids to exclude
    ids_to_exclude = []
    for exc in exclusion_list:
        for i in range(len(feature_names)):
            if exc in feature_names[i]:
                ids_to_exclude += [i]
    ids_to_exclude = sorted(list(set(ids_to_exclude)))

    idx = np.array(ids_to_exclude)
    idx = idx[:, np.newaxis]
    X_selected = np.delete(X, idx, axis=1)

    idx = list(idx.flatten())
    features_selected = []
    for i in range(len(feature_names)):
        if i not in idx:
            features_selected += [feature_names[i]]

    return X_selected, y, features_selected


if __name__ == "__main__":
    X_train, y_train, _ = bake_train_Xy()
    print("X_train.shape =", X_train.shape)
    print("y_train.shape =", y_train.shape)

    X_test, y_test, _ = bake_test_Xy()
    print("X_test.shape =", X_test.shape)
    print("y_test.shape =", y_test.shape)

    protein_start = 2
    ligand_start = 1 # Use 193 for ignoring auto curr 2d descriptors

    features = get_feature_names(protein_start, ligand_start)
    print("OVERVIEW :")
    print("\tLength of feature column names =", len(features))
    print("\t0 ->", features[0], " ... 54 ->", features[54])
    print("\t55 ->", features[55], " ... %d ->" % (len(features)-1), features[-1])

    X_train, y, features = bake_train_Xy_manual_feature_selection()
    print("X_train.shape =", X_train.shape)
    print("y_train.shape =", y_train.shape)
    print("len(features) = ", len(features))

    X_train, y, features = bake_train_Xy_exclude_features_families(["AUTOCORR2D_"])
