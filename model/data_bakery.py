import numpy as np
import sklearn.model_selection
from sklearn.utils import shuffle

def bake_Xy(output_variable_file, input_variable_file):

    with open(output_variable_file) as regression_file:
        regression_list = [r.strip() for r in regression_file.readlines()]

    regression = {}
    resolution = {}
    max_resolution = 0

    for r in regression_list:
        # Not including the comments in the file.
        if r[0] != "#":
            key = r.split()[0]
            try:
                res_val = float(r.split()[1])
            except ValueError:
                res_val = 5 # This is choses because this is the max resolution found in the data set.
            value = r.split()[3]
            regression[key] = float(value)
            resolution[key] = res_val
            if(max_resolution < resolution[key]):
                max_resolution = resolution[key]

    print("Max resolution", max_resolution)
    # Resolution statistics
    stats = {}
    for key in resolution:
        data_resolution = int(round(resolution[key]))
        if(data_resolution not in stats):
            stats[data_resolution] = 0
        stats[data_resolution] += 1

    print("Resolution Statistics:")
    print("    Resolution    Num datapoint")
    for key in sorted(stats):
        print("     ~",key,"           ",stats[key])

    import matplotlib.pyplot as plt
    fig = plt.figure()
    hist_plot = []
    for key in resolution:
        hist_plot += [resolution[key]]
    n, bins, patches = plt.hist(hist_plot, bins=20, density=False)
    plt.xlabel("resolution")
    plt.ylabel("frequency")
    fig.savefig('resolution_distribution.png', dpi=fig.dpi)

    with open(input_variable_file) as input_var_f:
        input_list = [r.strip() for r in input_var_f.readlines()]


    X = []
    y = []
    data_weights = []

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
        data_weights += [max_resolution/resolution[complex_name]] # --> Hyperbolic weighting
        #data_weights += [max_resolution + 1 - resolution[complex_name]] # Linear weighting

    X = np.asarray(X, dtype='float64')
    y = np.asarray(y, dtype='float64')

    return X, y, get_feature_names(protein_start, ligand_start), data_weights

def get_feature_names(protein_start, ligand_start):
    with open("../data/protein_descriptors.txt") as names_f:
        names_desc_prot = ["protein." + n.strip() for n in names_f.readlines()]

    with open("../data/ligand_descriptors.txt") as c:
        ligand_columns = ["ligand." + i.strip() for i in c.readlines()]

    return names_desc_prot[protein_start:] + ligand_columns[ligand_start:]

def remove_features(feature_names, exclusion_ids):
    features_selected = []
    for i in range(len(feature_names)):
        if i not in exclusion_ids:
            features_selected += [feature_names[i]]

    return features_selected

def remove_columns(X, idx):
    idx = np.array(idx)
    idx = idx[:, np.newaxis]
    X_selected = np.delete(X, idx, axis=1)
    return X_selected

def bake_train_Xy():
    output_variable_file = "../data/INDEX_general_PL_data.2019"
    input_variable_file = "../data/train/train_model_input_all_proteins_mol2_fp_no_nan.data"
    return bake_Xy(output_variable_file, input_variable_file)

def bake_test_Xy():
    output_variable_file = "../data/INDEX_general_PL_data.2019"
    input_variable_file = "../data/test/test_model_input_all_proteins_mol2_fp_no_nan.data"
    return bake_Xy(output_variable_file, input_variable_file)

def bake_train_Xy_with_given_features(features):
    X, y, feature_names, weights = bake_train_Xy()

    list_indexes = [id for id, e in enumerate(features) if e == 0]
    X_selected = remove_columns(X, list_indexes)
    features_selected = remove_features(feature_names, list_indexes)

    return X_selected, y, features_selected, weights


def bake_Xy_manual_feature_selection(X, y, feature_names, weights):

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

    list_indexes = [id for id, e in enumerate(best) if e == 0]
    X_selected = remove_columns(X, list_indexes)
    features_selected = remove_features(feature_names, list_indexes)

    return X_selected, y, features_selected, weights

def bake_train_Xy_manual_feature_selection():
    X, y, feature_names, weights = bake_train_Xy()
    return bake_Xy_manual_feature_selection(X, y, feature_names, weights)

def bake_test_Xy_manual_feature_selection():
    X, y, feature_names, weights = bake_test_Xy()
    return bake_Xy_manual_feature_selection(X, y, feature_names, weights)

def bake_train_Xy_with_specific_columns(selected_list):
    X, y, feature_names, weights = bake_train_Xy()

    required_columns = []
    for col in selected_list:
        for i in range(len(feature_names)):
            if col in feature_names[i]:
                required_columns += [1]
            else:
                required_columns += [0]

    return bake_train_Xy_with_given_features(required_columns)

def bake_Xy_correlated_feature_selection(pearson, spearman, X, y, feature_names, weights):

    ligand_feature_file_name = None
    protein_feature_file_name = None

    if(pearson):
        ligand_feature_file_name = "pearson_corr_ligand.txt"
        protein_feature_file_name = "pearson_corr_protein.txt"

    if(spearman):
        ligand_feature_file_name = "spearman_corr_ligand.txt"
        protein_feature_file_name = "spearman_corr_protein.txt"

    required_names = []
    with open(ligand_feature_file_name) as file:
        required_names += ["ligand." + r.strip() for r in file.readlines() if r[0] != '#']

    with open(protein_feature_file_name) as file:
        required_names += ["protein." + r.strip() for r in file.readlines() if r[0] != '#']

    required_columns = []
    for i in range(len(feature_names)):
        required_columns += [0]
        for col in required_names:
            if col == feature_names[i]:
                required_columns[-1] = 1

    print(sum(required_columns))
    print(len(required_columns))

    return bake_train_Xy_with_given_features(required_columns)

def bake_train_Xy_correlated_feature_selection(pearson = False, spearman = False):
    X, y, feature_names, weights = bake_train_Xy()
    return bake_Xy_correlated_feature_selection(pearson, spearman, X, y, feature_names, weights)

def bake_test_Xy_correlated_feature_selection(pearson = False, spearman = False):
    X, y, feature_names, weights = bake_test_Xy()
    return bake_Xy_correlated_feature_selection(pearson, spearman, X, y, feature_names, weights)


def bake_train_Xy_exclude_features_families(exclusion_list):
    X, y, feature_names, weights = bake_train_Xy()

    # Get all the ids to exclude
    ids_to_exclude = []
    for exc in exclusion_list:
        for i in range(len(feature_names)):
            if exc in feature_names[i]:
                ids_to_exclude += [i]
    list_indexes = sorted(list(set(ids_to_exclude)))

    X_selected = remove_columns(X, ids_to_exclude)
    features_selected = remove_features(feature_names, list_indexes)

    return X_selected, y, features_selected, weights

def test_train_split(X, y, weights, test_size=0.2):
    weights = np.array(weights, dtype='float64')
    X = np.concatenate((X, weights[:, np.newaxis]), axis=1)
    print("X concatenated with weights")
    print("    X.shape =", X.shape)
    print("    y.shape =", y.shape)

    X_train, X_validate, y_train, y_validate = sklearn.model_selection.train_test_split(X, y, test_size=test_size)

    weights_train = np.array(X_train[:, -1], dtype='float64')
    weights_validate = np.array(X_validate[:, -1], dtype='float64')

    X = np.delete(X, -1, axis=1)
    X_train = np.delete(X_train, -1, axis=1)
    X_validate = np.delete(X_validate, -1, axis=1)
    print("X_train.shape =", X_train.shape)
    print("y_train.shape =", y_train.shape)
    
    return X_train, X_validate, y_train, y_validate, weights_train, weights_validate

def duplicate_data(X_train, y_train, weights_train):

    weights_train = np.round(weights_train)
    weights_train = np.array(weights_train, dtype='int64')
    X_train = np.repeat(X_train, weights_train, axis=0)
    y_train = np.repeat(y_train, weights_train, axis=0)

    print("After weight duplication")
    print("    X_train.shape =", X_train.shape)
    print("    y_train.shape =", y_train.shape)

    X_train, y_train = shuffle(X_train, y_train)

    # After duplication of data, the weight of each data point is equal
    return X_train, y_train, np.array([1]* X_train.shape[0])

if __name__ == "__main__":
    X_train, y_train, _, _ = bake_train_Xy()
    print("X_train.shape =", X_train.shape)
    print("y_train.shape =", y_train.shape)

    X_test, y_test, _, _ = bake_test_Xy()
    print("X_test.shape =", X_test.shape)
    print("y_test.shape =", y_test.shape)

    protein_start = 2
    ligand_start = 1 # Use 193 for ignoring auto curr 2d descriptors

    features = get_feature_names(protein_start, ligand_start)
    print("OVERVIEW :")
    print("\tLength of feature column names =", len(features))
    print("\t0 ->", features[0], " ... 54 ->", features[54])
    print("\t55 ->", features[55], " ... %d ->" % (len(features)-1), features[-1])

    X_train, y, features, _ = bake_train_Xy_manual_feature_selection()
    print("X_train.shape =", X_train.shape)
    print("y_train.shape =", y_train.shape)
    print("len(features) = ", len(features))

    X_train, y, features, _ = bake_train_Xy_exclude_features_families(["AUTOCORR2D_"])
    print("X_train.shape =", X_train.shape)
    print("y_train.shape =", y_train.shape)
    print("len(features) = ", len(features))

    f = remove_features(["a", "b", "c"], [0, 1])
    print(f)

    selected_features = [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    X, y, features, _ = bake_train_Xy_with_given_features(selected_features)
    print(sum(selected_features))
    print(X.shape)
