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

    return X, y

def bake_train_Xy():
    output_variable_file = "regression_var.data"
    input_variable_file = "../data/train/train_model_input_all_proteins_mol2_fp_no_nan.data"
    return bake_Xy(output_variable_file, input_variable_file)

def bake_test_Xy():
    output_variable_file = "regression_var.data"
    input_variable_file = "../data/test/test_model_input_all_proteins_mol2_fp_no_nan.data"
    return bake_Xy(output_variable_file, input_variable_file)

if __name__ == "__main__":
    X_train, y_train = bake_train_Xy()
    print("X_train.shape =", X_train.shape)
    print("y_train.shape =", y_train.shape)

    X_test, y_test = bake_test_Xy()
    print("X_test.shape =", X_test.shape)
    print("y_test.shape =", y_test.shape)
