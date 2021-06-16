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
    row_ligand = [row_ligand[i] for i in ligand_indexes]
    # print(row_protein)
    # print(row_ligand)
    # exit()
    X += [row_protein + row_ligand]
    y += [regression[complex_name]]

X = np.array(X)
print("Len of X = ", np.shape(X))
print("Len of y = ", len(y))

###################################################################################
# MACHINE LEARNING MODEL BEGINS
###################################################################################

