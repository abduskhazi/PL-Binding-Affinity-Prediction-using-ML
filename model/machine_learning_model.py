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

for row in input_list:
    row_values = row.split()
    complex_name = row_values[0][:4]
    # Removing 0, 1 and 57 as they are only strings.
    # First removing the 57th columns
    # Then remove the first 2 columns
    row_values = row_values[:57] + row_values[58:]
    row_values = row_values[2:]
    X += [row_values]
    y += [regression[complex_name]]

X = np.array(X)
print("Len of X = ", np.shape(X))
print("Len of y = ", len(y))
