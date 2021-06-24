from scipy.stats import spearmanr
from scipy.stats import pearsonr
import numpy as np

import sys
print("Input the ligand_file Name : ", file=sys.stderr)

# ligand_file_name = "ligand_desc_mol2.data"
ligand_file_name = input () # "./general/mol2/ligand_desc_mol2.data" # input()

print("Input the protein description filename : ", file=sys.stderr)
protein_file_name = input() # "protein_descriptors.data"

complexes_file_name = "complexes"


with open(complexes_file_name) as complex_f:
    complexes_list = [c.strip() for c in complex_f.readlines()]

with open(protein_file_name) as prot_f:
    proteins_list = [p.strip() for p in prot_f.readlines()]

with open(ligand_file_name) as ligand_f:
    ligands_list = [l.strip() for l in ligand_f.readlines()]

proteins = {}
ligands = {}

regression = {}

for p in proteins_list:
    key = p.split()[0][:4]
    if key not in proteins:
        proteins[key] = []
    proteins[key] += [p]

for l in ligands_list:
    key = l.split()[0][:4]
    ligands[key] = l


regression_file = "regression_var.data"

with open(regression_file) as regression_f:
    regression_list= [r.strip() for r in regression_f.readlines()]

for r in regression_list:
    if r[0] != "#":
        key = r.split()[0]
        regression[key] = float(r.split()[1])
        # print(key, regression[key])


with open("names_protein_descriptors.txt") as names_f:
    names_desc_prot = [n.strip() for n in names_f.readlines()]

# print(len(names_desc_prot))


print("************************************")
print("Protein Feature selection")
print("************************************")

correlation_map_s = []
correlation_map_p = []

for col in zip(list(range(57))[3:], names_desc_prot[3:]):
    X = []
    y = []
    i, name = col
    for key in complexes_list:
        if key in proteins and key in regression:
            for p_d in proteins[key]:
                val = float(p_d.split()[i])
                import math
                if not math.isnan(val):
                    X += [float(p_d.split()[i])]
                    y += [regression[key]]

    corr_s, _ = spearmanr(X, y)
    corr_p, _ = pearsonr(X, y)
    corr_list_s = []
    corr_list_p = []
    for chunk in zip(np.array_split(X, 5), np.array_split(y, 5)):
        X_chunk = chunk[0]
        y_chunk = chunk[1]
        corr_list_s += [spearmanr(X_chunk, y_chunk)[0]]
        corr_list_p += [pearsonr(X_chunk, y_chunk)[0]]
    # print(name, corr_s, corr_list_s, len(y))
    correlation_map_s += [(i, name, corr_s, corr_list_s[:])]
    correlation_map_p += [(i, name, corr_p, corr_list_p[:])]
    # print( "\t", name, corr_p, corr_list_p, len(y))

# Remove the nans from the correlations because they will not help us in any computations.
# This is because they are constant values.
correlation_map_s = [x for x in correlation_map_s if not math.isnan(x[2])]
correlation_map_p = [x for x in correlation_map_p if not math.isnan(x[2])]

# Select the top features based on the correlation
correlation_map_s.sort(key=lambda x:abs(x[2]), reverse=True)
correlation_map_p.sort(key=lambda x:abs(x[2]), reverse=True)
print("================== The best features according to spearman coefficient ====================")
for i in range(20):
    i, name, val, chunks = correlation_map_s[i]
    print(i, name, "%.4f" % val, "\t\t-->" , ["%.4f" % x for x in chunks] )
print("================= The best features according to pearson coefficient ================")
for i in range(20):
    i, name, val, chunks = correlation_map_p[i]
    print(i, name, "%.4f" % val, "\t\t-->" , ["%.4f" % x for x in chunks] )
    # print("%.4f" % correlation_map_p[i])


print("************************************")
print("Ligand Feature Selection")
print("************************************")
with open("names_ligand_descriptors.txt") as c:
    ligand_columns = c.readlines()

ligand_columns = [c.strip() for c in ligand_columns]
# ligand_columns = ["ligand_name"] + ligand_columns

correlation_map_s = []
correlation_map_p = []

for col in zip(list(range(403))[1:], ligand_columns[1:]):
    X = []
    y = []
    i, name = col
    for key in complexes_list:
        if key in ligands and key in regression:
            val = float(ligands[key].split()[i])
            import math
            if not math.isnan(val) and not math.isinf(val):
                X += [val]
                y += [regression[key]]

    corr_s, _ = spearmanr(X, y)
    corr_p, _ = pearsonr(X, y)
    corr_list_s = []
    corr_list_p = []
    for chunk in zip(np.array_split(X, 5), np.array_split(y, 5)):
        X_chunk = chunk[0]
        y_chunk = chunk[1]
        corr_list_s += [spearmanr(X_chunk, y_chunk)[0]]
        corr_list_p += [pearsonr(X_chunk, y_chunk)[0]]
    # print(name, corr_s, corr_list_s, len(y))
    correlation_map_s += [(i, name, corr_s, corr_list_s[:])]
    correlation_map_p += [(i, name, corr_p, corr_list_p[:])]

# Remove the nans from the correlations because they will not help us in any computations.
# This is because they are constant values.
correlation_map_s = [x for x in correlation_map_s if not math.isnan(x[2])]
correlation_map_p = [x for x in correlation_map_p if not math.isnan(x[2])]

# Select the top features based on the correlation
correlation_map_s.sort(key=lambda x:abs(x[2]), reverse=True)
correlation_map_p.sort(key=lambda x:abs(x[2]), reverse=True)
print("================== The best features according to spearman coefficient ====================")
for i in range(20):
    i, name, val, chunks = correlation_map_s[i]
    print(i, name, "%.4f" % val, "\t\t-->" , ["%.4f" % x for x in chunks] )
    # print("%.4f" % correlation_map_s[i])
print("================= The best features according to pearson coefficient ================")
for i in range(20):
    i, name, val, chunks = correlation_map_p[i]
    print(i, name, "%.4f" % val, "\t\t-->" , ["%.4f" % x for x in chunks] )
    # print("%.4f" % correlation_map_p[i])

