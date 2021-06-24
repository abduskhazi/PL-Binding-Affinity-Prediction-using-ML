import sys
print("Input the ligand_file Name : ", file=sys.stderr)

# ligand_file_name = "ligand_desc_mol2.data"
ligand_file_name = input()

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

for p in proteins_list:
    key = p.split()[0][:4]
    if key not in proteins:
        proteins[key] = []
    proteins[key] += [p]

for l in ligands_list:
    key = l.split()[0][:4]
    ligands[key] = l

def get_resulting_string(p_d, l_d):
    # result = "\t".join(p_d[:2]) + "\t" + l_d[0] + "\t" + "\t".join(p_d[2:]) + "\t" + "\t".join(l_d[1:])
    result = "\t".join(p_d) + "\t" + "\t".join(l_d)
    return result

for pl_complex in complexes_list:
    p_ds = proteins[pl_complex] if pl_complex in proteins else None
    l_d = ligands[pl_complex] if pl_complex in ligands else None 
    if p_ds and l_d:
        for p_d in p_ds:
            single_line_descriptor = get_resulting_string(p_d.split(), l_d.split())
            print(single_line_descriptor)
