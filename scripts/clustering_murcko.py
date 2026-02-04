import pickle
import os, sys
from collections import defaultdict
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Clustering based on Murcko scaffolds")
    parser.add_argument(
        "--input-sdf",
        type=str,
        default=None,
        help="example) IrDB/raw/IrDB.sdf",
    )

    return parser.parse_args()

def read_sdf(sdf_path):
    suppl = Chem.SDMolSupplier(sdf_path)
    mol_list = []
    for i, mol in enumerate(suppl):
        smiles = Chem.MolToSmiles(mol)
        mol_id = None
        if mol.HasProp("ID"):
            mol_id = mol.GetProp("ID")
        mol_list.append((mol_id, smiles))
    return mol_list

def fp_from_smiles(smi, radius=2, nBits=2048):
    m = Chem.MolFromSmiles(smi)
    gen = GetMorganGenerator(radius=radius, fpSize=nBits)
    return gen.GetFingerprint(m)

args = get_args()
smiles_dict = defaultdict(list)
sdf_path = args.input_sdf
if sdf_path is None:
    raise Exception("Please check input sdf file")

mol_list = read_sdf(sdf_path)
for (name, smiles) in mol_list:
    mol = Chem.MolFromSmiles(smiles)
    scaf_mol = MurckoScaffold.GetScaffoldForMol(mol)
    scaf_smiles = Chem.MolToSmiles(scaf_mol)
    smiles_dict[scaf_smiles].append((name, smiles, scaf_smiles))

smiles_list = sorted(list(smiles_dict.keys()))

fps = [fp_from_smiles(s) for s in smiles_list]
n = len(fps)

# 1 - Tanimoto similarity = distance
dist = np.zeros((n, n), dtype=float)
for i in range(n):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
    dist[i, :] = [1.0 - s for s in sims]

condensed = squareform(dist, checks=False)
Z = linkage(condensed, method="complete")

name_groups = {}
num_fold = 5
labels = fcluster(Z, t=num_fold, criterion="maxclust")
smiles_len = [0 for _ in range(num_fold)]
for idx, i_fold in enumerate(labels):
    scaf_smiles = smiles_list[idx]
    smiles_len[i_fold-1] += len(smiles_dict[scaf_smiles])

for idx, i_fold in enumerate(labels):
    scaf_smiles = smiles_list[idx]
    for sub in smiles_dict[scaf_smiles]:
        name, smiles = sub[0:2]
        name_groups[name] = (name, smiles, scaf_smiles, i_fold-1)

w_list = ['molecule_id,fold,SMILES,SCAFFOLD\n']
for (name, _) in mol_list:
    smiles = name_groups[name][1]
    scaf_smiles = name_groups[name][2]
    fold = name_groups[name][3]
    w_line = f'{name},{fold},{smiles},{scaf_smiles}\n'
    w_list.append(w_line)

with open('IrDB.cluster.csv', 'w') as f:
    f.writelines(w_list)

