import rdkit.Chem as Chem
import pandas as pd


df = pd.read_csv("./out_new_model_bs_128_hd_200_ld_50.csv")
smiles_arr = [smile_str.strip() for smile_str in df["0"]]
invalid_count = 0
for smi in smiles_arr:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        invalid_count += 1

print("In-valid count: ", invalid_count)
