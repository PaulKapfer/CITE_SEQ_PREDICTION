import scanpy as sc
import pandas as pd
from pathlib import Path

INPUT_FILE = "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Annotation/Kotliarov/Output/0.Raw/H1_day0.h5ad"
OUTPUT_DIR = Path("C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Annotation/Kotliarov/Output/5.ADT_proteins")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(INPUT_FILE)

if 'ADT' in adata.obsm:
    adt = adata.obsm['ADT']
    if hasattr(adt, 'columns'):
        protein_names = list(adt.columns)
    else:
        protein_names = [f'protein_{i}' for i in range(adt.shape[1])]
elif hasattr(adata, 'obsm') and any('adt' in k.lower() for k in adata.obsm.keys()):
    key = next(k for k in adata.obsm.keys() if 'adt' in k.lower())
    adt = adata.obsm[key]
    protein_names = list(adt.columns) if hasattr(adt, 'columns') else [f'protein_{i}' for i in range(adt.shape[1])]
else:
    raise KeyError(f"No ADT data found in adata.obsm. Available keys: {list(adata.obsm.keys())}")

print(f"Found {len(protein_names)} proteins in ADT assay:\n")
for name in protein_names:
    print(f"  {name}")

df = pd.DataFrame({'protein': protein_names})
out_file = OUTPUT_DIR / "ADT_protein_names.csv"
df.to_csv(out_file, index=False)
print(f"\nSaved to: {out_file}")
