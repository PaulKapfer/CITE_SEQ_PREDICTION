#!/usr/bin/env python3
"""
One-time script: fetch GO_Biological_Process_2023 from Enrichr and save
the endocytosis-related gene sets as a bundled JSON file.

Run once from this directory:
    python fetch_endocytosis_genesets.py

Writes: endocytosis_gene_sets.json
"""
import json
import re
import requests
from pathlib import Path

ENDOCYTOSIS_PATHWAYS = [
    "GOBP_ENDOCYTOSIS",
    "GOBP_CLATHRIN_DEPENDENT_ENDOCYTOSIS",
    "GOBP_CLATHRIN_COAT_ASSEMBLY",
    "GOBP_RECEPTOR_MEDIATED_ENDOCYTOSIS",
    "GOBP_RECEPTOR_INTERNALIZATION",
    "GOBP_MEMBRANE_INVAGINATION",
    "GOBP_PINOCYTOSIS",
    "GOBP_PHAGOCYTOSIS",
    "GOBP_REGULATION_OF_ENDOCYTOSIS",
    "GOBP_REGULATION_OF_CLATHRIN_DEPENDENT_ENDOCYTOSIS",
    "GOBP_REGULATION_OF_RECEPTOR_INTERNALIZATION",
    "GOBP_REGULATION_OF_RECEPTOR_MEDIATED_ENDOCYTOSIS",
    "GOBP_REGULATION_OF_PHAGOCYTOSIS",
    "GOBP_NEGATIVE_REGULATION_OF_ENDOCYTOSIS",
    "GOBP_NEGATIVE_REGULATION_OF_RECEPTOR_INTERNALIZATION",
    "GOBP_NEGATIVE_REGULATION_OF_RECEPTOR_MEDIATED_ENDOCYTOSIS",
    "GOBP_NEGATIVE_REGULATION_OF_PHAGOCYTOSIS",
    "GOBP_POSITIVE_REGULATION_OF_ENDOCYTOSIS",
    "GOBP_POSITIVE_REGULATION_OF_RECEPTOR_INTERNALIZATION",
    "GOBP_POSITIVE_REGULATION_OF_RECEPTOR_MEDIATED_ENDOCYTOSIS",
    "GOBP_POSITIVE_REGULATION_OF_PHAGOCYTOSIS",
    "GOBP_G_PROTEIN_COUPLED_RECEPTOR_INTERNALIZATION",
    "GOBP_RECEPTOR_RECYCLING",
    "GOBP_REGULATION_OF_RECEPTOR_RECYCLING",
    "GOBP_REGULATION_OF_EARLY_ENDOSOME_TO_LATE_ENDOSOME_TRANSPORT",
    "GOBP_ENDOCYTIC_RECYCLING",
    "GOBP_REGULATION_OF_ENDOCYTIC_RECYCLING",
    "GOBP_ENDOSOMAL_TRANSPORT",
    "GOBP_ENDOSOMAL_VESICLE_FUSION",
    "GOBP_ENDOSOME_ORGANIZATION",
    "GOBP_ENDOSOME_TO_LYSOSOME_TRANSPORT",
    "GOBP_ENDOSOME_TO_LYSOSOME_TRANSPORT_VIA_MULTIVESICULAR_BODY_SORTING_PATHWAY",
    "GOBP_LATE_ENDOSOME_TO_LYSOSOME_TRANSPORT",
    "GOBP_LATE_ENDOSOME_TO_VACUOLE_TRANSPORT",
    "GOBP_LATE_ENDOSOME_TO_VACUOLE_TRANSPORT_VIA_MULTIVESICULAR_BODY_SORTING_PATHWAY",
    "GOBP_MULTIVESICULAR_BODY_ORGANIZATION",
    "GOBP_MULTIVESICULAR_BODY_SORTING_PATHWAY",
    "GOBP_UBIQUITIN_DEPENDENT_PROTEIN_CATABOLIC_PROCESS_VIA_THE_MULTIVESICULAR_BODY_SORTING_PATHWAY",
    "GOBP_PROTEIN_LOCALIZATION_TO_ENDOSOME",
    "GOBP_PRESYNAPTIC_ENDOCYTOSIS",
]

def _norm_gobp(name: str) -> str:
    return name.replace("GOBP_", "").replace("_", " ").lower()

def _norm_enrichr(name: str) -> str:
    name = re.sub(r"\s*\(GO:\d+\)", "", name)
    name = re.sub(r"[-/]", " ", name)
    return name.lower().strip()

def fetch_library_via_requests(library_name: str) -> dict:
    """Fetch an Enrichr gene set library directly, bypassing gseapy."""
    url = f"https://maayanlab.cloud/Enrichr/geneSetLibrary"
    params = {"mode": "text", "libraryName": library_name}
    print(f"Fetching {library_name} from Enrichr...")
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    gene_sets = {}
    for line in resp.text.splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        term = parts[0]
        genes = [g for g in parts[2:] if g]
        gene_sets[term] = genes
    print(f"  {len(gene_sets)} gene sets retrieved")
    return gene_sets

def main():
    go_sets = fetch_library_via_requests("GO_Biological_Process_2023")
    norm_to_genes = {_norm_enrichr(k): v for k, v in go_sets.items()}

    result = {}
    found, missing = [], []
    for pw in ENDOCYTOSIS_PATHWAYS:
        key = _norm_gobp(pw)
        if key in norm_to_genes:
            result[pw] = norm_to_genes[key]
            found.append(pw)
        else:
            missing.append(pw)

    print(f"\nResolved: {len(found)}/{len(ENDOCYTOSIS_PATHWAYS)} pathways")
    if missing:
        print(f"Unresolved: {missing}")

    out_path = Path(__file__).parent / "endocytosis_gene_sets.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_path}")
    all_genes = set(g for genes in result.values() for g in genes)
    print(f"Union: {len(all_genes)} unique genes across {len(result)} pathways")

if __name__ == "__main__":
    main()
