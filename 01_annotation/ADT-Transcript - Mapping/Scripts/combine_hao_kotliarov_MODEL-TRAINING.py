"""
Combine the Hao and Kotliarov ADT-to-transcript mapping tables into the
antibody-level list used for model training and evaluation.

SCOPE: model training / testing. This is the counterpart to
combine_hao_kotliarov_RNA-FM.py and differs in exactly one way — the
deduplication key.

Why not deduplicate by RNA_transcript
-------------------------------------
The RNA-FM variant deduplicates by RNA_transcript because each transcript
needs exactly one embedding. Applying that key to training data silently
discards antibodies: Hao panels several distinct clones against the same
transcript, and only the first survives. ENST00000442510 (PTPRC) alone covers
CD45-1, CD45-2, CD45RA, CD45RB and CD45RO — five different antibodies, five
different epitopes, five different measured ADT columns — collapsed to one.
Training on the transcript-deduplicated table therefore saw 190 usable
proteins instead of 204, losing CD45RA, CD45RB, CD45RO, CD8a, TCR-V-9 and the
-2 clones (CD4-2, CD11b-2, CD38-2, CD44-2, CD45-2, CD56-2, CD133-2, CD138-2,
CD275-2).

Deduplication key: Clone
------------------------
One row per antibody clone. A clone is the actual reagent that produced an
ADT_<name> column in the reference data, so it is the correct unit for a
prediction target. Hao is concatenated first, so when the same clone appears
in both panels the Hao row wins and its ADT_feature naming — which matches the
ADT_<name> columns in reference_data.parquet — is preserved.

Two clones in Hao are genuinely identical (CD26-1 and CD26-2 are both BA5b);
the duplicate is dropped, taking Hao from 219 to 218 rows.

Clone strings are free text and are formatted differently across panels, so
they are compared on a normalised key (_clone_key), while the original string
is kept in the output:
  - case and whitespace:        Hao "clone 7"  vs Kotliarov "Clone 7"   (CD133)
  - Excel scientific notation:  Hao "5.00E+10" vs Kotliarov "5E10"      (CD90)
    — the Hao sheet passed through Excel, which turned clone 5E10 into a
    number; 5.00E+08, 1.00E+03 and 9.00E+02 carry the same damage
  - parenthetical aliases:      Hao "Ber-ACT35 (ACT35)" vs "Ber-ACT35"  (CD134)
Every Hao row is a distinct ADT channel with its own barcode, even where the
sheet assigns two rows one clone (CD26-1 / CD26-2, since relabelled BA5b*), so
the script refuses to run if normalisation would merge two Hao rows.

A final pass deduplicates by ADT_feature as a safety net. ADT_feature must be
unique because downstream scripts use it both to look up the ADT_<name> column
and to assign protein_id.

Multiple antibodies per gene
----------------------------
Antibodies sharing an RNA gene are kept as separate rows and become separate
protein-RNA training pairs. They share the gene's RNA-FM embedding and
RNA_<gene> expression column (RNA-FM features are stored per HGNC gene symbol
and loaded by symbol) but receive distinct protein_id values, so the model can
learn clone-specific offsets. The gene-level CV split in the training scripts
assigns folds per gene, so all antibodies against one gene stay in the same
fold and no clone leaks between train and test.

Input : Output2/hao_adt_mapping.csv
        Output2/kotliarov_adt_mapping.csv
Output: Output2/combined_adt_mapping_MODEL-TRAINING.csv
        Columns: ADT_feature, Clone, Ensembl_ID, RNA_gene, RNA_transcript,
                 RNA_transcript_source, Source
"""

import os
import re

import pandas as pd

BASE        = r"C:\Users\Paul\Desktop\Publications\CITE-SEQ_pred\Matching_ADT_Transcript"
HAO_PATH    = os.path.join(BASE, "Output2", "hao_adt_mapping.csv")
KOT_PATH    = os.path.join(BASE, "Output2", "kotliarov_adt_mapping.csv")
OUTPUT_PATH = os.path.join(BASE, "Output2", "combined_adt_mapping_MODEL-TRAINING.csv")

COLS = ["ADT_feature", "Clone", "Ensembl_ID", "RNA_gene", "RNA_transcript",
        "RNA_transcript_source"]


def _clone_key(clone: str) -> str:
    s = str(clone).strip().lower()
    s = re.sub(r"\s*\([^)]*\)", "", s)
    m = re.fullmatch(r"(\d+)\.0*e\+?0*(\d+)", s)
    if m:
        s = f"{m.group(1)}e{m.group(2)}"
    return re.sub(r"\s+", " ", s).strip()


def main():
    hao = pd.read_csv(HAO_PATH, dtype=str, keep_default_na=True)
    kot = pd.read_csv(KOT_PATH, dtype=str, keep_default_na=True)

    hao_out = hao[COLS].copy()
    hao_out["Source"] = "Hao"

    kot_out = kot[COLS].copy()
    kot_out["Source"] = "Kotliarov"

    combined = pd.concat([hao_out, kot_out], ignore_index=True)
    n_total = len(combined)

    combined = combined[combined["RNA_transcript"].notna()
                         & (combined["RNA_transcript"].str.strip() != "")]
    print(f"{n_total} total rows (Hao {len(hao_out)} + Kotliarov {len(kot_out)}); "
          f"{len(combined)} have a resolved transcript")

    missing_clone = combined["Clone"].isna() | (combined["Clone"].str.strip() == "")
    if missing_clone.any():
        raise ValueError(
            f"{int(missing_clone.sum())} row(s) have no Clone and cannot be "
            f"deduplicated: {combined.loc[missing_clone, 'ADT_feature'].tolist()}"
        )

    combined["_clone_key"] = combined["Clone"].map(_clone_key)

    hao_keys = combined.loc[combined["Source"] == "Hao"]
    clash = hao_keys[hao_keys.duplicated("_clone_key", keep=False)]
    if len(clash):
        raise ValueError(
            "Clone normalisation would merge distinct Hao ADT channels:\n"
            + clash[["ADT_feature", "Clone", "_clone_key"]].to_string(index=False)
        )

    dropped = combined[combined.duplicated("_clone_key", keep="first")]
    before = len(combined)
    combined = combined.drop_duplicates(subset="_clone_key", keep="first")
    print(f"Deduplicated by normalised Clone: {before} -> {len(combined)} "
          f"({before - len(combined)} duplicate clone(s) dropped, Hao kept on conflict)")
    kept = combined.set_index("_clone_key")
    for _, r in dropped.iterrows():
        w = kept.loc[r["_clone_key"]]
        if r["Clone"] != w["Clone"]:
            print(f"  matched only after normalisation: {r['Source']} {r['ADT_feature']!r} "
                  f"clone {r['Clone']!r} == {w['Source']} {w['ADT_feature']!r} clone {w['Clone']!r}")
    combined = combined.drop(columns="_clone_key")

    before = len(combined)
    dup_names = combined.loc[combined.duplicated("ADT_feature", keep=False), "ADT_feature"]
    combined = combined.drop_duplicates(subset="ADT_feature", keep="first")
    if before != len(combined):
        print(f"Deduplicated by ADT_feature: {before} -> {len(combined)} "
              f"(clone spelling differs across panels for: "
              f"{sorted(set(dup_names))})")

    per_gene = combined.groupby("RNA_gene").size()
    multi = per_gene[per_gene > 1].sort_values(ascending=False)
    print(f"\nFinal set by source: {dict(combined['Source'].value_counts())}")
    print(f"{combined['RNA_gene'].nunique()} unique RNA genes across "
          f"{len(combined)} antibodies; {len(multi)} gene(s) carry >1 antibody")
    for gene, n in multi.items():
        names = combined.loc[combined["RNA_gene"] == gene, "ADT_feature"].tolist()
        print(f"  {gene:<10} {n}  {names}")

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}  ({len(combined)} rows)")


if __name__ == "__main__":
    main()
