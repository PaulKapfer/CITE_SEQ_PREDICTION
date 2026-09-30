"""
Combine the Hao and Kotliarov ADT-to-transcript mapping tables into a single
list for RNA-FM embedding and downstream transcript-feature scripts.

SCOPE: RNA-FM / transcript features ONLY. Because rows are deduplicated by
RNA_transcript, antibodies that target the same transcript collapse into one
row (e.g. CD45-1/CD45-2/CD45RA/CD45RB/CD45RO all share ENST00000442510).
That is correct here — each transcript needs exactly one embedding — but it is
wrong for model training, where each antibody clone is its own prediction
target. Model training uses combine_hao_kotliarov_MODEL-TRAINING.py, which
deduplicates by Clone instead.

Rows are deduplicated by RNA_transcript (not ADT_feature/gene symbol) —
requested explicitly so the embedding step computes each transcript exactly
once even though the same gene can appear under both panels. Hao is listed
first, so when a transcript is covered by both panels (e.g. a Kotliarov
clone matched to Hao) the Hao row — whose ADT_feature name matches the
actual Hao h5ad / reference_data.parquet column naming — wins; the
Kotliarov duplicate of that same transcript is dropped. Kotliarov rows for
antibodies/genes Hao doesn't have survive and add to the combined transcript
set (useful for a wider RNA-FM gene universe even though today's training
data is Hao-cell-based; Ablation_study / Final_Run naturally ignore any
ADT_feature that doesn't have a matching ADT_<name> column in that data).

Input : Output2/hao_adt_mapping.csv
        Output2/kotliarov_adt_mapping.csv
Output: Output2/combined_adt_mapping.csv
        Columns: ADT_feature, Ensembl_ID, RNA_gene, RNA_transcript,
                 RNA_transcript_source, Source
"""

import os

import pandas as pd

BASE        = r"C:\Users\Paul\Desktop\Publications\CITE-SEQ_pred\Matching_ADT_Transcript"
HAO_PATH    = os.path.join(BASE, "Output2", "hao_adt_mapping.csv")
KOT_PATH    = os.path.join(BASE, "Output2", "kotliarov_adt_mapping.csv")
OUTPUT_PATH = os.path.join(BASE, "Output2", "combined_adt_mapping.csv")

COLS = ["ADT_feature", "Ensembl_ID", "RNA_gene", "RNA_transcript", "RNA_transcript_source"]


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
    n_with_transcript = len(combined)
    print(f"{n_total} total rows (Hao {len(hao_out)} + Kotliarov {len(kot_out)}); "
          f"{n_with_transcript} have a resolved transcript")

    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset="RNA_transcript", keep="first")
    print(f"Deduplicated by RNA_transcript: {before_dedup} -> {len(combined)} "
          f"({before_dedup - len(combined)} doublet(s) dropped, Hao kept on conflict)")

    source_counts = combined["Source"].value_counts()
    print(f"Final combined set by source: {dict(source_counts)}")

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}  ({len(combined)} rows)")


if __name__ == "__main__":
    main()
