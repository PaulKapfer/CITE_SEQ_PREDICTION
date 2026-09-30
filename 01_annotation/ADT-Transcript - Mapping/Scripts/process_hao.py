"""
Build the Hao ADT-to-transcript mapping table.

Hao's antibody panel CSV already carries an author-assigned "Ensembl Gene Id"
per antibody, so no name-based gene lookup is needed here — only resolving
each Ensembl gene ID to its MANE Select (preferred) or canonical transcript,
plus the current gene symbol (for downstream scripts that key RNA columns by
symbol, e.g. "RNA_{gene}").

Rows with no author-assigned Ensembl Gene Id (isotype controls, TCR variable
region clones) are dropped — there's no single target gene to resolve.

Input : Input/antibody_info_Hao.CSV   (cp1252-encoded)
Output: Output2/hao_adt_mapping.csv
        Columns: ADT_feature, Clone, Ensembl_ID, RNA_gene, RNA_transcript,
                 RNA_transcript_source
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lookup_utils import resolve_transcripts_batch

BASE        = r"C:\Users\Paul\Desktop\Publications\CITE-SEQ_pred\Matching_ADT_Transcript"
INPUT_PATH  = os.path.join(BASE, "Input", "antibody_info_Hao.CSV")
OUTPUT_PATH = os.path.join(BASE, "Output2", "hao_adt_mapping.csv")


def main():
    df = pd.read_csv(INPUT_PATH, encoding="cp1252", dtype=str, keep_default_na=True)
    print(f"Loaded {len(df)} Hao antibody rows")

    df = df.rename(columns={"#protein": "ADT_feature", "Ensembl Gene Id": "Ensembl_ID"})

    # The antibody CSV's "#protein" names use "_" for clone-number suffixes
    # (e.g. "CD3_1", "Integrin_7"), but the actual Hao h5ad / reference_data
    # ADT_<name> columns use "-" instead ("ADT_CD3-1", "ADT_Integrin-7").
    # Every underscore in this column is one of these clone-suffix/isotype
    # names (verified: no other compound names use "_"), so a blanket
    # substitution safely restores the real single-cell object naming.
    n_renamed = df["ADT_feature"].str.contains("_", na=False).sum()
    df["ADT_feature"] = df["ADT_feature"].str.replace("_", "-", regex=False)
    print(f"Renamed {n_renamed} ADT_feature value(s) '_' -> '-' to match h5ad ADT_<name> column naming")

    # A handful of rows carry a comma-separated list of Ensembl IDs instead
    # of one: either a mouse ID alongside the human one (e.g. CD11b_1 ->
    # "ENSMUSG00000030786, ENSG00000169896" — the antibody's cross-reactivity
    # noted, not two targets), or, for CD11a/CD18, genuinely two different
    # human genes (a heterodimer, ITGAL + ITGB2). Keep only human (ENSG*)
    # IDs, and — matching this project's established convention for
    # multi-antigen ADT names (map_adt_to_ensembl.py: first component before
    # "/") — use the first human ID when more than one is present.
    def clean_ensembl_id(raw):
        if pd.isna(raw):
            return raw
        parts = [p.strip() for p in str(raw).split(",")]
        human = [p for p in parts if p.startswith("ENSG")]
        if not human:
            return None
        return human[0]

    multi_valued = df.loc[df["Ensembl_ID"].astype(str).str.contains(",", na=False),
                           ["ADT_feature", "Ensembl_ID"]]
    if len(multi_valued):
        print(f"Cleaning {len(multi_valued)} row(s) with multi-value Ensembl_ID cells:")
        for _, r in multi_valued.iterrows():
            print(f"  {r['ADT_feature']:15s} {r['Ensembl_ID']}")
    df["Ensembl_ID"] = df["Ensembl_ID"].map(clean_ensembl_id)

    n_before = len(df)
    df = df[df["Ensembl_ID"].notna() & (df["Ensembl_ID"].str.strip() != "")].copy()
    print(f"Dropped {n_before - len(df)} row(s) with no Ensembl Gene Id "
          f"(isotype controls / TCR clones) — {len(df)} remaining")

    ensg_list = sorted(df["Ensembl_ID"].unique())
    print(f"Resolving transcripts for {len(ensg_list)} unique Ensembl gene ID(s) ...")
    resolved = resolve_transcripts_batch(ensg_list)

    df["RNA_gene"]              = df["Ensembl_ID"].map(lambda e: resolved.get(e, ("", "", ""))[0])
    df["RNA_transcript"]        = df["Ensembl_ID"].map(lambda e: resolved.get(e, ("", "", ""))[1])
    df["RNA_transcript_source"] = df["Ensembl_ID"].map(lambda e: resolved.get(e, ("", "", ""))[2])

    n_ok   = (df["RNA_transcript"] != "").sum()
    n_fail = len(df) - n_ok
    print(f"Resolved: {n_ok} OK, {n_fail} unresolved")
    if n_fail:
        print("  Unresolved rows:")
        print(df.loc[df["RNA_transcript"] == "", ["ADT_feature", "Ensembl_ID"]].to_string(index=False))

    out_cols = ["ADT_feature", "Clone", "Ensembl_ID", "RNA_gene",
                "RNA_transcript", "RNA_transcript_source"]
    out = df[out_cols]
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
