"""
Build the Kotliarov ADT-to-transcript mapping table.

Unlike Hao, Kotliarov's panel CSV has no author-assigned Ensembl Gene Id —
protein names live in "Specificity" instead. Pipeline:

  1. Keep only Species == "Human" rows.
  2. Match to Hao by Clone (excluding Clone == NA): if the same antibody
     clone was already resolved in hao_adt_mapping.csv, copy its
     Ensembl_ID / RNA_gene / RNA_transcript straight across — no need to
     re-derive it from the name.
  3. Everything else: parse "Specificity" into a primary name plus any
     alternative name(s) — e.g. "CD127 (IL-7R?)" -> primary "CD127", alt
     "IL-7R"; "CX3CR1/GPR13/CCRL1" -> primary "CX3CR1", alts "GPR13",
     "CCRL1"; "HLA-A,B,C" is special-cased to "HLA-A" — then resolve via
     the same mygene -> HGNC -> Ensembl alias-search chain used for Hao's
     original ADT-name matching, trying the primary name first and falling
     back to alternates (CD-prefixed names can fail alias lookup where the
     bracketed alias succeeds).
  4. Add Clone_in_Hao (Yes/No): whether this row's clone appears anywhere
     in Hao's antibody panel at all (independent of whether that Hao row
     had a usable Ensembl_ID).

Input : Input/antibody_info_Kotliarov.csv   (cp1252-encoded)
        Output2/hao_adt_mapping.csv          (from process_hao.py)
Output: Output2/kotliarov_adt_mapping.csv
        Columns: ADT_feature, Clone, Species, Ensembl_ID, RNA_gene,
                 RNA_transcript, RNA_transcript_source, Clone_in_Hao,
                 Match_method
"""

import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lookup_utils import find_gene_from_candidates, resolve_transcripts_batch

BASE            = r"C:\Users\Paul\Desktop\Publications\CITE-SEQ_pred\Matching_ADT_Transcript"
KOT_INPUT_PATH  = os.path.join(BASE, "Input", "antibody_info_Kotliarov.csv")
HAO_INPUT_PATH  = os.path.join(BASE, "Input", "antibody_info_Hao.CSV")
HAO_MAPPED_PATH = os.path.join(BASE, "Output2", "hao_adt_mapping.csv")
OUTPUT_PATH     = os.path.join(BASE, "Output2", "kotliarov_adt_mapping.csv")

# Exact-string overrides for names that don't fit the general "NAME (ALT)" /
# "NAME/ALT" parsing pattern.
SPECIAL_CASES = {
    "HLA-A,B,C": ["HLA-A"],
}

# Known-problematic loci where even a direct, current-symbol Ensembl lookup
# is unreliable: MHC (HLA) and immunoglobulin regions have multiple
# overlapping gene models sharing the same display name (primary assembly +
# alternate-haplotype contigs), and repeated /lookup/symbol calls for the
# same term have been observed to return different ones across runs. Verified
# manually against Ensembl and hardcoded directly — bypasses name lookup
# entirely for these, the same way Hao's author-assigned IDs do.
DIRECT_ENSEMBL_OVERRIDES = {
    "HLA-A,B,C": "ENSG00000206503",  # HLA-A, primary chr6 assembly (symbol
                                       # lookup intermittently returns the
                                       # HSCHR6_MHC_DBB_CTG1 alt-contig copy)
    "IgA":       "ENSG00000211895",  # IGHA1, primary chr14 assembly (alias
                                       # search fuzzy-matches "IgA" to CD79A;
                                       # even a direct IGHA1 symbol lookup
                                       # has returned an alt-contig copy)
}

PAREN_RE = re.compile(r"^(.*?)\s*\(([^)]*)\)\s*$")


def parse_name_candidates(raw_name):
    """
    Return an ordered list of candidate query terms for a Specificity string:
    primary name first, then any alternate name(s) found in parentheses or
    slash-separated synonyms. Stray encoding artefacts ('?' standing in for
    an unrenderable character, e.g. Greek alpha) are stripped.
    """
    name = raw_name.strip()
    if name in SPECIAL_CASES:
        return list(SPECIAL_CASES[name])

    candidates = []
    m = PAREN_RE.match(name)
    if m:
        primary, alt_group = m.group(1).strip(), m.group(2).strip()
        candidates.append(primary)
        candidates.extend(a.strip() for a in alt_group.split(",") if a.strip())
    elif "/" in name:
        candidates.extend(p.strip() for p in name.split("/") if p.strip())
    else:
        candidates.append(name)

    cleaned = []
    for c in candidates:
        c = c.replace("?", "").strip(" ,")
        if c:
            cleaned.append(c)
    return cleaned


def main():
    kot = pd.read_csv(KOT_INPUT_PATH, encoding="cp1252", dtype=str, keep_default_na=True)
    print(f"Loaded {len(kot)} Kotliarov antibody rows")

    kot = kot.rename(columns={"Specificity": "ADT_feature"})

    n_before = len(kot)
    kot = kot[kot["Species"].notna() & (kot["Species"].str.strip() == "Human")].copy()
    print(f"Kept {len(kot)}/{n_before} row(s) with Species == 'Human'")

    # All Hao clones (regardless of whether that Hao row had a usable Ensembl
    # ID) — used only for the informational Clone_in_Hao flag.
    hao_all = pd.read_csv(HAO_INPUT_PATH, encoding="cp1252", dtype=str, keep_default_na=True)
    hao_all_clones = set(hao_all["Clone"].dropna().str.strip())

    # Resolved Hao rows — used to copy Ensembl_ID/RNA_gene/RNA_transcript
    # across for clone matches.
    hao_mapped = pd.read_csv(HAO_MAPPED_PATH, dtype=str, keep_default_na=True)
    hao_by_clone = {}
    for _, row in hao_mapped.iterrows():
        clone = str(row["Clone"]).strip() if pd.notna(row["Clone"]) else ""
        if clone and clone not in hao_by_clone:
            hao_by_clone[clone] = row

    kot["Clone_stripped"] = kot["Clone"].apply(lambda c: c.strip() if isinstance(c, str) else "")
    kot["Clone_in_Hao"] = kot["Clone_stripped"].apply(
        lambda c: "Yes" if c and c in hao_all_clones else "No")

    ensg_col, gene_col, enst_col, src_col, method_col = [], [], [], [], []
    unresolved_rows = []  # (idx, candidates) for batch-friendly programmatic resolution

    for idx, row in kot.iterrows():
        clone = row["Clone_stripped"]
        if clone and clone in hao_by_clone:
            hao_row = hao_by_clone[clone]
            ensg_col.append(hao_row["Ensembl_ID"])
            gene_col.append(hao_row["RNA_gene"])
            enst_col.append(hao_row["RNA_transcript"])
            src_col.append(hao_row["RNA_transcript_source"])
            method_col.append(f"hao_clone_match:{clone}")
        else:
            ensg_col.append(None); gene_col.append(None)
            enst_col.append(None); src_col.append(None)
            method_col.append(None)  # filled in below
            unresolved_rows.append(idx)

    kot["Ensembl_ID"]            = ensg_col
    kot["RNA_gene"]              = gene_col
    kot["RNA_transcript"]        = enst_col
    kot["RNA_transcript_source"] = src_col
    kot["Match_method"]          = method_col

    n_clone_matched = len(kot) - len(unresolved_rows)
    print(f"Matched {n_clone_matched} row(s) to Hao by Clone; "
          f"{len(unresolved_rows)} row(s) need programmatic name resolution")

    # ── Programmatic name resolution for everything else ──────────────────
    print("\nResolving remaining rows by name (Ensembl direct -> mygene -> HGNC) ...")
    for i, idx in enumerate(unresolved_rows, 1):
        raw_name = kot.at[idx, "ADT_feature"]

        if raw_name.strip() in DIRECT_ENSEMBL_OVERRIDES:
            ensg = DIRECT_ENSEMBL_OVERRIDES[raw_name.strip()]
            print(f"  [{i:3d}/{len(unresolved_rows)}] [OVERRIDE  ] {raw_name:35s} -> {ensg} (hardcoded)")
            kot.at[idx, "Ensembl_ID"]   = ensg
            kot.at[idx, "Match_method"] = f"direct_override:{ensg}"
            continue

        candidates = parse_name_candidates(raw_name)
        sym, ensg, matched_term = find_gene_from_candidates(candidates)
        status = "OK" if ensg else "NOT FOUND"
        print(f"  [{i:3d}/{len(unresolved_rows)}] [{status:9s}] {raw_name:35s} "
              f"candidates={candidates} -> {sym or '?'} {ensg or ''}")
        if ensg:
            kot.at[idx, "Ensembl_ID"]   = ensg
            kot.at[idx, "RNA_gene"]     = sym
            kot.at[idx, "Match_method"] = f"name_lookup:{matched_term}"
        else:
            kot.at[idx, "Match_method"] = "unresolved"

    # Batch-resolve transcripts for every programmatically-found Ensembl ID
    # that doesn't already have one copied from Hao.
    need_transcript = sorted(set(
        kot.loc[kot["RNA_transcript"].isna() & kot["Ensembl_ID"].notna(), "Ensembl_ID"]
    ))
    if need_transcript:
        print(f"\nResolving transcripts for {len(need_transcript)} "
              f"programmatically-matched Ensembl gene ID(s) ...")
        resolved = resolve_transcripts_batch(need_transcript)
        for idx in kot.index:
            if pd.isna(kot.at[idx, "RNA_transcript"]) and pd.notna(kot.at[idx, "Ensembl_ID"]):
                ensg = kot.at[idx, "Ensembl_ID"]
                sym, enst, source = resolved.get(ensg, ("", "", ""))
                kot.at[idx, "RNA_transcript"]        = enst
                kot.at[idx, "RNA_transcript_source"] = source
                if sym and not kot.at[idx, "RNA_gene"]:
                    kot.at[idx, "RNA_gene"] = sym

    has_transcript = kot["RNA_transcript"].notna() & (kot["RNA_transcript"].fillna("").str.strip() != "")
    n_ok   = has_transcript.sum()
    n_fail = len(kot) - n_ok
    print(f"\nFinal: {n_ok}/{len(kot)} row(s) resolved to a transcript, {n_fail} unresolved")
    if n_fail:
        print(kot.loc[kot["RNA_transcript"].isna(),
                       ["ADT_feature", "Clone", "Match_method"]].to_string(index=False))

    out_cols = ["ADT_feature", "Clone", "Species", "Ensembl_ID", "RNA_gene",
                "RNA_transcript", "RNA_transcript_source", "Clone_in_Hao", "Match_method"]
    kot[out_cols].to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}  ({len(kot)} rows)")


if __name__ == "__main__":
    main()
