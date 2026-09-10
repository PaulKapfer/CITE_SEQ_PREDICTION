"""
ADT-to-ENSEMBL Gene Mapping
============================
Maps CITE-seq ADT feature names to ENSEMBL gene IDs using public APIs only.
No manually curated alias tables — all mappings are derived from databases.

Lookup order for each ADT name:
  1. Full name  → HGNC alias_symbol  (most authoritative for human genes)
  2. Full name  → HGNC symbol
  3. Full name  → mygene.info alias/symbol
  4. Strip clone suffix (-1/-2) → repeat 1-3
  5. Stripped name → ENSEMBL /lookup/symbol (last resort)

Clone-suffix stripping (-1, -2) is applied only as a fallback so that names
where the suffix is part of the protein identity (e.g. LOX-1 = OLR1) are
resolved correctly from their full name first.

For multi-antigen names (e.g. "CD66a/c/e", "CD11a/CD18") the first component
before '/' is used as the query term.

Transcript-level annotation is retrieved from ENSEMBL and a canonical
transcript is selected by: MANE Select > APPRIS Principal 1 > TSL 1 >
Ensembl canonical flag. The gene-level ENSG ID is reported in the output.

Dependencies: pandas, requests  (both in base conda env)
"""

import re
import sys
import time

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
ENSEMBL_REST = "https://rest.ensembl.org"
MYGENE_URL   = "https://mygene.info/v3/query"
HGNC_FETCH   = "https://rest.genenames.org/fetch"   # exact-match endpoint

# ---------------------------------------------------------------------------
# Robust HTTP helper
# ---------------------------------------------------------------------------

def _get(url, params=None, headers=None, timeout=15, retries=4):
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 15))
                time.sleep(wait)
                continue
            return resp
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# HGNC lookup  (uses /fetch for exact-field matching, not /search which is partial)
# ---------------------------------------------------------------------------

def hgnc_lookup(term):
    """
    Exact-match lookup via HGNC /fetch endpoint.
    Tries alias_symbol, then symbol, then prev_symbol.
    Returns (approved_symbol, ensembl_gene_id) or (None, None).
    """
    for field in ("alias_symbol", "symbol", "prev_symbol"):
        resp = _get(
            f"{HGNC_FETCH}/{field}/{requests.utils.quote(term)}",
            headers={"Accept": "application/json"},
        )
        time.sleep(0.12)
        if resp and resp.status_code == 200:
            docs = resp.json().get("response", {}).get("docs", [])
            if docs:
                d = docs[0]
                sym  = d.get("symbol")
                ensg = d.get("ensembl_gene_id")
                if sym and ensg:
                    return sym, ensg
    return None, None


# ---------------------------------------------------------------------------
# mygene.info lookup
# ---------------------------------------------------------------------------

def mygene_lookup(term):
    """Query mygene.info alias/symbol. Returns (symbol, ensembl_gene_id)."""
    resp = _get(
        MYGENE_URL,
        params={
            "q": f"alias:{term} OR symbol:{term}",
            "species": "human",
            "fields": "symbol,ensembl.gene",
            "size": 3,
        },
    )
    time.sleep(0.12)
    if resp and resp.status_code == 200:
        hits = resp.json().get("hits", [])
        if hits:
            hit = hits[0]
            sym = hit.get("symbol")
            ens = hit.get("ensembl", {})
            if isinstance(ens, list):
                ens = ens[0]
            ensg = ens.get("gene") if isinstance(ens, dict) else None
            if sym and ensg:
                return sym, ensg
    return None, None


# ---------------------------------------------------------------------------
# ENSEMBL symbol lookup + transcript hierarchy
# ---------------------------------------------------------------------------

_APPRIS_RANK = {
    "principal1": 0, "principal2": 1, "principal3": 2,
    "principal4": 3, "principal5": 4,
    "alternative1": 5, "alternative2": 6,
}


def _transcript_score(tx):
    """Lower score = higher priority."""
    # MANE Select is top priority
    if tx.get("is_mane_select"):
        return (0, 0, 0)

    appris = tx.get("appris_annotation") or ""
    appris_score = _APPRIS_RANK.get(appris, 99)

    tsl_raw = tx.get("transcript_support_level") or "NA"
    try:
        tsl_score = int(tsl_raw)
    except ValueError:
        tsl_score = 99

    canonical = 0 if tx.get("is_canonical") else 1

    return (1, appris_score, tsl_score, canonical)


def ensembl_lookup(symbol):
    """
    Look up *symbol* directly in ENSEMBL, expand transcripts, apply hierarchy.
    Returns (display_name, ensg_id).
    """
    resp = _get(
        f"{ENSEMBL_REST}/lookup/symbol/homo_sapiens/{requests.utils.quote(symbol)}",
        params={"expand": 1, "mane": 1},
        headers={"Content-Type": "application/json"},
    )
    time.sleep(0.12)
    if not (resp and resp.status_code == 200):
        return None, None

    data = resp.json()
    transcripts = data.get("Transcript", [])
    if transcripts:
        best = min(transcripts, key=_transcript_score)
        _ = best  # transcript chosen; gene ID is shared across all transcripts

    return data.get("display_name"), data.get("id")


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def first_component(name):
    """'CD66a/c/e' → 'CD66a';  'CD11a/CD18' → 'CD11a'."""
    return name.split("/")[0] if "/" in name else name


def strip_clone_suffix(name):
    """Remove trailing -1 or -2 (clone markers added by the panel vendor)."""
    return re.sub(r"-[12]$", "", name)


# ---------------------------------------------------------------------------
# Main lookup pipeline
# ---------------------------------------------------------------------------

def find_gene(adt_name):
    """
    Return (gene_symbol, ensembl_gene_id) for an ADT feature name.
    Tries full name first, then suffix-stripped fallback.
    """
    full    = first_component(adt_name)
    stripped = strip_clone_suffix(full)

    # Build ordered list of query terms: full name first, stripped second
    terms = [full]
    if stripped != full:
        terms.append(stripped)

    for term in terms:
        # mygene.info first — broad NCBI alias coverage
        sym, ensg = mygene_lookup(term)
        if sym and ensg:
            return sym, ensg

        # HGNC exact fetch — authoritative approved symbol + ENSG
        sym, ensg = hgnc_lookup(term)
        if sym and ensg:
            return sym, ensg

    # Final fallback: ENSEMBL direct symbol search on stripped name
    sym, ensg = ensembl_lookup(stripped)
    if sym and ensg:
        return sym, ensg

    return None, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    input_path   = (r"C:\Users\Paul\Documents\GitHub\CITE_SEQ_PREDICTION"
                    r"\01_annotation\ADT-Transcript - Mapping\Input\adt_names.csv")
    output_path  = (r"C:\Users\Paul\Documents\GitHub\CITE_SEQ_PREDICTION"
                    r"\01_annotation\ADT-Transcript - Mapping\Output\adt_rna_mapping.csv")
    control_path = (r"C:\Users\Paul\Documents\GitHub\CITE_SEQ_PREDICTION"
                    r"\01_annotation\ADT-Transcript - Mapping\Output\adt_rna_mapping_control.csv")

    df = pd.read_csv(input_path)

    results = []
    # Cache by the first query term so clones (CD4-1, CD4-2) share one API call
    cache = {}

    for adt in df["ADT_feature"]:
        cache_key = strip_clone_suffix(first_component(adt))

        if cache_key not in cache:
            sym, ensg = find_gene(adt)
            cache[cache_key] = (sym or "", ensg or "")
            flag = "OK" if ensg else "NOT FOUND"
            print(f"  [{flag:9s}] {adt:28s} -> {sym or '?':15s}  {ensg or ''}")
        else:
            sym, ensg = cache[cache_key]

        results.append({"ADT_feature": adt, "Ensembl_ID": ensg, "RNA_gene": sym})

    out_df = pd.DataFrame(results, columns=["ADT_feature", "Ensembl_ID", "RNA_gene"])
    out_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    # ------------------------------------------------------------------
    # Compare with control (informational only — divergences are expected)
    # ------------------------------------------------------------------
    ctrl   = pd.read_csv(control_path)
    merged = out_df.merge(ctrl, on="ADT_feature", suffixes=("_out", "_ctrl"))
    mismatch = merged[
        (merged["Ensembl_ID_out"] != merged["Ensembl_ID_ctrl"]) |
        (merged["RNA_gene_out"]   != merged["RNA_gene_ctrl"])
    ]
    n, total = len(mismatch), len(out_df)
    print(f"\n--- Divergences from control: {n}/{total} ({100*n/total:.1f}%) ---")
    for _, row in mismatch.iterrows():
        print(f"  {row['ADT_feature']:28s}"
              f"  data-driven=({row['Ensembl_ID_out']}, {row['RNA_gene_out']})"
              f"  control=({row['Ensembl_ID_ctrl']}, {row['RNA_gene_ctrl']})")


if __name__ == "__main__":
    main()
