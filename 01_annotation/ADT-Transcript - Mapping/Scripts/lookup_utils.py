"""
Shared Ensembl / gene-symbol lookup helpers for the Hao + Kotliarov
antibody-panel matching pipeline.

Two lookup directions are needed:
  1. ENSG -> (gene_symbol, canonical/MANE transcript)   [batched, for rows
     that already have an author-assigned Ensembl Gene Id — Hao, and any
     Kotliarov row matched to Hao by Clone]
  2. protein/alias name -> ENSG -> (gene_symbol, transcript)   [per-name,
     for Kotliarov rows with no Clone match to Hao — same alias-search
     chain used by the original map_adt_to_ensembl.py: mygene.info ->
     HGNC exact fetch -> direct Ensembl symbol lookup]
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ENSEMBL_REST = "https://rest.ensembl.org"
MYGENE_URL   = "https://mygene.info/v3/query"
HGNC_FETCH   = "https://rest.genenames.org/fetch"

HEADERS             = {"Content-Type": "application/json", "Accept": "application/json"}
POST_BATCH_SIZE     = 80
POST_WORKERS        = 4   # concurrent batch requests (I/O-bound — real speedup, not just fewer round-trips)
MAX_RETRIES         = 6
RETRY_PAUSE         = 15.0
RETRYABLE_STATUSES  = (429, 500, 502, 503, 504)


# ---------------------------------------------------------------------------
# Low-level HTTP helpers (retry-aware)
# ---------------------------------------------------------------------------

def _get(url, params=None, headers=None, timeout=20, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.exceptions.RequestException:
            time.sleep(RETRY_PAUSE)
            continue
        if resp.status_code == 200:
            return resp
        if resp.status_code in RETRYABLE_STATUSES:
            wait = float(resp.headers.get("Retry-After", RETRY_PAUSE))
            time.sleep(wait)
            continue
        return resp
    return None


def ensembl_post(path, ids, params=None):
    """POST {ids} to an Ensembl REST path with retry. Returns parsed JSON or {}."""
    url = ENSEMBL_REST + path
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, headers=HEADERS, params=params,
                                  json={"ids": ids}, timeout=60)
        except requests.exceptions.RequestException:
            time.sleep(RETRY_PAUSE)
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in RETRYABLE_STATUSES:
            time.sleep(RETRY_PAUSE)
            continue
        print(f"  POST {path} -> HTTP {resp.status_code}: {resp.text[:200]}")
        return {}
    print(f"  POST {path} failed after {MAX_RETRIES} attempts — batch of {len(ids)} unresolved")
    return {}


# ---------------------------------------------------------------------------
# Direction 1: ENSG -> (symbol, transcript, source)  [batched]
# ---------------------------------------------------------------------------

def _select_transcript(gene_detail):
    for t in gene_detail.get("Transcript", []):
        if t.get("is_mane_select"):
            return t["id"].split(".")[0], "mane_select"
    canonical = gene_detail.get("canonical_transcript", "")
    if canonical:
        return canonical.split(".")[0], "canonical"
    return "", ""


def _resolve_chunk(chunk):
    data = ensembl_post("/lookup/id", chunk, params={"expand": 1, "mane": 1})
    out = {}
    for ensg in chunk:
        gene_detail = data.get(ensg)
        if gene_detail:
            enst, source = _select_transcript(gene_detail)
            symbol = gene_detail.get("display_name", "")
        else:
            symbol, enst, source = "", "", ""
        out[ensg] = (symbol, enst, source)
    return out


def resolve_transcripts_batch(ensg_list):
    """{ensg: (symbol, enst_id, source)} via POST /lookup/id, expand=1+mane=1.

    Batches are fetched concurrently (POST_WORKERS threads) — this is
    I/O-bound work, so real wall-clock speedup, not just fewer round-trips.
    """
    results = {}
    ids = sorted(set(e for e in ensg_list if e))
    chunks = [ids[i:i + POST_BATCH_SIZE] for i in range(0, len(ids), POST_BATCH_SIZE)]
    if not chunks:
        return results

    print(f"  Resolving {len(ids)} gene ID(s) in {len(chunks)} batch(es) "
          f"({POST_WORKERS} concurrent) ...")
    with ThreadPoolExecutor(max_workers=min(POST_WORKERS, len(chunks))) as pool:
        futures = {pool.submit(_resolve_chunk, c): i for i, c in enumerate(chunks)}
        n_done = 0
        for fut in as_completed(futures):
            results.update(fut.result())
            n_done += 1
            print(f"    batch {n_done}/{len(chunks)} done")
    return results


# ---------------------------------------------------------------------------
# Direction 2: protein/alias name -> (symbol, ensg)   [per-name alias search]
# ---------------------------------------------------------------------------

def mygene_lookup(term):
    resp = _get(MYGENE_URL, params={
        "q": f"alias:{term} OR symbol:{term}",
        "species": "human",
        "fields": "symbol,ensembl.gene",
        "size": 3,
    })
    if resp is not None and resp.status_code == 200:
        hits = resp.json().get("hits", [])
        if hits:
            hit = hits[0]
            sym = hit.get("symbol")
            ens = hit.get("ensembl", {})
            if isinstance(ens, list):
                ens = ens[0] if ens else {}
            ensg = ens.get("gene") if isinstance(ens, dict) else None
            if sym and ensg:
                return sym, ensg
    return None, None


def hgnc_lookup(term):
    for field in ("alias_symbol", "symbol", "prev_symbol"):
        resp = _get(f"{HGNC_FETCH}/{field}/{requests.utils.quote(term)}",
                     headers={"Accept": "application/json"})
        if resp is not None and resp.status_code == 200:
            docs = resp.json().get("response", {}).get("docs", [])
            if docs:
                d = docs[0]
                sym, ensg = d.get("symbol"), d.get("ensembl_gene_id")
                if sym and ensg:
                    return sym, ensg
    return None, None


def ensembl_symbol_lookup(term):
    resp = _get(f"{ENSEMBL_REST}/lookup/symbol/homo_sapiens/{requests.utils.quote(term)}",
                params={"expand": 0}, headers=HEADERS)
    if resp is not None and resp.status_code == 200:
        data = resp.json()
        return data.get("display_name"), data.get("id")
    return None, None


def find_gene_by_name(term):
    """
    (symbol, ensg) for a single query term, or (None, None).

    Direct Ensembl symbol lookup is tried first: for a term that's already a
    gene's current official symbol, it's an exact match and reliably returns
    the primary-assembly gene. mygene.info/HGNC alias search is tried only
    as a fallback (needed for non-official names like "CD10" -> MME) because
    it can return an alternate-haplotype-contig copy of the gene instead of
    the primary one for terms that coincide with an old alias — e.g. "HLA-A"
    alias-matched to a copy on the MHC alt-contig HSCHR6_MHC_DBB_CTG1 rather
    than the primary chr6 gene, and "IgA" alias-fuzzy-matched to CD79A (whose
    old alias is "Ig-alpha") instead of failing to match an antibody isotype
    that isn't a single gene at all.
    """
    for fn in (ensembl_symbol_lookup, mygene_lookup, hgnc_lookup):
        sym, ensg = fn(term)
        if sym and ensg:
            return sym, ensg
    return None, None


def find_gene_from_candidates(candidates):
    """
    Try each candidate name in order (primary name first, then aliases),
    each through the full Ensembl -> mygene -> HGNC chain. Returns
    (symbol, ensg, matched_term) for the first candidate that resolves.
    """
    for term in candidates:
        term = term.strip()
        if not term:
            continue
        sym, ensg = find_gene_by_name(term)
        if sym and ensg:
            return sym, ensg, term
    return None, None, None
