#!/usr/bin/env python3
"""
RNA-FM Feature Extraction — Final Run
======================================
Reads gene symbols AND their pre-resolved canonical transcript IDs from
combined_adt_mapping_RNA_FM.csv (columns RNA_gene, Ensembl_ID, RNA_transcript),
and computes 640-dimensional embeddings using RNA-FM (rna_fm_t12).

combined_adt_mapping_RNA_FM.csv is the union of Hao's and Kotliarov's ADT antibody
panels, resolved to one canonical/MANE transcript per gene by
Matching_ADT_Transcript/Scripts2/{process_hao,process_kotliarov,
combine_hao_kotliarov}.py, already deduplicated by RNA_transcript (Hao wins
on conflict — see combine_hao_kotliarov.py) — this script does NOT re-resolve
transcripts or re-deduplicate itself, only embeds each transcript once.

Input:
  ADT_MAPPING_CSV – CSV with columns ADT_feature, Ensembl_ID, RNA_gene,
                    RNA_transcript, RNA_transcript_source, Source
                    (Matching_ADT_Transcript\\Output2\\combined_adt_mapping_RNA_FM.csv)

Pipeline
--------
1. Load combined_adt_mapping_RNA_FM.csv → unique RNA_gene list with its
   pre-resolved Ensembl_ID and RNA_transcript.
2. Load metadata.csv from a previous run of this script (if present) as a
   cache: gene_symbol -> {canonical_transcript_id, ...}. A gene is reused
   without any network call when the cached transcript ID still matches the
   current RNA_transcript and its .npy is still on disk — otherwise (new
   gene, or the shared mapping resolved a different transcript since the
   cache was written) it's recomputed, overwriting any stale .npy.
3. Batch-fetch cDNA sequence for every non-cached gene's transcript via
   POST /sequence/id?type=cdna (chunked, not one call per gene).
4. Convert T → U (DNA → RNA alphabet).
5. Run RNA-FM (rna_fm_t12, 640-dim).
   Positional embedding table is extended to 16,000 nucleotides via periodic
   tiling to accommodate all relevant transcript lengths.
6. Extract representations from the final transformer layer (layer 12) and
   mean-pool across all nucleotide positions, excluding <cls> and <eos>,
   yielding a single 640-dimensional feature vector per gene.
7. Save one .npy per gene symbol to OUT_DIR.
8. Save metadata.csv with per-gene status, sequence length, transcript ID, etc.

Dependencies
------------
  pip install rna-fm torch pandas requests
"""

import os
import sys
import time
import logging
import threading
import requests
import numpy as np
import pandas as pd
import torch

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
ADT_MAPPING_CSV = r"C:\Users\Paul\Desktop\Publications\CITE-SEQ_pred\Matching_ADT_Transcript\Output2\combined_adt_mapping_RNA_FM.csv"

OUT_DIR = r"C:\Users\Paul\Desktop\Publications\CITE-SEQ_pred\Create_RNA_FM_features\Output\rnafm_features"

# RNA-FM
RNA_FM_MODEL = "rna_fm_t12"   # 12-layer, 640-dim
REPR_LAYER   = 12
PE_TARGET    = 16_002          # extend positional table to ~16,000 nt

# Ensembl API
N_TORCH_THREADS = 5
ENSEMBL_REST    = "https://rest.ensembl.org"
API_RATE_SEC    = 0.1
MAX_RETRIES     = 5
RETRY_PAUSE     = 12.0
POST_BATCH_SIZE = 50   # Ensembl POST /lookup/id and /sequence/id both cap around here

# ── SETUP ──────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

# Windows consoles default to cp1252, which can't encode box-drawing/unicode
# characters used in log messages (e.g. the "── Summary ──" divider) and
# raises UnicodeEncodeError mid-run. Force UTF-8 on stdout so any log message
# is safe regardless of the terminal's codepage.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUT_DIR, "rnafm_features.log"), mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

HEADERS        = {"Content-Type": "application/json", "Accept": "application/json"}
_api_lock      = threading.Lock()
_last_api_call = 0.0


def _rate_limited_sleep():
    global _last_api_call
    with _api_lock:
        delta = API_RATE_SEC - (time.monotonic() - _last_api_call)
        if delta > 0:
            time.sleep(delta)
        _last_api_call = time.monotonic()


# ── ENSEMBL HELPERS ────────────────────────────────────────────────────────────
# HTTP statuses worth retrying: 429 (rate limited), 502/503/504 (proxy/gateway
# hiccups), and 500 — Ensembl's REST layer returns 500 for what are usually
# transient overload blips rather than a permanently broken endpoint.
RETRYABLE_STATUSES = (429, 500, 502, 503, 504)


def ensembl_get(path, params=None, text=False):
    url  = ENSEMBL_REST + path
    hdrs = {"Accept": "text/plain"} if text else HEADERS
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        _rate_limited_sleep()
        try:
            resp = requests.get(url, headers=hdrs, params=params, timeout=30)
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            log.warning(f"  network error ({exc.__class__.__name__}) on {path} — "
                        f"waiting {RETRY_PAUSE}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(RETRY_PAUSE)
            continue

        if resp.status_code == 200:
            return resp.text if text else resp.json()
        if resp.status_code in RETRYABLE_STATUSES:
            log.warning(f"  HTTP {resp.status_code} — waiting {RETRY_PAUSE}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(RETRY_PAUSE)
        else:
            raise RuntimeError(f"Ensembl GET {path} -> HTTP {resp.status_code}: {resp.text[:200]}")

    if last_exc is not None:
        raise RuntimeError(f"Ensembl GET {path} failed after {MAX_RETRIES} attempts: {last_exc}") from last_exc
    raise RuntimeError(f"Ensembl GET {path} failed after {MAX_RETRIES} attempts")


def ensembl_post(path, ids, params=None):
    """POST {ids} to path with retry. Returns the parsed JSON body, or {} on failure."""
    url = ENSEMBL_REST + path
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        _rate_limited_sleep()
        try:
            resp = requests.post(url, headers=HEADERS, params=params,
                                  json={"ids": ids}, timeout=60)
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            log.warning(f"  network error ({exc.__class__.__name__}) on POST {path} — "
                        f"waiting {RETRY_PAUSE}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(RETRY_PAUSE)
            continue

        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in RETRYABLE_STATUSES:
            log.warning(f"  HTTP {resp.status_code} on POST {path} — "
                        f"waiting {RETRY_PAUSE}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(RETRY_PAUSE)
        else:
            log.warning(f"  POST {path} -> HTTP {resp.status_code}: {resp.text[:200]}")
            return {}

    log.warning(f"  POST {path} failed after {MAX_RETRIES} attempts"
                f"{f': {last_exc}' if last_exc else ''} — batch of {len(ids)} unresolved")
    return {}


def fetch_cdna_batch(enst_list: list[str]) -> dict[str, str | None]:
    """
    Batch-fetch cDNA sequences for a list of ENST IDs via POST /sequence/id,
    chunked to POST_BATCH_SIZE. Missing/failed IDs map to None.
    """
    results: dict[str, str | None] = {}
    ids = sorted({e.split(".")[0] for e in enst_list})
    for i in range(0, len(ids), POST_BATCH_SIZE):
        chunk = ids[i:i + POST_BATCH_SIZE]
        data = ensembl_post("/sequence/id", chunk, params={"type": "cdna"})
        by_id = {}
        if isinstance(data, list):
            for item in data:
                eid = (item or {}).get("id")
                seq = (item or {}).get("seq")
                if eid:
                    by_id[eid] = seq.strip().upper() if seq else None
        for enst in chunk:
            results[enst] = by_id.get(enst)
    return results


def dna_to_rna(seq: str) -> str:
    return seq.replace("T", "U")


# ── RNA-FM INFERENCE ───────────────────────────────────────────────────────────
def embed_sequence(model, alphabet, seq_rna: str, label: str) -> np.ndarray:
    """Mean-pool RNA-FM layer-12 representations over all sequence positions → (640,) float32."""
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([(label, seq_rna)])
    tokens = tokens.to(next(model.parameters()).device)
    with torch.no_grad():
        out = model(tokens, repr_layers=[REPR_LAYER], return_contacts=False)
    token_reps = out["representations"][REPR_LAYER][0, 1:-1]  # strip <cls>/<eos>
    return token_reps.mean(0).cpu().float().numpy().astype(np.float32)


# ── 1. LOAD INPUT ──────────────────────────────────────────────────────────────
log.info("=" * 65)
log.info("RNA-FM Feature Extraction — Final Run")
log.info("=" * 65)
log.info(f"ADT mapping : {ADT_MAPPING_CSV}")
log.info(f"Output      : {OUT_DIR}")

adt_df = pd.read_csv(ADT_MAPPING_CSV, dtype=str).fillna("")

# Deduplicate on RNA_gene — one embedding per gene is sufficient
keep_cols = [c for c in ("RNA_gene", "Ensembl_ID", "RNA_transcript", "RNA_transcript_source")
             if c in adt_df.columns]
unique_genes = (
    adt_df[keep_cols]
    .drop_duplicates(subset="RNA_gene")
    .reset_index(drop=True)
)
unique_genes = unique_genes[unique_genes["RNA_gene"].str.strip() != ""].reset_index(drop=True)
log.info(f"{len(unique_genes)} unique gene symbols loaded (from {len(adt_df)} ADT entries)")

if "RNA_transcript" not in unique_genes.columns:
    log.error(f"'RNA_transcript' column missing from {ADT_MAPPING_CSV} — "
              f"run resolve_canonical_transcripts.py first.")
    sys.exit(1)

if "RNA_transcript_source" in unique_genes.columns:
    transcript_source_col = unique_genes["RNA_transcript_source"].astype(str).str.strip()
else:
    transcript_source_col = pd.Series([""] * len(unique_genes))

input_df = pd.DataFrame({
    "gene_symbol":            unique_genes["RNA_gene"].str.strip(),
    "ensembl_gene_id":        unique_genes["Ensembl_ID"].str.strip(),
    "canonical_transcript_id": unique_genes["RNA_transcript"].str.strip(),
    "transcript_source":      transcript_source_col,
    "cache_hit":              False,
    "cached_seq_len":         -1,
})

# ── 1b. LOAD CACHE FROM A PREVIOUS RUN'S metadata.csv ────────────────────────
# Since the transcript is now pre-resolved centrally (not by this script),
# a cache hit only requires the cached transcript ID to still match what the
# shared mapping currently says — no Ensembl call needed to validate it.
prev_meta_path = os.path.join(OUT_DIR, "metadata.csv")
prev_cache = {}
if os.path.isfile(prev_meta_path):
    prev_meta_df = pd.read_csv(prev_meta_path, dtype=str).fillna("")
    # Older runs (before the "source" column was split out) wrote status
    # "cached" for reused entries instead of "ok". Normalize so those rows
    # remain valid cache hits instead of being silently re-resolved.
    prev_meta_df.loc[prev_meta_df["status"] == "cached", "status"] = "ok"
    prev_cache = {r["gene_symbol"]: r for _, r in prev_meta_df.iterrows()}
    log.info(f"Loaded cache: {len(prev_cache)} gene(s) from previous run ({prev_meta_path})")
else:
    log.info("No previous metadata.csv found — starting without a cache.")

# ── 2. DETERMINE CACHE HITS ───────────────────────────────────────────────────
log.info("\nChecking cache against pre-resolved transcripts (adt_rna_mapping_final3.csv) ...")
for idx, row in input_df.iterrows():
    gene_sym = row["gene_symbol"]
    enst     = row["canonical_transcript_id"]
    if not enst:
        log.warning(f"  {gene_sym:<14} NO TRANSCRIPT in mapping")
        continue

    cached = prev_cache.get(gene_sym)
    out_fp = os.path.join(OUT_DIR, f"{gene_sym}.npy")
    if (cached is not None
            and cached.get("status") == "ok"
            and cached.get("canonical_transcript_id") == enst
            and os.path.isfile(out_fp)):
        input_df.at[idx, "cache_hit"] = True
        try:
            input_df.at[idx, "cached_seq_len"] = int(cached.get("seq_len", -1))
        except (TypeError, ValueError):
            pass
        log.info(f"  {gene_sym:<14} {enst}  [cached]")
        continue

    if cached is not None and cached.get("canonical_transcript_id") not in ("", enst):
        log.info(f"  {gene_sym:<14} transcript changed since cache "
                 f"({cached.get('canonical_transcript_id')} -> {enst}) — will recompute")
    else:
        log.info(f"  {gene_sym:<14} {enst}  (needs sequence + embedding)")

# ── 3. LOAD RNA-FM MODEL ───────────────────────────────────────────────────────
log.info("\nLoading RNA-FM model ...")
try:
    import fm
except ImportError:
    log.error("rna-fm not installed. Run:  pip install rna-fm")
    sys.exit(1)

torch.set_num_threads(N_TORCH_THREADS)
torch.set_num_interop_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"  Device: {device}  |  PyTorch threads: {torch.get_num_threads()}")

model, alphabet = getattr(fm.pretrained, RNA_FM_MODEL)()

# Extend positional embedding table via periodic tiling to ~16,000 nt
_pe     = model.embed_positions.weight.data
_orig   = _pe.shape[0]
if PE_TARGET > _orig:
    _pattern = _pe[2:]          # skip the two special-token rows
    _n_extra = PE_TARGET - _orig
    _extra   = torch.stack([_pattern[i % len(_pattern)] for i in range(_n_extra)])
    model.embed_positions.weight = torch.nn.Parameter(
        torch.cat([_pe, _extra], dim=0)
    )
    log.info(f"  Positional embeddings extended: {_orig} -> {PE_TARGET} rows")

model = model.to(device)
model.eval()
log.info(f"  Model: {RNA_FM_MODEL}  (layer {REPR_LAYER}, dim 640)")

# ── 4. BATCH-FETCH cDNA SEQUENCES ─────────────────────────────────────────────
# Only rows that are (a) not a valid cache hit and (b) actually have a
# resolved transcript need a sequence fetched.
rows_needing_seq = [
    (i, row["gene_symbol"], row["ensembl_gene_id"],
     row["canonical_transcript_id"], row["transcript_source"])
    for i, row in input_df.iterrows()
    if not row["cache_hit"] and row["canonical_transcript_id"]
]

log.info(f"\nFetching {len(rows_needing_seq)} cDNA sequence(s) via POST /sequence/id "
          f"({POST_BATCH_SIZE} per request) ...")
seq_by_enst = fetch_cdna_batch([enst for _, _, _, enst, _ in rows_needing_seq])

# ── 5. EMBED ───────────────────────────────────────────────────────────────────
log.info("\nComputing embeddings ...")

metadata_rows = []
n_total = len(input_df)
n_done  = 0

for idx, row in input_df.iterrows():
    gene_sym   = row["gene_symbol"]
    ensg       = row["ensembl_gene_id"]
    enst       = row["canonical_transcript_id"]
    tx_source  = row["transcript_source"]
    cache_hit  = row["cache_hit"]
    out_fp     = os.path.join(OUT_DIR, f"{gene_sym}.npy")
    prefix     = f"  [{n_done+1:3d}/{n_total}]  {gene_sym:<14}"

    meta = {
        "gene_symbol":            gene_sym,
        "ensembl_gene_id":        ensg,
        "canonical_transcript_id": enst,
        "transcript_source":      tx_source,
        "seq_len":                -1,
        "status":                 "",
        "source":                 "cached" if cache_hit else "computed",
    }

    if cache_hit and os.path.isfile(out_fp):
        meta["status"]  = "ok"
        meta["seq_len"] = int(row["cached_seq_len"])
        log.info(f"{prefix}  CACHED (reused prior embedding + transcript)")
        n_done += 1
        metadata_rows.append(meta)
        continue

    if not enst:
        meta["status"] = "no_transcript"
        log.warning(f"{prefix}  NO_TRANSCRIPT")
        n_done += 1
        metadata_rows.append(meta)
        continue

    seq_dna = seq_by_enst.get(enst.split(".")[0])
    if not seq_dna:
        meta["status"] = "no_sequence"
        log.warning(f"{prefix}  NO_SEQUENCE")
        n_done += 1
        metadata_rows.append(meta)
        continue

    seq_rna = dna_to_rna(seq_dna)
    seq_len = len(seq_rna)
    meta["seq_len"] = seq_len
    log.info(f"{prefix}  seq={seq_len:,} nt  ({enst}, {tx_source})")

    try:
        embedding = embed_sequence(model, alphabet, seq_rna, gene_sym)
    except Exception as exc:
        log.warning(f"{prefix}  RNA-FM error: {exc}")
        meta["status"] = f"rnafm_error: {exc}"
        n_done += 1
        metadata_rows.append(meta)
        continue

    np.save(out_fp, embedding)
    meta["status"] = "ok"
    log.info(f"{prefix}  -> saved  shape={embedding.shape}  [{out_fp}]")
    n_done += 1
    metadata_rows.append(meta)

# ── 6. SAVE METADATA ──────────────────────────────────────────────────────────
meta_df = pd.DataFrame(metadata_rows)
meta_path = os.path.join(OUT_DIR, "metadata.csv")
meta_df.to_csv(meta_path, index=False)

ok_count      = (meta_df["status"] == "ok").sum()
cache_count   = (meta_df["source"] == "cached").sum()
computed_count = ok_count - cache_count
mane_count    = (meta_df["transcript_source"] == "mane_select").sum()
canon_count   = (meta_df["transcript_source"] == "canonical").sum()
fail_count    = len(meta_df) - ok_count

log.info(f"\n── Summary ─────────────────────────────────────────────")
log.info(f"  Computed (fresh) : {computed_count}")
log.info(f"    MANE Select    : {mane_count}")
log.info(f"    Ensembl canon  : {canon_count}")
log.info(f"  Reused (cached)  : {cache_count}")
log.info(f"  Failed           : {fail_count}")
log.info(f"  Metadata         -> {meta_path}")
log.info("Done.")
