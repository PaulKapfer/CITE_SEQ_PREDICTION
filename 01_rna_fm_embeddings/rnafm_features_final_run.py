#!/usr/bin/env python3
"""
RNA-FM Feature Extraction — Final Run
======================================
Reads gene symbols from adt_rna_mapping.csv (column "RNA_gene"), resolves
their canonical MANE Select transcripts via the Ensembl REST API, and computes
640-dimensional embeddings using RNA-FM (rna_fm_t12).

Input:
  ADT_MAPPING_CSV – CSV with columns ADT_feature, Ensembl_ID, RNA_gene
                    (C:\\...\\Final_Run\\Input\\adt_rna_mapping.csv)

Pipeline
--------
1. Load adt_rna_mapping.csv → unique RNA_gene list with pre-supplied Ensembl IDs.
2. For each gene:
     a. Use Ensembl_ID from the CSV when present; otherwise look up via
        /lookup/symbol/homo_sapiens/{symbol}.
     b. Fetch all transcripts for the ENSG ID and select the MANE Select
        isoform (is_mane_select=1).  Falls back to the Ensembl canonical
        transcript when no MANE Select entry is available.
     c. Fetch cDNA sequence via /sequence/id/{ENST}?type=cdna.
3. Convert T → U (DNA → RNA alphabet).
4. Run RNA-FM (rna_fm_t12, 640-dim).
   Positional embedding table is extended to 16,000 nucleotides via periodic
   tiling to accommodate all relevant transcript lengths.
5. Extract representations from the final transformer layer (layer 12) and
   mean-pool across all nucleotide positions, excluding <cls> and <eos>,
   yielding a single 640-dimensional feature vector per gene.
6. Save one .npy per gene symbol to OUT_DIR.
7. Save metadata.csv with per-gene status, sequence length, transcript ID, etc.

Dependencies
------------
  pip install rna-fm torch pandas requests
"""

import os
import sys
import time
import queue
import logging
import threading
import requests
import numpy as np
import pandas as pd
import torch

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
ADT_MAPPING_CSV = r"C:\Users\Paul\Desktop\Publications\CITE-SEQ_pred\Create_RNA_FM_features\Input\adt_rna_mapping.csv"

OUT_DIR = r"C:\Users\Paul\Desktop\Publications\CITE-SEQ_pred\Create_RNA_FM_features\Output\rnafm_features"

# RNA-FM
RNA_FM_MODEL = "rna_fm_t12"   # 12-layer, 640-dim
REPR_LAYER   = 12
PE_TARGET    = 16_002          # extend positional table to ~16,000 nt

# Threading / Ensembl API
N_TORCH_THREADS = 5
FETCH_WORKERS   = 1
PREFETCH_QUEUE  = 8
ENSEMBL_REST    = "https://rest.ensembl.org"
API_RATE_SEC    = 0.1
MAX_RETRIES     = 5
RETRY_PAUSE     = 12.0

# ── SETUP ──────────────────────────────────────────────────────────────────────
# Create the output directory up front (no-op if it already exists).
os.makedirs(OUT_DIR, exist_ok=True)

# Configure logging once: INFO level, timestamped, written both to the console
# (StreamHandler → stdout) and to a log file inside the output directory.
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUT_DIR, "rnafm_features.log"), mode="w"),
    ],
)
log = logging.getLogger(__name__)

# Shared state for talking to the Ensembl REST API:
HEADERS        = {"Content-Type": "application/json", "Accept": "application/json"}  # default JSON request headers
_api_lock      = threading.Lock()   # serialises rate-limit bookkeeping across threads
_last_api_call = 0.0                 # wall-clock time of the most recent API call


def _rate_limited_sleep():
    """Throttle API calls so consecutive requests are ≥ API_RATE_SEC apart.

    Thread-safe: the lock guarantees only one thread reads/updates the
    last-call timestamp at a time, so concurrent fetch workers cannot exceed
    Ensembl's request-rate limit.
    """
    global _last_api_call
    with _api_lock:
        # How much longer we must wait before the next call is allowed.
        delta = API_RATE_SEC - (time.monotonic() - _last_api_call)
        if delta > 0:
            time.sleep(delta)
        # Record this call's time as the new reference point.
        _last_api_call = time.monotonic()


# ── ENSEMBL HELPERS ────────────────────────────────────────────────────────────
def ensembl_get(path, params=None, text=False):
    """GET an Ensembl REST endpoint with rate-limiting and retry.

    Returns parsed JSON (default) or raw text (text=True). Retries on transient
    429/503 responses; raises RuntimeError on other errors or after MAX_RETRIES.
    """
    url  = ENSEMBL_REST + path
    hdrs = {"Accept": "text/plain"} if text else HEADERS   # plain text for sequences, JSON otherwise
    for attempt in range(1, MAX_RETRIES + 1):
        _rate_limited_sleep()                              # respect the request-rate limit
        resp = requests.get(url, headers=hdrs, params=params, timeout=30)
        if resp.status_code == 200:                        # success → return body
            return resp.text if text else resp.json()
        if resp.status_code in (429, 503):                 # rate-limited / unavailable → back off and retry
            log.warning(f"  HTTP {resp.status_code} — waiting {RETRY_PAUSE}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(RETRY_PAUSE)
        else:                                              # any other status is a hard error
            raise RuntimeError(f"Ensembl GET {path} -> HTTP {resp.status_code}: {resp.text[:200]}")
    raise RuntimeError(f"Ensembl GET {path} failed after {MAX_RETRIES} attempts")


def resolve_ensg(gene_symbol: str) -> str:
    """Look up ENSG ID for a gene symbol via Ensembl REST. Returns '' on failure."""
    try:
        info = ensembl_get(f"/lookup/symbol/homo_sapiens/{gene_symbol}", params={"expand": 0})
        return info.get("id", "")
    except RuntimeError as e:
        log.warning(f"  symbol lookup failed for {gene_symbol}: {e}")
        return ""


def resolve_mane_transcript(ensg: str) -> tuple[str, str]:
    """
    Return (enst_id, source) for the best available transcript for ensg.

    Selection hierarchy:
      1. MANE Select  (is_mane_select == 1)
      2. Ensembl canonical_transcript field
      3. '' — no transcript found

    source is one of 'mane_select', 'canonical', or ''.
    """
    try:
        gene_detail = ensembl_get(f"/lookup/id/{ensg}", params={"expand": 1})
    except RuntimeError as e:
        log.warning(f"  gene lookup failed for {ensg}: {e}")
        return "", ""

    transcripts = gene_detail.get("Transcript", [])   # all transcripts of this gene

    # 1. Prefer the MANE Select transcript when one is flagged.
    for t in transcripts:
        if t.get("is_mane_select"):
            return t["id"].split(".")[0], "mane_select"   # strip version suffix (.N)

    # 2. Otherwise fall back to Ensembl's canonical transcript field.
    canonical = gene_detail.get("canonical_transcript", "")
    if canonical:
        return canonical.split(".")[0], "canonical"

    # 3. Nothing usable found.
    return "", ""


def fetch_cdna(enst_id: str) -> str | None:
    """Fetch the cDNA sequence for a transcript ID. Returns None on failure."""
    enst_base = enst_id.split(".")[0]   # drop version suffix before querying
    try:
        seq = ensembl_get(f"/sequence/id/{enst_base}", params={"type": "cdna"}, text=True)
        return seq.strip().upper() if seq else None
    except RuntimeError as e:
        log.warning(f"  cDNA fetch failed for {enst_base}: {e}")
        return None


def dna_to_rna(seq: str) -> str:
    """Convert a DNA sequence to RNA alphabet (T → U) for RNA-FM input."""
    return seq.replace("T", "U")


# ── RNA-FM INFERENCE ───────────────────────────────────────────────────────────
def embed_sequence(model, alphabet, seq_rna: str, label: str) -> np.ndarray:
    """Mean-pool RNA-FM layer-12 representations over all sequence positions → (640,) float32."""
    batch_converter = alphabet.get_batch_converter()       # tokeniser for RNA-FM
    _, _, tokens = batch_converter([(label, seq_rna)])     # encode the single sequence to token IDs
    tokens = tokens.to(next(model.parameters()).device)    # move tokens to the model's device (CPU/GPU)
    with torch.no_grad():                                  # inference only — no gradient tracking
        out = model(tokens, repr_layers=[REPR_LAYER], return_contacts=False)
    token_reps = out["representations"][REPR_LAYER][0, 1:-1]  # per-nucleotide vectors, strip <cls>/<eos>
    return token_reps.mean(0).cpu().float().numpy().astype(np.float32)  # mean over positions → (640,)


# ── 1. LOAD INPUT ──────────────────────────────────────────────────────────────
log.info("=" * 65)
log.info("RNA-FM Feature Extraction — Final Run")
log.info("=" * 65)
log.info(f"ADT mapping : {ADT_MAPPING_CSV}")
log.info(f"Output      : {OUT_DIR}")

adt_df = pd.read_csv(ADT_MAPPING_CSV, dtype=str).fillna("")

# Deduplicate on RNA_gene — one embedding per gene is sufficient
unique_genes = (
    adt_df[["RNA_gene", "Ensembl_ID"]]
    .drop_duplicates(subset="RNA_gene")
    .reset_index(drop=True)
)
unique_genes = unique_genes[unique_genes["RNA_gene"].str.strip() != ""].reset_index(drop=True)
log.info(f"{len(unique_genes)} unique gene symbols loaded (from {len(adt_df)} ADT entries)")

# Working table: one row per gene. The transcript columns are filled in below.
input_df = pd.DataFrame({
    "gene_symbol":            unique_genes["RNA_gene"].str.strip(),
    "ensembl_gene_id":        unique_genes["Ensembl_ID"].str.strip(),
    "canonical_transcript_id": "",   # resolved in step 2
    "transcript_source":      "",    # 'mane_select' / 'canonical' / '' (set in step 2)
})

# ── 2. RESOLVE ENSG IDS AND MANE SELECT TRANSCRIPTS ──────────────────────────
log.info("\nResolving MANE Select / canonical transcript IDs ...")
for idx, row in input_df.iterrows():
    gene_sym = row["gene_symbol"]
    ensg     = row["ensembl_gene_id"]

    # Fill missing ENSG via symbol lookup
    if not ensg:
        ensg = resolve_ensg(gene_sym)
        input_df.at[idx, "ensembl_gene_id"] = ensg
        log.info(f"  {gene_sym:<14} ENSG lookup -> {ensg or 'NOT FOUND'}")

    if not ensg:
        continue

    enst, source = resolve_mane_transcript(ensg)
    input_df.at[idx, "canonical_transcript_id"] = enst
    input_df.at[idx, "transcript_source"]        = source
    log.info(f"  {gene_sym:<14} {ensg}  ->  {enst or 'NO TRANSCRIPT'}  [{source}]")

# ── 3. LOAD RNA-FM MODEL ───────────────────────────────────────────────────────
log.info("\nLoading RNA-FM model ...")
try:
    import fm                       # the rna-fm package
except ImportError:
    log.error("rna-fm not installed. Run:  pip install rna-fm")
    sys.exit(1)

# Limit CPU thread counts (avoids oversubscription) and pick GPU if available.
torch.set_num_threads(N_TORCH_THREADS)
torch.set_num_interop_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"  Device: {device}  |  PyTorch threads: {torch.get_num_threads()}")

# Load the pretrained model and its tokeniser (alphabet).
model, alphabet = getattr(fm.pretrained, RNA_FM_MODEL)()

# Extend the positional-embedding table to PE_TARGET rows so long transcripts
# (up to ~16,000 nt) can be encoded. Extra rows are filled by periodically
# tiling the existing learned positions (skipping the 2 special-token rows).
_pe     = model.embed_positions.weight.data
_orig   = _pe.shape[0]
if PE_TARGET > _orig:
    _pattern = _pe[2:]                                    # learned position rows to repeat
    _n_extra = PE_TARGET - _orig                          # how many rows to add
    _extra   = torch.stack([_pattern[i % len(_pattern)] for i in range(_n_extra)])  # tiled rows
    model.embed_positions.weight = torch.nn.Parameter(
        torch.cat([_pe, _extra], dim=0)                  # original rows + tiled extension
    )
    log.info(f"  Positional embeddings extended: {_orig} -> {PE_TARGET} rows")

model = model.to(device)   # move weights to the chosen device
model.eval()               # inference mode (disables dropout etc.)
log.info(f"  Model: {RNA_FM_MODEL}  (layer {REPR_LAYER}, dim 640)")

# ── 4. FETCH + EMBED (prefetch pipeline) ──────────────────────────────────────
# Producer/consumer design: background worker thread(s) download sequences and
# push them onto a bounded queue, while the main thread (which owns the model)
# pulls sequences off the queue and runs RNA-FM. This overlaps network I/O with
# GPU/CPU inference. PREFETCH_QUEUE bounds how many sequences are buffered.
log.info("\nFetching sequences and computing embeddings ...")

_fetch_queue: queue.Queue = queue.Queue(maxsize=PREFETCH_QUEUE)


def _fetch_worker(rows):
    """Download cDNA for each gene and enqueue (…, seq_rna, status) tuples.

    Skips genes already saved to disk, and tags genes with no transcript / no
    sequence so the consumer can record them without attempting inference.
    """
    for idx, gene_sym, ensg, enst, tx_source in rows:
        out_fp = os.path.join(OUT_DIR, f"{gene_sym}.npy")
        if os.path.isfile(out_fp):                         # already computed → skip (resume support)
            _fetch_queue.put((idx, gene_sym, ensg, enst, tx_source, None, "skipped"))
            continue
        if not enst:                                       # no transcript resolved in step 2
            _fetch_queue.put((idx, gene_sym, ensg, enst, tx_source, None, "no_transcript"))
            continue
        seq_dna = fetch_cdna(enst)                         # download the cDNA sequence
        if seq_dna is None:                                # download failed
            _fetch_queue.put((idx, gene_sym, ensg, enst, tx_source, None, "no_sequence"))
        else:                                              # success → convert to RNA and enqueue
            _fetch_queue.put((idx, gene_sym, ensg, enst, tx_source, dna_to_rna(seq_dna), "fetched"))


# Flatten the per-gene table into plain tuples for the worker(s).
all_rows = [
    (i, row["gene_symbol"], row["ensembl_gene_id"],
     row["canonical_transcript_id"], row["transcript_source"])
    for i, row in input_df.iterrows()
]
# Round-robin split the genes across FETCH_WORKERS background threads.
worker_batches = [all_rows[i::FETCH_WORKERS] for i in range(FETCH_WORKERS)]

# Launch one daemon fetch thread per batch.
threads = []
for batch in worker_batches:
    t = threading.Thread(target=_fetch_worker, args=(batch,), daemon=True)
    t.start()
    threads.append(t)


def _sentinel():
    """Wait for all fetch workers to finish, then push a None to stop the consumer."""
    for t in threads:
        t.join()
    _fetch_queue.put(None)   # sentinel = "no more items"


# Run the sentinel in its own thread so the main thread can start consuming now.
threading.Thread(target=_sentinel, daemon=True).start()

# ── Main thread: drain queue, run RNA-FM ──────────────────────────────────────
metadata_rows = []          # one record per gene for the final metadata.csv
n_total = len(input_df)
n_done  = 0

while True:
    item = _fetch_queue.get()   # block until the next fetched item is available
    if item is None:            # sentinel → all workers finished
        break

    idx, gene_sym, ensg, enst, tx_source, seq_rna, status = item
    out_fp = os.path.join(OUT_DIR, f"{gene_sym}.npy")
    prefix = f"  [{n_done+1:3d}/{n_total}]  {gene_sym:<14}"   # progress label for logs

    # Provisional metadata record (status/seq_len may be updated below).
    meta = {
        "gene_symbol":            gene_sym,
        "ensembl_gene_id":        ensg,
        "canonical_transcript_id": enst,
        "transcript_source":      tx_source,
        "seq_len":                -1,
        "status":                 status,
    }

    # Already-saved genes: record and move on.
    if status == "skipped":
        log.info(f"{prefix}  SKIP (already saved)")
        n_done += 1
        metadata_rows.append(meta)
        continue

    # Genes with no transcript / no sequence: record the failure and move on.
    if status in ("no_transcript", "no_sequence"):
        log.warning(f"{prefix}  {status.upper()}")
        n_done += 1
        metadata_rows.append(meta)
        continue

    # We have a sequence → record its length and run the model.
    seq_len = len(seq_rna)
    meta["seq_len"] = seq_len
    log.info(f"{prefix}  seq={seq_len:,} nt  ({enst}, {tx_source})")

    try:
        embedding = embed_sequence(model, alphabet, seq_rna, gene_sym)   # (640,) vector
    except Exception as exc:                                # e.g. sequence too long / OOM
        log.warning(f"{prefix}  RNA-FM error: {exc}")
        meta["status"] = f"rnafm_error: {exc}"
        n_done += 1
        metadata_rows.append(meta)
        continue

    # Persist the embedding as <gene>.npy and mark success.
    np.save(out_fp, embedding)
    meta["status"] = "ok"
    log.info(f"{prefix}  -> saved  shape={embedding.shape}  [{out_fp}]")
    n_done += 1
    metadata_rows.append(meta)

# ── 5. SAVE METADATA ──────────────────────────────────────────────────────────
# Write one CSV row per gene (status, transcript, sequence length, …).
meta_df = pd.DataFrame(metadata_rows)
meta_path = os.path.join(OUT_DIR, "metadata.csv")
meta_df.to_csv(meta_path, index=False)

# Tally outcomes for the run summary.
ok_count      = (meta_df["status"] == "ok").sum()
skip_count    = (meta_df["status"] == "skipped").sum()
mane_count    = (meta_df["transcript_source"] == "mane_select").sum()
canon_count   = (meta_df["transcript_source"] == "canonical").sum()
fail_count    = len(meta_df) - ok_count - skip_count

log.info(f"\n── Summary ─────────────────────────────────────────────")
log.info(f"  Completed        : {ok_count}")
log.info(f"    MANE Select    : {mane_count}")
log.info(f"    Ensembl canon  : {canon_count}")
log.info(f"  Skipped (cached) : {skip_count}")
log.info(f"  Failed           : {fail_count}")
log.info(f"  Metadata         -> {meta_path}")
log.info("Done.")
