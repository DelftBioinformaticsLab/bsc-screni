# SEA-AD Integration History

Design decisions and failure modes for the SEA-AD MTG Phase 1 (integration) work — written so the next maintainer doesn't re-derive the same lessons. Most of this is about the **unpaired branch**, which is the only part that was hard.

The paired branch was straightforward (standard WNN, same code path as PBMC) and is treated as "done" — students can rely on it. The unpaired branch is workable but has known limitations; for anyone considering a deeper fix later, this document lays out what was already tried.

## Dataset overview

SEA-AD provides two h5ad files, each pooling all donors:

- **RNA** (`SEAAD_MTG_RNAseq_*.h5ad`): 1,364,973 nuclei × 36,601 genes. Singleome snRNA-seq (~84 donors) + multiome RNA half (28 donors).
- **ATAC** (`SEAAD_MTG_ATACseq_*.h5ad`): 516,389 nuclei × 218,882 peaks. Singleome snATAC-seq + multiome ATAC half.

The `method` obs column (`10xMulti` = multiome, otherwise singleome) drives the paired/unpaired split. All 24 SEA-AD `Subclass` cell types are kept on disk through Phase 1 — subclass narrowing is left to each downstream sub-question.

## Paired branch (multiome) — went smoothly

138,118 cells × 28 donors with shared barcodes between RNA and ATAC. Standard WNN pipeline (`integrate_paired()` with `batch_key="Donor ID"` for donor-aware HVG):

1. RNA: normalize → log → HVG(2000) → scale → PCA(50)
2. ATAC: TF-IDF → LSI(50)
3. L2-normalize embeddings, per-modality kNN
4. WNN via `muon`, Leiden clustering, joint UMAP
5. Extract k=20 WNN neighbor indices (input for wScReNI)

Output: `seaad_paired_integrated.h5mu` (~91 GB). UMAP shows coherent subclass structure with donors mixed within each subclass — the expected pattern.

The one thing worth noting: ATAC `.X` ships as Tn5 insertion counts (integer-valued; mostly multiples of 2 because each fragment contributes two cuts). We verified this with [scripts/check_seaad_atac_counts.py](../scripts/check_seaad_atac_counts.py). The earlier worry that `.X` might be normalized was wrong — it's raw, just in unusual units. Downstream TF-IDF / VST / Spearman are all rank/scale-invariant so this doesn't matter, but it's worth knowing.

## Unpaired branch (singleome) — three iterations

RNA and ATAC are from different cells (no shared barcodes), so the integration has to discover correspondences. We went through three designs before settling on something we'd ship.

### Iteration 1 — per-donor everything

Initial design: for each donor with both modalities (85 donors after `MIN_CELLS_PER_DONOR=50`), compute gene activity from ATAC peaks, find HVGs on the donor's RNA, share genes, log-normalize both, concatenate, **per-donor PCA**, **per-donor Harmony** with `batch_key="modality"`, then RNA → nearest ATAC pairing with dedup by ATAC cell.

Result: **28,518 pairs total.** Pair-retention rate after dedup was 3–8% of `min(n_RNA, n_ATAC)` per donor — even on donors with near-balanced modality counts. RNA cells were crowding onto a few ATAC anchors, then dedup was throwing the rest away.

Two distinct problems were stacked here:

1. **Pairing asymmetry**: `RNA → ATAC + dedup-by-ATAC` caps output at `n_unique_ATAC_matched`. When alignment is at all uneven, that's much less than `n_ATAC`.
2. **Integration quality**: even after fixing the algorithm, the underlying Harmony output had RNA and ATAC in separated regions — per-donor PCA on ~10k cells with two modality batches doesn't give Harmony enough data to learn a meaningful correction.

### Iteration 2 — anchor on rare modality, no dedup

Fixed the algorithmic part. Pairing now anchors on the smaller modality (`anchor="atac"` when `n_ATAC < n_RNA`, else `anchor="rna"`), takes the nearest cross-modal neighbor for each anchor cell, no dedup. By construction `pair_count = min(n_RNA, n_ATAC)`.

Result: **371,905 pairs** (13× increase). `pair_fraction = 1.000` across all donors.

But: `cell_type_agreement` was **0.288** mean (vs ~0.10 random baseline on the marginal subclass distribution). Better than random, far from biological. Per-cell-type pair distributions also showed suspicious overrepresentation of rare neuronal subtypes (`VLMC` 12%, `Sst Chodl` 11%) — consistent with Harmony placing a few cells centrally so they became the default nearest neighbor for many ATAC cells of other types.

So the algorithmic fix worked, but it surfaced that the underlying integration was still poor.

### Iteration 3 — global embedding, per-donor pairing (current)

Switched from per-donor PCA + Harmony to:

1. Filter to usable donors (both modalities ≥ 50 cells)
2. One global gene-activity computation across all unpaired ATAC
3. Global HVGs from RNA on a 200k-cell donor-stratified subsample (Seurat v3 VST is variance-based, sample-size invariant for ranking)
4. Subset both modalities to shared HVGs ∩ gene-activity vars, log1p(normalize_total)
5. Concatenate (~1.6M cells) → one `combined` AnnData
6. **Global** sparse PCA (`zero_center=False`, truncated SVD)
7. Harmony with `batch_key="modality"` only — donor effects deliberately kept as biological variation
8. Per-donor cross-modal pairing on `X_harmony`, anchor on rare modality

Rationale: per-donor Harmony was starved of data. Global Harmony gets all 1.6M cells to learn the modality correction; donor structure is preserved because Harmony was never told about it. Pairing stays per-donor so each pair has a single, unambiguous `donor_id`.

Result: pair count still 371,905 (algorithm unchanged). `cell_type_agreement` improved modestly over 0.288, but UMAPs still show **modalities forming clearly disjoint clouds within each donor's region**. The improvement is real but not transformative — the underlying limitation is gene-activity as a noisy proxy for expression, which a linear correction can't bridge regardless of how much data Harmony sees.

## Current state of unpaired

Integration is "working" in the sense that:

- Every aligned donor produces `min(n_RNA, n_ATAC)` pairs
- Pairs share `donor_id` on both sides — per-donor / per-AD-status analyses are well-defined
- Output files persist cleanly and downstream pipeline can consume them

Integration is **not "good"** in the sense that:

- RNA and ATAC remain visibly separated in the joint embedding
- Cell-type agreement is well above random but not at the ≥ 0.7 mark that would indicate biologically reliable matching
- Pairs encode coarse cell-type concordance, not fine-grained correspondence

The gap is most likely intrinsic to Harmony-on-gene-activity. ATAC peak-sum-over-gene-body is a noisy proxy for transcription; a linear correction can't bridge the resulting representation gap. This is a known limitation of this class of approach in the multi-omics integration literature.

## What we tried that didn't help

- **More aggressive Harmony parameters** (`lamb`, `theta`): small effect on the modality separation, not enough to pursue alone.
- **Dedup-by-ATAC pairing**: turned out to confound algorithmic cell loss with integration-quality issues. Removed.

## What we have not tried (open work)

Natural escalations if the unpaired branch needs to be biologically usable, not just workable:

1. **Better gene-activity**: ArchR-/Signac-style weighting by distance to TSS (exponential decay) rather than flat peak-body overlap. Moderate effort, possibly modest gains.
2. **scGLUE** (Cao et al., 2022): purpose-built for unpaired snRNA + snATAC. Uses a regulator graph (peak ↔ gene linkage) as cross-modal anchor and learns a joint VAE embedding. Designed exactly for this failure mode. Adds PyTorch + scglue dependencies and ~1 day of integration work; GPU recommended for training.
3. **Donor filtering**: drop donors with extreme RNA:ATAC imbalance to give Harmony cleaner per-donor distributions. Easy to test, likely incremental.
4. **Glia-only**: filter to Microglia-PVM / Astrocyte / Oligodendrocyte / OPC before integration. These are biologically far from neurons and from each other; Harmony tends to handle them better even when fine neuronal subtypes fail. (Already supported at Phase 3 via `scripts/subsample_seaad_paired.py --cell-types`; doing it at integration time is the open work.)

For sub-questions that primarily care about cell-type-specific GRNs in glia (most of the proposed sub-questions), option 4 + the paired data alone is probably sufficient and avoids the unpaired-integration issue entirely. **Option 2 (scGLUE) is the right answer if a fully multi-donor unpaired analysis is needed.**

## Pointers to specific code

The integration code is in [src/screni/data/integration_seaad.py](../src/screni/data/integration_seaad.py):

- `integrate_seaad_paired()` — the WNN wrapper
- `integrate_seaad_unpaired_global()` — the Iteration 3 design
- `_pair_cross_modal()` — the anchor-on-rare pairing helper
- `_sanitize_obs_for_h5ad()` — strips mixed-dtype obs columns that h5py rejects (SEA-AD ships some, notably `experiment_component_failed`)

Loading lives in [src/screni/data/loading_seaad.py](../src/screni/data/loading_seaad.py). QC figures are produced by [scripts/inspect_seaad_integration.py](../scripts/inspect_seaad_integration.py).
