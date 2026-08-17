# Test Datasets

**Real sequencing data to run this block against, one dataset per antibody format.** The formats
themselves — what each one must produce, and which ones are a named skip — live in `CLAUDE.md`. The
committed fixtures used by the integration suite live in `test/assets/README.md`. This file only says
which public dataset serves which format, and which blocks a project needs to carry it here.

Every dataset below comes from
[platforma-studies-library](https://github.com/milaboratory/platforma-studies-library). That repo
holds manifests, not data: the reads sit in one object store, and each study's `README.md` is the
authority on what the data is. Read it before you set a project up.

## The Choice

| Format | Dataset | Samples | Size | MiXCR preset |
|---|---|---|---|---|
| Fv | [`2026-02-in-vivo-antibody-discovery-sc-plus-bulk/sc-bcr-vdj`](https://github.com/milaboratory/platforma-studies-library/blob/main/2026-02-in-vivo-antibody-discovery-sc-plus-bulk/README.md) | 18 | 154 GB | `10x-sc-xcr-vdj` |
| one V domain, no partner chain | [`2026-07-human-phage-display-rsv/bulk-scfv`](https://github.com/milaboratory/platforma-studies-library/blob/main/2026-07-human-phage-display-rsv/README.md) | 5 | 5.5 GB | `generic-amplicon` |
| scFv | [`2026-07-macaque-scfv-siv-gp140/scfv-pacbio`](https://github.com/milaboratory/platforma-studies-library/blob/main/2026-07-macaque-scfv-siv-gp140/README.md) | 6 | 20 MB | none — see below |

### Fv — The Only Format That Runs End To End

**A human single-cell BCR repertoire, paired heavy and light per cell.** The `10x-sc-xcr-vdj` preset
covers it and human is the species MiXCR's bundled reference library covers best, so clonotyping
needs no custom library. Paired chains are what the Structure Prediction block's `ABodyBuilder2`
branch takes, which makes this the dataset that exercises the shipped chain rather than a guard.

**Add two or three samples, not the set.** The 18 samples total 154 GB, about 8.6 GB each. The study
also carries a `reference-antibodies.csv` answer key, so recovered clonotypes can be checked against
real leads.

**A mouse alternative of the same shape** is
[`2026-02-sc_mouse_Bcells-PRJNA1190485/bcr-vdj`](https://github.com/milaboratory/platforma-studies-library/blob/main/2026-02-sc_mouse_Bcells-PRJNA1190485/README.md)
— 15 samples, 103 GB, the same preset.

### One V Domain — A Lone VH Reaching The Block

**A human phage-display library whose reads each carry one complete variable domain, heavy or
light.** The NGS amplifies the two chains separately and pools them, so after the by-germline chain
split the heavy clonotypes have no partner chain. This is the smallest useful dataset in the set — 5
samples, 5.5 GB — and its `generic-amplicon` preset is real.

**Deep mutational scans are the wrong tool for this format.** The other single-chain studies
(`2026-05-titeseq-PRJNA1284397`, `2026-07-emibetuzumab-affinity-specificity`) carry
`mixcr_preset: null` because they are variant libraries against one parental clone, not repertoires,
so clonotyping does not apply to them.

**Unverified:** which engine Structure Prediction picks for a heavy-only, non-camelid clonotype. Run
that block alone on this dataset and look before treating the result as the format's test case.

### scFv — No Project Can Drive It

**Two independent walls stop scFv reads from reaching this block.** The Structure Prediction block
rejects scFv input, and this dataset's own `meta.yaml` names `mixcr-scfv-clonotyping` against
*Macaca mulatta* germline as its preset — not a block in the shipped chain. So the scFv case stays
what `CLAUDE.md` says it is: a guard for a hand-supplied file.

**Test it with a hand-built PDB, one chain carrying two V domains both numbered 1–128.** That is the
collision step 1 detects, and `structure-multi-domain-chain` is the skip reason it must write. The
study's `reference-scfv-panel.fasta` (130 records) is the sequence source for building that file.

### VHH — Not Reachable From This Library

**Three VHH studies exist and none can be processed.** The reads are camelid and MiXCR's bundled
reference library does not cover alpaca IGH, so there is nothing to turn those reads into
clonotypes. `test/assets/README.md` records the same blocker for the VHH fixture, and the fix is the
same in both places: build the alpaca library first.

## Blocks A Project Needs

**Four blocks make the required chain, in order.**

1. **Samples & Data** — the dataset's fastqs plus its `metadata.csv`. The library mirrors
   `metadata.csv` into the object store beside the reads, so sample-to-condition arrives with the data.
2. **MiXCR Clonotyping** — the preset from the study's `meta.yaml`, per the table above.
3. **[3D Structure Prediction](https://github.com/platforma-open/3d-structure-prediction)** — the one
   hard dependency. It writes the `pl7.app/structure/pdb` anchor and the per-residue confidence, and
   picks its engine per dataset.
4. **Antibody Variant Designer** — this block. Its only required argument is the structure dataset.

**Two more are optional, for different reasons.** **Lead Selection** writes the
`pl7.app/lead-selection` subset column that scopes the run to chosen clonotypes; without it a run
covers every clonotype in the structure dataset, which is a lot of AntiFold work on a whole
repertoire. **3D Structure-Based Liabilities** is a peer consumer, not upstream — it reads the same
anchor and writes none of its own, so it never has to run. Add it only to compare its scan against
this block's own.

**`assets-antifold-weights` is not a project block.** It is an assets package that ships as a
software dependency of this block's steps.
