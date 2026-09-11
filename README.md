# Antibody Variant Designer

Ranked antibody variant hypotheses that remove chemical liabilities and,
optionally, humanize non-human framework positions.

The block reads a predicted V-domain structure and its per-residue confidence
from the [3D Structure
Prediction](https://github.com/platforma-open/3d-structure-prediction) block,
rescans the liability motifs and solvent exposure in-block, triages which
liabilities are worth fixing, reads [AntiFold](https://doi.org/10.1093/bioadv/vbae202)
per-position matrices once per antibody — alongside a Sapiens human-repertoire
prior when the run also humanizes — keeps only candidates that clear their
target without introducing a new liability, and emits ranked variant
hypotheses — as a fixed-column CSV for synthesis and as a PFrame for its own
UI.

Three run modes are available. `liabilities` designs against the liability
objective alone. `humanization` designs against the humanization objective
alone, proposing substitutions at non-human framework positions.
`liabilities + humanization` runs both. In the combined
mode the liability objective targets only liabilities that lie entirely
inside a CDR, so the two objectives never propose against the same residue.
Variants from the two objectives are then combined into one dataset-wide
rank that blends structural tolerance with a humanness score.

Sister blocks: [Antibody Sequence
Liabilities](https://github.com/platforma-open/antibody-sequence-liabilities)
(sequence-only motif scan) and [3D Structure-Based
Liabilities](https://github.com/platforma-open/3D-Structure-Based-Liabilities)
(structure-filtered liability calls and developability cost). This block is the
first one that proposes a change rather than reporting a state.

Every emitted variant is an **unvalidated hypothesis**. The block predicts no
variant structure and estimates no binding strength.

## Pipeline

The workflow is **three steps over four execs** — one per step, plus one that
prints the shared liability taxonomy. Each step is its own Tengo template under
`workflow/src/steps/`, and `main.tpl.tengo` holds nothing but the wiring, so the
pipeline is readable without opening python.

```mermaid
flowchart LR
  U["3D Structure Prediction<br/>pdb + per-residue confidence"] --> M
  T["software-liability-definitions<br/>definitions.json"] --> M
  M["main.tpl.tengo<br/>pdbs/, one file per clonotype"] --> S1
  S1["1. index-and-scan<br/>onCPU"] --> S2["2. read-tolerance<br/>AntiFold + Sapiens prior<br/>onCPU / onGPU"] --> S3["3. build-variants<br/>onCPU"]
  S3 --> O["variants PFrame<br/>liabilities PFrame<br/>synthesis CSV"]
  S1 --> O
```

The chain has no fan-out. Every step runs **once over the whole dataset** and
loops over the parent antibodies inside that one exec, so a failure keeps the
name of the antibody it happened to — a rejection row keyed by clonotype, never a
failed job.

| Step | Entrypoint | Does | Resources |
|---|---|---|---|
| 1. `index-and-scan` | `index_and_scan.py` | builds the residue index from the ATOM records, computes solvent exposure, detects every liability motif, then triages each hit into `exposed` / `buried` / `fixability-declined` with an independent low-confidence flag | `onCPU` 2 cpu / 4 GiB |
| 2. `read-tolerance` | `read_tolerance.py` | loads the AntiFold checkpoint once, then reads per-position log-probabilities and perplexity for every antibody that still has actionable work; alongside it, reads a Sapiens human-repertoire prior over that antibody's framework positions | `onCPU` or `onGPU` 4 cpu / 8 GiB / 6 GiB vram |
| 3. `build-variants` | `build_variants.py` | proposes substitutions at editable positions for each objective the run mode selects — liability removal always, framework humanization when the mode adds it — **re-scans each candidate**, discards a liability candidate that fails to clear its own target, discards a humanization candidate that introduces more new liabilities than the run allows, never counts an ignored liability toward that allowance, ranks what survives per parent, then re-ranks the whole run by a blended structural-tolerance / humanness score | `onCPU` 2 cpu / 4 GiB |

## Data Flow

**Intermediates are plain files staged workdir to workdir.** No PColumn spec
exists for any of them and nothing imports them — only what the UI reads becomes
a PFrame. Each intermediate is one file per clonotype in a directory, named
after the clonotype key, and a directory crosses an exec boundary as a
`saveFileSet` capture restaged with `addFiles`, never as a path.

**No file lists the parent clonotypes.** Each step recovers the clonotypes it
must loop over from a staged directory listing, because every staged file is
named after the clonotype it belongs to. A list written by the workflow would be
one value resource holding every key, and the platform caps that at 3 MiB.

| Artifact | Producer → consumer | Format | Read/write by |
|---|---|---|---|
| `definitions.json` | taxonomy exec → steps 1, 3 | the taxonomy package's document — `schemaVersion` / `liabilities` / `fixabilityWeights` | `taxonomy_store.py` |
| `per_residue_confidence.tsv` | `main.tpl.tengo` → step 1 | clonotype key + JSON records, zero or one per run | `--per-residue-confidence` |
| `clonotype_filter.tsv` | `main.tpl.tengo` → step 1 | the optional Lead Selection subset, zero or one per run | `--clonotype-filter` |
| `pdbs/<key>.pdb` | `main.tpl.tengo` → steps 1, 2 | the upstream PDB ResourceMap, one blob per clonotype; the listing is also the parent-clonotype list | `parent_clonotypes.py` |
| `residues/<stem>.json` | step 1 → steps 2, 3 | the residue index, written only when the index phase passed | `residue_store.py` |
| `triaged/<stem>.json` | step 1 → steps 2, 3 | the actionable liabilities, written only when at least one is actionable; step 2 reads it purely as a gate | `liability_store.py` |
| `tolerance/<stem>.tsv` | step 2 → step 3 | `chain ⇥ imgt ⇥ perplexity ⇥ 20 log-probability columns` | `tolerance_store.py` |
| `tolerance/<stem>.prior.tsv` | step 2 → step 3 | `chain ⇥ imgt ⇥ 20 Sapiens log-probability columns`, framework positions only; read only when the run mode includes humanization | `sapiens_prior.py` |
| `rejected.tsv` | every step → `outputs` | `clonotypeKey ⇥ reason ⇥ detail ⇥ rejectedType`, one per step so three per run | `rejection_store.py` (write), the model (reduce) |

## Layout

| Path | What it holds |
|---|---|
| `workflow/` | Tengo templates — orchestration, exec, p-frame assembly |
| `model/` | `BlockModelV3` — block data, args projection, outputs for the UI |
| `ui/` | Vue 3 + `@platforma-sdk/ui-vue` — the variant table and the settings panel |
| `software/` | the python package doing the per-residue work |
| `block/` | the published block facade (`block.components`, `block.meta`) |
| `test/` | integration tests |
| `docs/` | the published description, and [`test-datasets.md`](docs/test-datasets.md) — which public dataset to run against per antibody format, and the blocks a project needs |

## Development

```bash
pnpm install
pnpm run build:dev-local     # requires a running docker daemon
pnpm run check
pnpm run test                # requires a backend on PL_ADDRESS
pnpm run upgrade-sdk         # refresh the canonical structure + SDK catalog
```
