# Antibody Variant Designer

Ranked antibody variant hypotheses that remove chemical liabilities.

The block reads a predicted V-domain structure and its per-residue confidence
from the [3D Structure
Prediction](https://github.com/platforma-open/3d-structure-prediction) block,
rescans the liability motifs and solvent exposure in-block, triages which
liabilities are worth fixing, reads [AntiFold](https://doi.org/10.1093/bioadv/vbae202)
per-position matrices once per antibody, keeps only candidates that clear the
target liability without introducing a new one, and emits ranked variant
hypotheses — as a fixed-column CSV for synthesis and as a PFrame for its own UI.

Sister blocks: [Antibody Sequence
Liabilities](https://github.com/platforma-open/antibody-sequence-liabilities)
(sequence-only motif scan) and [3D Structure-Based
Liabilities](https://github.com/platforma-open/3D-Structure-Based-Liabilities)
(structure-filtered liability calls and developability cost). This block is the
first one that proposes a change rather than reporting a state.

Every emitted variant is an **unvalidated hypothesis**. The block predicts no
variant structure and estimates no binding strength.

## Layout

| Path | What it holds |
|---|---|
| `workflow/` | Tengo templates — orchestration, exec, p-frame assembly |
| `model/` | `BlockModelV3` — block data, args projection, outputs for the UI |
| `ui/` | Vue 3 + `@platforma-sdk/ui-vue` — the variant table and the settings panel |
| `software/` | the python package doing the per-residue work |
| `block/` | the published block facade (`block.components`, `block.meta`) |
| `test/` | integration tests |

## Development

```bash
pnpm install
pnpm run build:dev-local     # requires a running docker daemon
pnpm run check
pnpm run test                # requires a backend on PL_ADDRESS
pnpm run upgrade-sdk         # refresh the canonical structure + SDK catalog
```
