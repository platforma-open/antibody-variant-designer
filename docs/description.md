# Overview

Proposes repaired antibody variants. The block reads a predicted V-domain
structure and its per-residue confidence from the [3D Structure
Prediction](https://github.com/platforma-open/3d-structure-prediction) block,
finds the chemical liability motifs, decides which of them are worth fixing
from how solvent-exposed they are, proposes substitutions that the fold
tolerates, and discards any proposal that fails to remove the target liability.
A humanization proposal is discarded when it introduces more new liabilities
than the run allows, which is none by default.

Three run modes are available. `liabilities` designs against the liability
objective alone, the original behavior. `humanization` designs against the
humanization objective alone, proposing substitutions at non-human framework
positions, scored against a Sapiens human-repertoire prior read alongside
AntiFold's own fold-tolerance matrices. `liabilities + humanization` runs both.
In the combined mode the liability
objective targets only liabilities that lie entirely inside a CDR, so the two
objectives never propose against the same residue. Their variants are then
combined into one dataset-wide rank that blends structural tolerance with a
humanness score.

Each surviving variant carries its rank within its parent antibody, the
positions it changed, how tolerant the fold is at those positions, a
humanness score when the humanization objective produced it, a binding-risk
band, and a permanent marker that it is an unvalidated hypothesis. The
humanness score is a raw OASis identity at a fixed tier — the fraction of the
variant's 9-mer peptides also found in human antibody repertoires — not a
calibrated OASis percentile, so it is not directly comparable to a calibrated
percentile published elsewhere. The result leaves the block as a
fixed-column CSV for synthesis, alongside a per-parent summary of which
liabilities and which non-human positions were found and what was decided
about each.

Substitution tolerance comes from [AntiFold](https://doi.org/10.1093/bioadv/vbae202)
per-position matrices, read once per antibody. No variant structure is
predicted and no binding strength is estimated — binding is judged empirically
in the wet lab.

The Sapiens prior reads inside the same step as AntiFold rather than a step of
its own. Both need torch, so a second step would still need its own GPU
allocation on the same queue — a permanent cost paid on every run — to save at
most a small fraction of that step's wall time, since the Sapiens checkpoint
(≈4.4 MiB) is tiny next to AntiFold's (540.6 MiB). Splitting them buys no real
concurrency the block doesn't already have to pay a second GPU job for.

Supported formats: VHH, mAb, Fab, scFv. V-domain only.
