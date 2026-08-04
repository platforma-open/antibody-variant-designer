# Overview

Proposes repaired antibody variants. The block reads a predicted V-domain
structure and its per-residue confidence from the [3D Structure
Prediction](https://github.com/platforma-open/3d-structure-prediction) block,
finds the chemical liability motifs, decides which of them are worth fixing
from how solvent-exposed they are, proposes substitutions that the fold
tolerates, and discards any proposal that fails to remove the target liability
or that introduces a new one.

Each surviving variant carries its rank within its parent antibody, the
positions it changed, how tolerant the fold is at those positions, a
binding-risk band, and a permanent marker that it is an unvalidated hypothesis.
The result leaves the block as a fixed-column CSV for synthesis.

Substitution tolerance comes from [AntiFold](https://doi.org/10.1093/bioadv/vbae202)
per-position matrices, read once per antibody. No variant structure is
predicted and no binding strength is estimated — binding is judged empirically
in the wet lab.

Supported formats: VHH, mAb, Fab, scFv. V-domain only.
