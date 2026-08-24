# Antibody Variant Designer — target use cases

**This file names the antibody formats the python pipeline must accept, and what each one must
produce.** Every case below is a test in `software/tests/`. A format that is not in this list is a
named skip, never a best-effort guess.

## Vocabulary

| Term | Meaning |
|---|---|
| V domain | one variable domain, IMGT positions 1–128 |
| C domain | one constant domain — CH1, CH2, CH3, CL |
| role | `H` or `L`, taken from the `REMARK 99 PLATFORMA CDR*` records the Structure Prediction block writes |
| role-bearing chain | a PDB chain that a `REMARK 99` record names |
| in scope | a residue the block scans for liabilities and may propose an edit at |
| researched | scanned by `liability_motifs.py` and `liability_cysteines.py`, and offered to the design steps |

## The scope rule

**A residue is in scope when its chain carries a role AND the residue carries an IMGT region.** One
rule decides every case below. `residue_store.Residue.in_scope` is the single place it lives;
`liability_motifs.py` and `liability_cysteines.py` read it and never re-derive it.

**Out-of-scope residues stay in the residue index.** They are dropped from research, not from the
file and not from the index, for two reasons. Solvent exposure needs them — a V-domain residue
buried against CH1 or against the partner arm must read as buried, and `residue_exposure.py`
computes rSASA over the whole PDB. AntiFold needs them — step 3 parses the same PDB itself, so an
index that silently disagreed with AntiFold's residue set would shift the tolerance matrix against
the index.

## Case 1 — nanobody (VHH)

**One chain, role `H`, V domain only.** The block researches the whole chain.

The VHH hallmark extra disulfide (a CDR1–CDR3 cysteine pair) does not raise `extra_cysteines`.
`liability_cysteines.py` counts only FR1 and FR3, the two IMGT-conserved positions, so a cysteine in
a CDR is neither expected nor counted. This is correct for a canonical VHH. The cost is that a
genuinely unpaired free cysteine in a CDR is not reported either.

## Case 2 — Fv

**Two chains, roles `H` and `L`, V domains only.** Both chains are researched in full. Each chain is
scanned on its own sequence, so no motif can span the two chains.

## Case 3 — scFv

**One chain that holds VH, a linker, and VL — a named skip, not a partial result.** Two V domains on
one chain both number 1–128, so `(chain, imgt)` stops identifying one residue, and every downstream
join uses that pair. Step 1 detects the repeated `(chain, resSeq, insertionCode)` and writes the skip
reason `structure-multi-domain-chain`.

The upstream Structure Prediction block already blocks scFv input
(`text/work/projects/3d-structures-and-clustering/block-structure-prediction.md:163`), so this case
is a guard against a hand-supplied or future file, not a path the shipped chain produces.

## Case 4 — Fab

**Two chains: `H` = VH + CH1, `L` = VL + CL. The V domains are researched, the C domains are not.**
A C domain residue carries no IMGT region, so the scope rule excludes it. That stops a design
proposal landing in a constant region, which is not a per-clonotype design target and which AntiFold
was not asked to score.

**A C domain that restarts numbering at 1 is the same collision as an scFv.** Step 1 raises
`structure-multi-domain-chain` for it. A C domain numbered onward from 129 parses normally and is
excluded by the scope rule.

## Case 5 — mAb

**Four chains — two heavy, two light. Only the two chains a `REMARK 99` record names are
researched.** The second Fab arm is a sequence-identical copy of the first, so researching it would
report every liability twice and rank two identical variants against each other. The scope rule
excludes it, because no role points at it.

**The same rule excludes an antigen chain.** An antigen carries no role and no IMGT region, so it is
never scanned and never edited, while still contributing to burial in `residue_exposure.py`.

## Case 6 — half mAb, or one arm with the rest constant

**One role-bearing V domain plus a large constant mass — the V domain is researched, everything else
is excluded.** This is Case 4 and Case 5 applied together, with no new rule: constant residues have
no region, and the chains that carry no `REMARK 99` role have no role. The block reports liabilities
for the one variable domain it can design against, and the constant mass only shapes burial.

**A run that ends with zero in-scope residues is a skip, not an empty success.** Step 1 writes
`no-researchable-residue` when the scope rule retains nothing, so a file of pure constant region
never reaches the scan step as a silent empty index.

## What Upstream Actually Delivers Today

**Only Case 1 and Case 2 arrive through the shipped chain.** The Structure Prediction block predicts
variable domains only — ImmuneBuilder runs on variable-domain sequences
(`block-structure-prediction.md:32`), VHH mode drops the light chain (`:40`), and scFv is blocked
(`:163`). A Fab or mAb dataset therefore arrives as its Fv part. Cases 3 to 6 are guards for
hand-supplied files and for a future upstream that predicts more, and they exist so that such a file
skips with a name rather than losing residues without a word.

## Step 1 Skip Reasons

| Reason | Raised when |
|---|---|
| `no-structure` | the PDB holds no parseable ATOM record |
| `structure-not-imgt` | no `REMARK 99 PLATFORMA CDR` record and no residue at IMGT 10 |
| `structure-multi-domain-chain` | one chain repeats a `(resSeq, insertionCode)` pair, or two roles name one chain |
| `no-researchable-residue` | the scope rule retains no residue |
