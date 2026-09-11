"""What the entrypoints share: the design engine.

The engine owns residue indexing, exposure, triage, the confidence gate, candidate
generation, binding-risk banding and ranking, plus the store for every file that crosses a
step boundary. An objective plugs into it through `design_objective` rather than
reimplementing any of that.

Nothing here has a `main`. The three entrypoints sit one level up, at the artifact root the
`cmd` path names — `index_and_scan.py` and `build_variants.py` here, `read_tolerance.py` in
the AntiFold package, which reaches this engine through the copy `stage-build.sh` stages.

Modules are named concept first, role last, so a listing groups them: `liability_*` detect
and triage, `residue_*` index and annotate, `variant_*` propose and rank, `*_store` read and
write one boundary file each.
"""
