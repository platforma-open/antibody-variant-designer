# Test Assets

Fixtures for the end-to-end chain: Samples & Data → MiXCR clonotyping → 3D structure prediction →
this block. Two datasets are needed, not one, because the upstream structure block picks its engine
**per dataset**: a VHH dataset exercises the `NanoBodyBuilder2` branch and a paired dataset exercises
the `ABodyBuilder2` branch. One dataset cannot cover both.

`fixtures.test.ts` asserts everything stated below. If you change a fixture, that test is what tells
you whether the change is still honest.

## Size Ceiling

The total committed size of this directory stays **under 5 MB**. That ceiling is the reason no
`git lfs` and no `.gitattributes` are used anywhere in this repo: the fixture is small enough to
commit directly. A fixture that needs lfs is too big for this purpose — shrink it instead, via the
simulator's `read_depth` / `num_clusters`. The neighbouring precedent is 4 KB–416 KB per file in
`mixcr-clonotyping/test/assets/`.

One floor overrides the ceiling: `ground_truth.csv` must keep **at least 20 clone rows**. A fixture
too small to be a fixture is worse than one slightly over budget. If shrinking cannot get under 5 MB
without dropping below that floor, record the fact in `provenance.json` and let a reviewer judge.

## Paired Dataset — Referenced Across A Repo Boundary

`paired/` holds no reads. It points at BCR reads already committed in another block:

```
1_blocks/mixcr-clonotyping/test/assets/small_data_R1.fastq.gz   (862 B)
1_blocks/mixcr-clonotyping/test/assets/small_data_R2.fastq.gz   (902 B)
```

They are **referenced, not copied** — see `paired/README.md` for why, and for how the test resolves
the path. This is a real dependency on a sibling repo: if those files move or are renamed,
`fixtures.test.ts` fails with a message naming `mixcr-clonotyping/test/assets/`, so a cross-repo move
is not misread as a local breakage.

## VHH Dataset

VHH fixture status: absent

Reason: the VHH reads are camelid, and MiXCR's bundled reference library does not cover alpaca IGH.
Building an alpaca library is the prerequisite for the VHH half, and it cannot be done from this
worktree: `1_blocks/mixcr-library-builder/workflow/src/main.tpl.tengo` builds a library by running
`mixcr buildLibrary` with a `MI_LICENSE` secret inside the Platforma runtime, and this machine has no
MiXCR license and no `mixcr` binary. Building that library is a separate project. Until it exists,
there is nothing that can turn alpaca reads into clonotypes, so committing VHH reads would add
megabytes that no test could process.

The blocker is the reference library alone. Everything else in the VHH half is verified to work:

| Checked                                                        | Result                                                                |
| -------------------------------------------------------------- | --------------------------------------------------------------------- |
| `milaboratory/ab-in-vitro-gen` is reachable and runs           | yes, at commit `763f4a5b2e1bde72b991a3847051bddb05208050`             |
| `python simulate.py --config configurations/minimal_test.yaml` | completes in 13.6 s                                                   |
| Output layout                                                  | `fastq/` with Illumina naming, `metadata.csv`, `ground_truth.csv`     |
| `ground_truth.csv` rows                                        | 117 — well above the 20-row floor                                     |
| Samples produced                                               | 6 (`S1`–`S6`): library, 1 negative round, 4 positive rounds           |
| Size at the config's default `read_depth: 100000`              | 173 MB — 35× over the ceiling                                         |
| Size at `read_depth: 2500`                                     | 4.20 MiB across 12 FASTQ files, and `ground_truth.csv` still 117 rows |

So when the alpaca library lands, the VHH half is a short job: re-run the simulator with
`sequencing.read_depth: 2500` (measured, not estimated — it fits the ceiling while the ground-truth
row count is set by `num_clusters` and does not shrink with read depth), copy `fastq/`,
`metadata.csv` and `ground_truth.csv` into `vhh/`, and write `vhh/provenance.json`.

Note two places where the simulator differs from what the plan assumed, both already handled by
`fixtures.test.ts`: `minimal_test.yaml` runs **4** positive rounds, not 3; and `read_prefix: "MIN4"`
prepends a prefix, so a sample name reads `MIN4_Round1_Pos_Rep1`. The sample index `S{Index}` is
unique per sample, which is what the test checks.

### The Availability Signal

**The presence of `vhh/provenance.json` is the one signal that the VHH fixture shipped.** It is
written only when the fixture really exists, and is absent otherwise. TODO-7 keys its VHH skip off
exactly that file, and `fixtures.test.ts` puts every VHH assertion behind it. Nothing infers VHH
availability any other way — not by guessing, and not by checking for the directory.

The `VHH fixture status:` line above is held consistent with that file by a test: it must read
`absent` when `vhh/provenance.json` is missing and `present` when it exists. So this README cannot
drift out of step with reality without the suite going red.

## Generated Data Proves Wiring, Not Biology

The VHH reads, when they land, come from a simulator. A passing end-to-end test therefore shows that
the chain is wired correctly — anchor selection, the ResourceMap join, the asset mount, the PColumn
shapes. It shows **nothing** about biological plausibility. `ground_truth.csv` makes this a stronger
wiring test than real data would be, and a weaker biology test. Do not read a green suite as
biological validation.
