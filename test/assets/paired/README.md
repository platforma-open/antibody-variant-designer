# Paired Dataset

This directory holds **no reads on purpose**. The paired fixture is a pair of BCR FASTQ files already
committed in a sibling block:

```
1_blocks/mixcr-clonotyping/test/assets/small_data_R1.fastq.gz   (862 B)
1_blocks/mixcr-clonotyping/test/assets/small_data_R2.fastq.gz   (902 B)
```

## Why Nothing Is Generated Here

The paired half needs no generated data. Real BCR reads for it already exist, they are already small
enough to commit, and they already carry a chain MiXCR can process with its bundled human reference
library — no custom library is needed, which is exactly what blocks the VHH half (see
`../README.md`). Generating a second dataset would add weight and provenance questions for no gain.

## Why Referenced And Not Copied

Copying the files here would fork them. A fixture with two copies drifts: `mixcr-clonotyping` could
regenerate or re-trim its reads and this block would keep silently testing the old bytes, with
nothing to reveal the divergence. Referencing keeps exactly one source of truth.

The cost is a real dependency on a sibling repo, and `fixtures.test.ts` makes that cost visible
rather than surprising.

## How The Path Resolves

`fixtures.test.ts` looks for a `1_blocks` directory above this repo — first as a sibling of this
repo's checkout, then at `~/git/mil/1_blocks` — and then for
`mixcr-clonotyping/test/assets/small_data_R{1,2}.fastq.gz` inside it. Three outcomes, deliberately
kept distinct:

| Situation                                        | What the test does                                                                       |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| A `1_blocks` root exists and the files are there | Asserts both are gzip-readable, non-empty, and a valid 4-line-per-record FASTQ           |
| A `1_blocks` root exists but the files are gone  | **Fails**, naming `mixcr-clonotyping/test/assets/` and saying the upstream fixture moved |
| No `1_blocks` root exists at all                 | Skips, with the reason reported                                                          |

The third row is why the check is not an unconditional assertion. This block's CI clones this repo
alone, so `mixcr-clonotyping` is genuinely not on disk there — that is a checkout that cannot see the
sibling, not a broken fixture. Treating the two the same would either fail every CI run or hide a
real upstream move. The distinction is the point: a red suite here means the upstream files moved.

## Do Not Edit The Upstream Files

`mixcr-clonotyping` is read-only from this block. Reference its assets; never edit them and never
copy them in.
