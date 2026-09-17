---
"@platforma-open/milaboratories.antibody-variant-designer.software-antifold": minor
"@platforma-open/milaboratories.antibody-variant-designer.software-developability": minor
---

Run both software packages on the `3.12.10-antibody-design` run environment, which ships the whole closure
they install today — AntiFold's torch stack, Sapiens, freesasa and promb, about 2 GiB. The offline install
now finds every pin already present instead of resolving against an environment that never carried them.

biotite moves from 0.38 to 0.39. 0.38 publishes no wheel usable under Python 3.12, so the run environment
cannot carry it for any platform; 0.39.0 is the first line with cp312 wheels.
