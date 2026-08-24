"""End-to-end tests for `read_tolerance.py` against the real vendored AntiFold
package. The only test file in this package, and the only one that needs the
heavy set: run it with `uv run --extra antifold --group e2e pytest -m e2e`.
Every other test of this module is a unit test and lives in the
developability package's suite, which never installs torch — so this file
skips itself whenever torch is absent.

`isolated_antifold_module` loads our script under a private name and clears
every `antifold`/`antifold.*` entry around each test, so a case starts from
a fresh import of the vendored package rather than from whatever an earlier
case left in `sys.modules`, and leaves none of its own behind. Only
`_run_model` imports that package for real, here and nowhere else.

No real checkpoint is needed: `_load_IF1_local()` alone builds a full,
untrained model, and every case here only checks that AntiFold's own
coordinate-loading and forward-pass plumbing accepts the shape `_run_model`
builds — never the predicted values.
"""

import importlib.util
import math
import statistics
import sys
from pathlib import Path

import pytest

import liability_store
import liability_triage
import pdb_index_store
import residue_store
import sapiens_prior
import skip_store

pytest.importorskip("torch")

pytestmark = pytest.mark.e2e

_SCRIPT_PATH = Path(__file__).parent.parent / "src" / "read_tolerance.py"

# The real, tiny (4.4 MiB) Sapiens checkpoint + tokenizer TODO-11.1 already fetched, in
# the sibling asset repo this multi-repo workspace checks out beside this one. A
# standalone clone of this repo alone (e.g. CI) never has it — `sapiens_weights_root`
# below skips rather than fails when it is absent.
_SAPIENS_WEIGHTS_ROOT = (
    Path(__file__).resolve().parents[4].parent
    / "assets-sapiens-weights" / "sapiens" / "indexed_model" / "sapiens"
)


@pytest.fixture
def isolated_antifold_module():
    """Load our own script from its path, and evict any `antifold`/`antifold.*`
    entry already in `sys.modules` — a partially imported vendored package
    left by an earlier case would otherwise poison the lookup our script's
    internal `import antifold.antiscripts` performs. Restores the prior state
    afterward so a later test file's own vendored import is unaffected by
    this one having run."""
    saved = {name: mod for name, mod in sys.modules.items() if name.split(".")[0] == "antifold"}
    for name in saved:
        del sys.modules[name]

    spec = importlib.util.spec_from_file_location("_isolated_antifold", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    yield module

    for name in list(sys.modules):
        if name.split(".")[0] == "antifold":
            del sys.modules[name]
    sys.modules.update(saved)


# RCSB 1L2Y, model 1 only, chain relabelled A -> H. An arbitrary small real
# backbone (Trp-cage, 20 residues) — not an antibody domain, and never read
# as one: this only exercises AntiFold's PDB parsing and forward pass, which
# take any single-chain backbone.
_NANOBODY_LIKE_PDB = """\
MODEL        1
ATOM      1  N   ASN H   1      -8.901   4.127  -0.555  1.00  0.00           N
ATOM      2  CA  ASN H   1      -8.608   3.135  -1.618  1.00  0.00           C
ATOM      3  C   ASN H   1      -7.117   2.964  -1.897  1.00  0.00           C
ATOM      4  O   ASN H   1      -6.634   1.849  -1.758  1.00  0.00           O
ATOM     17  N   LEU H   2      -6.379   4.031  -2.228  1.00  0.00           N
ATOM     18  CA  LEU H   2      -4.923   4.002  -2.452  1.00  0.00           C
ATOM     19  C   LEU H   2      -4.136   3.187  -1.404  1.00  0.00           C
ATOM     20  O   LEU H   2      -3.391   2.274  -1.760  1.00  0.00           O
ATOM     36  N   TYR H   3      -4.354   3.455  -0.111  1.00  0.00           N
ATOM     37  CA  TYR H   3      -3.690   2.738   0.981  1.00  0.00           C
ATOM     38  C   TYR H   3      -4.102   1.256   1.074  1.00  0.00           C
ATOM     39  O   TYR H   3      -3.291   0.409   1.442  1.00  0.00           O
ATOM     57  N   ILE H   4      -5.342   0.925   0.689  1.00  0.00           N
ATOM     58  CA  ILE H   4      -5.857  -0.449   0.613  1.00  0.00           C
ATOM     59  C   ILE H   4      -5.089  -1.221  -0.470  1.00  0.00           C
ATOM     60  O   ILE H   4      -4.621  -2.334  -0.226  1.00  0.00           O
ATOM     76  N   GLN H   5      -4.907  -0.601  -1.645  1.00  0.00           N
ATOM     77  CA  GLN H   5      -4.122  -1.167  -2.743  1.00  0.00           C
ATOM     78  C   GLN H   5      -2.629  -1.321  -2.390  1.00  0.00           C
ATOM     79  O   GLN H   5      -1.986  -2.240  -2.884  1.00  0.00           O
ATOM     93  N   TRP H   6      -2.074  -0.459  -1.528  1.00  0.00           N
ATOM     94  CA  TRP H   6      -0.716  -0.631  -0.993  1.00  0.00           C
ATOM     95  C   TRP H   6      -0.631  -1.766   0.044  1.00  0.00           C
ATOM     96  O   TRP H   6       0.295  -2.579  -0.004  1.00  0.00           O
ATOM    117  N   LEU H   7      -1.600  -1.860   0.967  1.00  0.00           N
ATOM    118  CA  LEU H   7      -1.641  -2.932   1.963  1.00  0.00           C
ATOM    119  C   LEU H   7      -1.847  -4.319   1.342  1.00  0.00           C
ATOM    120  O   LEU H   7      -1.144  -5.248   1.742  1.00  0.00           O
ATOM    136  N   LYS H   8      -2.753  -4.481   0.360  1.00  0.00           N
ATOM    137  CA  LYS H   8      -3.024  -5.791  -0.269  1.00  0.00           C
ATOM    138  C   LYS H   8      -1.796  -6.427  -0.937  1.00  0.00           C
ATOM    139  O   LYS H   8      -1.719  -7.648  -1.030  1.00  0.00           O
ATOM    158  N   ASP H   9      -0.828  -5.607  -1.355  1.00  0.00           N
ATOM    159  CA  ASP H   9       0.466  -6.016  -1.905  1.00  0.00           C
ATOM    160  C   ASP H   9       1.481  -6.464  -0.832  1.00  0.00           C
ATOM    161  O   ASP H   9       2.545  -6.971  -1.194  1.00  0.00           O
ATOM    170  N   GLY H  10       1.185  -6.278   0.464  1.00  0.00           N
ATOM    171  CA  GLY H  10       2.060  -6.618   1.593  1.00  0.00           C
ATOM    172  C   GLY H  10       2.628  -5.412   2.353  1.00  0.00           C
ATOM    173  O   GLY H  10       3.496  -5.594   3.208  1.00  0.00           O
TER     305      SER H  20
ENDMDL
END
"""


@pytest.fixture
def nanobody_pdb(tmp_path):
    path = tmp_path / "nanobody.pdb"
    path.write_text(_NANOBODY_LIKE_PDB)
    return path


@pytest.fixture
def untrained_model(isolated_antifold_module):
    """`_load_IF1_local()` builds the architecture; no checkpoint load, so
    this needs no mounted weights asset — only the forward pass's plumbing
    is under test here, never a prediction's numeric value. Goes through
    `isolated_antifold_module`'s own vendor path so it resolves the same
    `antifold.esm` this test's `_run_model` call will use."""
    if isolated_antifold_module._VENDOR_DIR not in sys.path:
        sys.path.insert(0, isolated_antifold_module._VENDOR_DIR)
    import antifold.esm.pretrained as pretrained

    model, _ = pretrained._load_IF1_local()
    return model.eval()


@pytest.fixture
def sapiens_weights_root():
    if not _SAPIENS_WEIGHTS_ROOT.is_dir():
        pytest.skip(f"sibling asset repo not checked out beside this one: {_SAPIENS_WEIGHTS_ROOT}")
    return str(_SAPIENS_WEIGHTS_ROOT)


@pytest.fixture
def antifold_weights_path(tmp_path, untrained_model):
    """A fully offline AntiFold checkpoint: the untrained model's own state
    dict, saved and read back through the real `load_IF1_checkpoint` path —
    the same trick this file's other fixtures use to avoid a real download."""
    import torch

    path = tmp_path / "model.pt"
    torch.save(untrained_model.state_dict(), path)
    return str(path)


class TestRealEntrypointWithTheHumanPrior:
    """Enters `read_tolerance.main(argv)` itself — the exact entrypoint the workflow
    invokes — rather than `_run_model` or `_predict_scores` directly, so this
    covers the wiring between the tolerance read and the prior read too."""

    def test_both_parents_get_a_tolerance_and_a_prior_file(
        self,
        isolated_antifold_module,
        nanobody_pdb,
        antifold_weights_path,
        sapiens_weights_root,
        tmp_path,
    ):
        pytest.importorskip("sapiens")

        pdb_dir = tmp_path / "pdbs"
        residues_dir = tmp_path / "residues"
        triaged_dir = tmp_path / "triaged"
        for one_dir in (pdb_dir, residues_dir, triaged_dir):
            one_dir.mkdir()

        pdb_text = nanobody_pdb.read_text()
        entries = []
        for stem in ("parent-1", "parent-2"):
            (pdb_dir / f"{stem}.pdb").write_text(pdb_text)
            # The ten residues `_NANOBODY_LIKE_PDB` stages: eight framework
            # positions and, at offsets 3-4, one contiguous CDR1 pair — so the
            # prior's row count must differ from the residue count, proving
            # the framework filter still applies to a whole-chain scoring.
            residues = [
                residue_store.Residue(
                    chain="H", offset=i, imgt=str(i + 1), wild_type="A", res_name="ALA",
                    b_factor=20.0, region="CDR1" if i in (3, 4) else "FR1", chain_role="H",
                )
                for i in range(10)
            ]
            residue_store.write_residues(str(residues_dir / f"{stem}.json"), residues)
            liability_store.write_triaged(
                str(triaged_dir / f"{stem}.json"),
                [
                    liability_triage.Triaged(
                        definition_id="deamidation_ng",
                        liability_type="deamidation",
                        risk_level="High",
                        fixability="fixable",
                        site=residues[:1],
                        verdict="exposed",
                        low_confidence=False,
                        confidence_angstroms=3.0,
                        rsasa=0.5,
                    )
                ],
            )
            entries.append(pdb_index_store.Entry(clonotype_key=stem, filename=f"{stem}.pdb"))

        index_path = tmp_path / "pdb_index.tsv"
        pdb_index_store.write_index(str(index_path), entries)

        out_dir = tmp_path / "tolerance"
        out_skip = tmp_path / "skip.tsv"

        rc = isolated_antifold_module.main(
            [
                "--pdb-dir", str(pdb_dir),
                "--residues-dir", str(residues_dir),
                "--triaged-dir", str(triaged_dir),
                "--pdb-index", str(index_path),
                "--weights", antifold_weights_path,
                "--sapiens-weights", sapiens_weights_root,
                "--out-tolerance-dir", str(out_dir),
                "--out-skip", str(out_skip),
            ]
        )

        assert rc == 0
        # Every parent completed inside `_block_network()` with no exception —
        # a `backend-failed` row here would mean the model call raised.
        assert skip_store.read_skips(str(out_skip)) == [
            ("parent-1", "", ""),
            ("parent-2", "", ""),
        ]
        for stem in ("parent-1", "parent-2"):
            assert (out_dir / f"{stem}.tsv").is_file()
            prior_path = out_dir / f"{stem}{isolated_antifold_module.sapiens_prior.PRIOR_SUFFIX}"
            assert prior_path.is_file()
            # Header plus one row per framework position — eight, not the ten
            # residues staged: the two CDR1 offsets reached the model (the
            # whole-chain scoring) but are absent from what got emitted.
            assert len(prior_path.read_text().splitlines()) - 1 == 8


class TestNanobodyReachesTheForwardPass:
    """Regression tests for three stacked fixes: a nanobody row used to fail
    before the model ever ran — first on AntiFold's CSV column-presence
    check (missing `Lchain` column), then on its H/L chain lookup
    (`Lchain=None` passed as a real chain id), then — MPS backend absent
    only — on an unbound `device` inside the forward pass itself. All three
    were fixed; these prove the real vendored code accepts the result, not
    just the row's shape."""

    def test_a_nanobody_pdb_produces_one_tolerance_row_per_residue(
        self, isolated_antifold_module, nanobody_pdb, untrained_model
    ):
        rows = isolated_antifold_module._run_model(
            untrained_model, str(nanobody_pdb), h_chain="H", l_chain=None, nanobody_mode=True
        )

        assert len(rows) == 10  # the ten residues staged above
        assert {r["chain"] for r in rows} == {"H"}

    def test_a_nanobody_pdb_still_forwards_on_a_machine_with_no_mps_backend(
        self, isolated_antifold_module, nanobody_pdb, untrained_model, monkeypatch
    ):
        """Every CUDA/Linux host (the block's real deployment target) has no
        MPS backend. AntiFold's own device-resolution line inside the
        forward pass read an unbound `device` on that path and silently
        swallowed the resulting `UnboundLocalError` — this Mac's MPS backend
        being present is what hid it from the test above."""
        import torch

        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        rows = isolated_antifold_module._run_model(
            untrained_model, str(nanobody_pdb), h_chain="H", l_chain=None, nanobody_mode=True
        )

        assert len(rows) == 10


# Trastuzumab's real heavy-chain V-domain (RCSB 1N8Z chain B, V-domain part),
# the same sequence the manual test drives `humanness_gate.identity` against.
_TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTI"
    "SADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)

# Trastuzumab's canonical IMGT heavy-chain CDR loops, located by substring
# within the V-domain above rather than hardcoded as offsets — the spans and
# the loop strings this list names can never disagree.
_CDR_LOOPS = ("GFNIKDTY", "RIYPTNGYT", "SRWGGDGFYAMDY")


def _span(loop):
    start = _TRASTUZUMAB_VH.index(loop)
    return start, start + len(loop)


_CDR_SPANS = dict(zip(("CDR1", "CDR2", "CDR3"), (_span(loop) for loop in _CDR_LOOPS), strict=True))


def _region_at(offset):
    for name, (start, end) in _CDR_SPANS.items():
        if start <= offset < end:
            return name
    if offset < _CDR_SPANS["CDR1"][0]:
        return "FR1"
    if offset < _CDR_SPANS["CDR2"][0]:
        return "FR2"
    if offset < _CDR_SPANS["CDR3"][0]:
        return "FR3"
    return "FR4"


def _heavy_chain_index():
    """Trastuzumab's real heavy-chain V-domain, indexed with its real CDR
    spans — the fixture the divergence case scores."""
    return [
        residue_store.Residue(
            chain="H",
            offset=offset,
            imgt=str(offset + 1),
            wild_type=wild_type,
            res_name="ALA",
            b_factor=20.0,
            region=_region_at(offset),
            chain_role="H",
        )
        for offset, wild_type in enumerate(_TRASTUZUMAB_VH)
    ]


def _log_softmax_rows(frame, length):
    return [
        dict(
            zip(
                sapiens_prior.AMINO_ACIDS,
                sapiens_prior._log_softmax(
                    [float(frame[aa][position]) for aa in sapiens_prior.AMINO_ACIDS]
                ),
                strict=True,
            )
        )
        for position in range(length)
    ]


def _argmax_amino_acid(row):
    return max(sapiens_prior.AMINO_ACIDS, key=lambda aa: row[aa])


def _kl_divergence(p_log, q_log):
    return sum(math.exp(p_log[aa]) * (p_log[aa] - q_log[aa]) for aa in sapiens_prior.AMINO_ACIDS)


class TestThePriorMatchesAWholeChainScoring:
    """Scores the same real heavy chain two ways outside `build_prior_rows` —
    once over the whole in-scope chain, once over the framework-only
    concatenation the defect used to send — and quantifies how far apart
    they land. `emitted` must match the whole-chain scoring exactly and
    diverge from the framework-only one: the claim this row exists for."""

    def test_the_prior_matches_a_whole_chain_scoring(self, sapiens_weights_root):
        pytest.importorskip("sapiens")

        residues = _heavy_chain_index()
        framework_residues = sorted(
            (r for r in residues if r.region.startswith("FR")), key=lambda r: r.offset
        )
        # Pins the IMGT partition itself: a future re-scheme of `_CDR_LOOPS`
        # that silently shrinks or grows the CDR set must fail here first,
        # rather than only move the printed divergence number.
        assert len(framework_residues) == 90
        whole_sequence = "".join(r.wild_type for r in sorted(residues, key=lambda r: r.offset))
        fr_only_sequence = "".join(r.wild_type for r in framework_residues)

        checkpoint_dir = sapiens_prior._checkpoint_dir(sapiens_weights_root, "H")
        tokenizer_dir = str(Path(sapiens_weights_root) / "tokenizer")

        whole_rows = _log_softmax_rows(
            sapiens_prior._predict_scores(whole_sequence, "H", checkpoint_dir, tokenizer_dir),
            len(whole_sequence),
        )
        fr_only_rows = _log_softmax_rows(
            sapiens_prior._predict_scores(fr_only_sequence, "H", checkpoint_dir, tokenizer_dir),
            len(fr_only_sequence),
        )
        emitted = sapiens_prior.build_prior_rows(residues, sapiens_weights_root)
        emitted_by_imgt = {row["imgt"]: row for row in emitted}

        divergences = []
        flips = 0
        for position, residue in enumerate(framework_residues):
            whole_row = whole_rows[residue.offset]
            fr_only_row = fr_only_rows[position]
            divergences.append(_kl_divergence(whole_row, fr_only_row))
            if _argmax_amino_acid(whole_row) != _argmax_amino_acid(fr_only_row):
                flips += 1

            emitted_row = emitted_by_imgt[residue.imgt]
            for aa in sapiens_prior.AMINO_ACIDS:
                assert emitted_row[aa] == pytest.approx(whole_row[aa])

        print(
            f"prior divergence over {len(framework_residues)} framework positions: "
            f"mean KL {statistics.mean(divergences):.4f} nats, max KL {max(divergences):.4f} nats, "
            f"{flips} top-scoring-residue flip(s)"
        )
        # A whole-chain scoring that agreed everywhere with the framework-only
        # one would mean the CDRs never moved anything — not what a checkpoint
        # with absolute position embeddings does with a shorter input.
        assert flips > 0
