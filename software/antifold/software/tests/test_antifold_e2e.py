"""End-to-end tests for `antifold.py` against the real vendored AntiFold
package. The only test file in this package, and the only one that needs the
heavy set: run it with `uv run --extra antifold --group e2e pytest -m e2e`.
Every other test of this module is a unit test and lives in the
developability package's suite, which never installs torch — so this file
skips itself whenever torch is absent.

`src/antifold.py` and the vendored `src/vendor/AntiFold/antifold/` package
share the literal name `antifold`. Our flat script wins that name whenever
it is imported first, and then AntiFold's own internal `import
antifold.antiscripts` resolves against the script instead of the package —
which only bites once `_run_model` runs for real, as it does here and
nowhere else. `_isolated_antifold_module` loads our script under a private
name and clears any such stale binding around each test, so this file's
result never depends on what imported `antifold` before it, and it never
leaves a changed `sys.modules['antifold']` behind.

No real checkpoint is needed: `_load_IF1_local()` alone builds a full,
untrained model, and every case here only checks that AntiFold's own
coordinate-loading and forward-pass plumbing accepts the shape `_run_model`
builds — never the predicted values.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

pytestmark = pytest.mark.e2e

_SCRIPT_PATH = Path(__file__).parent.parent / "src" / "antifold.py"


@pytest.fixture
def isolated_antifold_module():
    """Load our own script under a name distinct from `antifold`, and evict
    any `antifold`/`antifold.*` entry already in `sys.modules` — a sibling
    test file's `import antifold` would otherwise poison the lookup our
    script's internal `import antifold.antiscripts` performs. Restores the
    prior state afterward so a later test file's own `import antifold` is
    unaffected by this one having run."""
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
