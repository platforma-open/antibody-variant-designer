"""One test per target antibody format — the cases `CLAUDE.md` names.

Each test runs the index phase and asserts the two things a format decides:
the skip reason, and which residues came out researchable. A format that is
not in `CLAUDE.md` must reach a named skip here, never a partial index.
"""

import json
from pathlib import Path

import liability_cysteines
import liability_motifs
import residue_store
from pdb_fixtures import make_pdb, platforma_cdr_remark
from residue_index import index_one

TAXONOMY = [
    {"id": "deamidation_ng", "name": "Deamidation (N[GS])",
     "liabilityType": "deamidation", "motif": r"N[GS]",
     "riskLevel": "High", "fixability": "fixable"},
    {"id": "missing_cysteines", "name": "Missing Cysteines",
     "liabilityType": "cysteine", "motif": None,
     "riskLevel": "High", "fixability": "structural"},
    {"id": "extra_cysteines", "name": "Extra Cysteines",
     "liabilityType": "cysteine", "motif": None,
     "riskLevel": "High", "fixability": "hard_to_fix"},
]

# The two IMGT-conserved cysteines of a well-formed V domain: FR1's at 24,
# FR3's at the region's last position, 104.
CONSERVED_CYS = (24, 104)


def v_domain(chain, cys_at=CONSERVED_CYS, overrides=None):
    """One V domain, IMGT 1..128 — ALA everywhere except `overrides`
    (`res_seq -> res_name`) and the conserved cysteines."""
    overrides = overrides or {}
    return [
        (chain, p, " ", overrides.get(p, "CYS" if p in cys_at else "ALA"), 1.0)
        for p in range(1, 129)
    ]


def c_domain(chain, start, overrides=None):
    """One constant domain of 100 residues, numbered from `start`."""
    overrides = overrides or {}
    return [(chain, p, " ", overrides.get(p, "ALA"), 1.0) for p in range(start, start + 100)]


def linker(chain, start=200, length=15):
    """A (G4S)3 scFv linker, numbered clear of both V domains."""
    return [(chain, start + i, " ", "GLY", 1.0) for i in range(length)]


def remarks(role, chain):
    """The three `REMARK 99 PLATFORMA CDR` records that give `chain` a role."""
    return "\n".join(
        platforma_cdr_remark(role, i, chain, start, end)
        for i, (start, end) in enumerate([(27, 38), (56, 65), (105, 117)], start=1)
    )


def run_structure(batch, text):
    """Run the index phase over one antibody's `text`. Returns
    `(skip_reason, residues)`.

    Calls `index_one` rather than the `index-and-scan` CLI because these
    cases assert the index phase's own contract — which reason a format gets
    and which residues come out researchable — not how a later phase reads
    it. `residues` is empty when the phase skipped, because a skipped
    antibody deliberately leaves no file."""
    entry = batch.add("clonotype-1", text)
    out_residues = Path(batch.dir("residues"), f"{entry.stem}.json")

    reason = index_one(str(Path(batch.pdb_dir, entry.filename)), str(out_residues))

    residues = (
        residue_store.read_residues(str(out_residues)) if out_residues.is_file() else []
    )
    return reason, residues


def researched(residues):
    return [r for r in residues if r.in_scope]


class TestCase1Nanobody:
    """One chain, role H, V domain only — researched in full."""

    def test_whole_chain_is_researched(self, batch):
        skip, residues = run_structure(
            batch, remarks("H", "H") + "\n" + make_pdb(v_domain("H"))
        )

        assert skip == ""
        assert len(residues) == 128
        assert len(researched(residues)) == 128
        assert {r.chain_role for r in residues} == {"H"}

    def test_vhh_hallmark_cdr_disulfide_is_not_an_extra_cysteine(self, batch):
        # The CDR1-CDR3 pair a canonical VHH carries. `liability_cysteines.py` counts
        # FR1 and FR3 only, so this must stay silent.
        _, residues = run_structure(
            batch,
            remarks("H", "H") + "\n" + make_pdb(v_domain("H", cys_at=(24, 104, 33, 110))),
        )

        assert liability_cysteines.detect_all(residues, TAXONOMY) == []


class TestCase2Fv:
    """Chains H and L, V domains only — both researched, neither joined."""

    def test_both_chains_are_researched(self, batch):
        skip, residues = run_structure(
            batch,
            remarks("H", "H") + "\n" + remarks("L", "L") + "\n"
            + make_pdb(v_domain("H") + v_domain("L")),
        )

        assert skip == ""
        assert len(researched(residues)) == 256
        assert {r.chain for r in researched(residues)} == {"H", "L"}

    def test_no_motif_spans_the_two_chains(self, batch):
        # H ends in ASN, L opens with GLY. Joined, that is an `N[GS]` hit;
        # per chain it is nothing.
        _, residues = run_structure(
            batch,
            remarks("H", "H") + "\n" + remarks("L", "L") + "\n"
            + make_pdb(v_domain("H", overrides={128: "ASN"})
                       + v_domain("L", overrides={1: "GLY"})),
        )

        assert liability_motifs.detect_all(residues, TAXONOMY) == []


class TestCase3ScFv:
    """VH + linker + VL on one chain — a named skip, not a partial index."""

    def test_two_v_domains_on_one_chain_skip(self, batch):
        skip, residues = run_structure(
            batch,
            remarks("H", "A") + "\n" + remarks("L", "A") + "\n"
            + make_pdb(v_domain("A") + linker("A") + v_domain("A")),
        )

        assert skip == "structure-multi-domain-chain"
        assert residues == []

    def test_two_roles_naming_one_chain_skip(self, batch):
        # The role collision alone is enough, even when the numbering
        # happens not to repeat.
        skip, _ = run_structure(
            batch,
            remarks("H", "A") + "\n" + remarks("L", "A") + "\n"
            + make_pdb(v_domain("A")),
        )

        assert skip == "structure-multi-domain-chain"


class TestCase4Fab:
    """H = VH + CH1, L = VL + CL — V domains researched, C domains not."""

    def test_constant_domains_are_indexed_but_not_researched(self, batch):
        skip, residues = run_structure(
            batch,
            remarks("H", "H") + "\n" + remarks("L", "L") + "\n"
            + make_pdb(v_domain("H") + c_domain("H", 129)
                       + v_domain("L") + c_domain("L", 129)),
        )

        assert skip == ""
        assert len(residues) == 456  # the C domains are still in the index
        assert len(researched(residues)) == 256
        assert all(r.region is not None for r in researched(residues))

    def test_no_liability_is_reported_inside_a_constant_domain(self, batch):
        _, residues = run_structure(
            batch,
            remarks("H", "H") + "\n"
            + make_pdb(v_domain("H") + c_domain("H", 129, overrides={150: "ASN", 151: "GLY"})),
        )

        assert liability_motifs.detect_all(residues, TAXONOMY) == []

    def test_constant_domain_restarting_at_one_skips(self, batch):
        skip, residues = run_structure(
            batch,
            remarks("H", "H") + "\n" + make_pdb(v_domain("H") + c_domain("H", 1)),
        )

        assert skip == "structure-multi-domain-chain"
        assert residues == []


class TestCase5Mab:
    """Two heavy and two light chains — only the role-bearing pair is researched."""

    def test_second_arm_is_indexed_but_not_researched(self, batch):
        skip, residues = run_structure(
            batch,
            remarks("H", "H") + "\n" + remarks("L", "L") + "\n"
            + make_pdb(v_domain("H") + v_domain("L") + v_domain("A") + v_domain("B")),
        )

        assert skip == ""
        assert len(residues) == 512
        assert {r.chain for r in researched(residues)} == {"H", "L"}

    def test_an_identical_second_arm_does_not_double_the_hits(self, batch):
        hit = {60: "ASN", 61: "GLY"}
        _, residues = run_structure(
            batch,
            remarks("H", "H") + "\n"
            + make_pdb(v_domain("H", overrides=hit) + v_domain("A", overrides=hit)),
        )

        hits = liability_motifs.detect_all(residues, TAXONOMY)

        assert [h.relevant.chain for h in hits] == ["H"]

    def test_an_antigen_chain_is_never_researched(self, batch):
        _, residues = run_structure(
            batch,
            remarks("H", "H") + "\n"
            + make_pdb(v_domain("H") + c_domain("X", 129, overrides={150: "ASN", 151: "GLY"})),
        )

        assert {r.chain for r in researched(residues)} == {"H"}
        assert liability_motifs.detect_all(residues, TAXONOMY) == []


class TestCase6HalfMab:
    """One role-bearing V domain plus a constant mass."""

    def test_only_the_one_variable_domain_is_researched(self, batch):
        skip, residues = run_structure(
            batch,
            remarks("H", "H") + "\n"
            + make_pdb(v_domain("H") + c_domain("H", 129)
                       + c_domain("C", 129) + c_domain("D", 129)),
        )

        assert skip == ""
        assert len(residues) == 428
        assert len(researched(residues)) == 128
        assert {r.chain for r in researched(residues)} == {"H"}

    def test_the_constant_mass_still_reaches_the_index_for_burial(self, batch):
        _, residues = run_structure(
            batch,
            remarks("H", "H") + "\n" + make_pdb(v_domain("H") + c_domain("C", 129)),
        )

        assert {r.chain for r in residues} == {"H", "C"}

    def test_a_file_with_no_variable_domain_skips(self, batch):
        skip, residues = run_structure(batch, make_pdb(c_domain("H", 1)))

        assert skip == "no-researchable-residue"
        assert residues == []


class TestBoundaryFileCarriesTheRole:
    """`chain_role` crosses the boundary file — `index_and_scan.py` and `read_tolerance.py`
    both re-read `residues.json` and apply the same scope rule the writer
    applied."""

    def test_chain_role_round_trips_through_residues_json(self, batch):
        _, residues = run_structure(
            batch,
            remarks("H", "H") + "\n" + make_pdb(v_domain("H") + c_domain("X", 129)),
        )

        rows = json.loads(Path(batch.dir("residues"), "clonotype-1.json").read_text())

        assert {row["chainRole"] for row in rows} == {"H", None}
        assert [r.chain_role for r in residues] == [row["chainRole"] for row in rows]
