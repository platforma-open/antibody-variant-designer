"""Unit tests for `parent_clonotypes.py` — recovering the parent clonotypes from a staged
directory listing, and the artifact names an entry carries."""

from engine import parent_clonotypes


class TestScanParentClonotypes:
    def test_it_recovers_one_entry_per_staged_file(self, tmp_path):
        for name in ("clone-1.pdb", "clone-2.pdb"):
            (tmp_path / name).write_text("")

        parents = parent_clonotypes.scan_parent_clonotypes(str(tmp_path), ".pdb")

        assert [e.clonotype_key for e in parents] == ["clone-1", "clone-2"]

    def test_it_orders_by_clonotype_key_whatever_the_listing_order(self, tmp_path):
        # Every step iterates this order, so each one's output rows stay in the
        # same order as every other step's.
        for name in ("b.pdb", "a.pdb", "a-.pdb"):
            (tmp_path / name).write_text("")

        parents = parent_clonotypes.scan_parent_clonotypes(str(tmp_path), ".pdb")

        assert [e.clonotype_key for e in parents] == ["a", "a-", "b"]

    def test_it_ignores_a_file_carrying_another_suffix(self, tmp_path):
        # A step's workdir holds its outputs beside its inputs, and the tolerance
        # step writes a prior TSV into the directory it also reads.
        (tmp_path / "clone-1.json").write_text("")
        (tmp_path / "clone-1.prior.tsv").write_text("")

        parents = parent_clonotypes.scan_parent_clonotypes(str(tmp_path), ".json")

        assert [e.clonotype_key for e in parents] == ["clone-1"]

    def test_it_keeps_a_dot_inside_the_key(self, tmp_path):
        # Only the trailing suffix is stripped, so a key holding a dot survives
        # the round trip through the filename.
        (tmp_path / "clone.1.pdb").write_text("")

        parents = parent_clonotypes.scan_parent_clonotypes(str(tmp_path), ".pdb")

        assert [e.clonotype_key for e in parents] == ["clone.1"]

    def test_an_empty_directory_holds_no_parent_clonotype(self, tmp_path):
        assert parent_clonotypes.scan_parent_clonotypes(str(tmp_path), ".pdb") == []

    def test_a_subdirectory_is_not_a_parent_clonotype(self, tmp_path):
        (tmp_path / "nested.pdb").mkdir()

        assert parent_clonotypes.scan_parent_clonotypes(str(tmp_path), ".pdb") == []


class TestParentClonotype:
    def test_the_stem_and_the_filename_both_come_from_the_key(self, tmp_path):
        entry = parent_clonotypes.ParentClonotype(clonotype_key="clone-1")

        assert entry.stem == "clone-1"
        assert entry.filename == "clone-1.pdb"
