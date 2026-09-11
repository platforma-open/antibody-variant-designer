"""The clonotypes one run processes.

Steps 1 and 2 stage one PDB blob per clonotype, named after the clonotype key, so a directory
listing recovers the roster with no separate index file. Step 3 stages no PDBs; its roster is
the keys the triaged artifact already carries, in the order it carries them.

A roster written as its own file would once have held every key in one value resource, capped
by the platform at 3 MiB — about 40k clonotypes. That reasoning does not reach a saved file,
which is a blob with no such cap, so step 3 reads its roster off the triaged artifact instead of
a directory listing.

A clonotype the upstream Structure Prediction block failed for has no entry in
its PDB ResourceMap at all, so nothing stages a file for it and it is absent
from both rosters. Absence upstream and absence here mean the same thing.
"""

from dataclasses import dataclass
from pathlib import Path

from engine import keyed_artifact

PDB_SUFFIX = ".pdb"


@dataclass(frozen=True)
class ParentClonotype:
    """One clonotype in the run.

    `stem` and `filename` are the key today, because `main.tpl.tengo` names each
    staged PDB after the key it belongs to. They stay separate properties so a
    producer that ever has to sanitize a key into a filename changes them here
    and nowhere else.
    """

    clonotype_key: str

    @property
    def stem(self) -> str:
        return self.clonotype_key

    @property
    def filename(self) -> str:
        return self.clonotype_key + PDB_SUFFIX


def scan_parent_clonotypes(directory: str, suffix: str) -> list[ParentClonotype]:
    """The parent clonotypes `directory` holds, in clonotype-key order.

    Every step iterates this order, so each one's output rows stay in the same
    order as every other step's.

    `suffix` is stripped off each filename to recover the key, so the caller
    passes the suffix the files in `directory` actually carry.
    """
    return [
        ParentClonotype(clonotype_key=key)
        for key in sorted(
            entry.name.removesuffix(suffix)
            for entry in Path(directory).iterdir()
            if entry.is_file() and entry.name.endswith(suffix)
        )
    ]


def read_parent_clonotypes(path: str) -> list[ParentClonotype]:
    """The parents a keyed artifact carries, in the order it carries them — which is
    clonotype-key order. Step 3 stages no PDBs, so the triaged artifact is its roster and its
    gate, as the triaged directory was."""
    return [ParentClonotype(clonotype_key=key) for key in keyed_artifact.keys_of(path)]
