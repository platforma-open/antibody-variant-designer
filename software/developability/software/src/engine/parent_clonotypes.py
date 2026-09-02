"""The clonotypes one run processes, recovered from a staged directory listing.

The workflow stages one file per clonotype and names it after the clonotype key,
so the listing carries them and no separate index file has to. A workflow
that wrote the list out instead would hold every key in one value resource,
which the platform caps at 3 MiB — about 40k clonotypes.

A clonotype the upstream Structure Prediction block failed for has no entry in
its PDB ResourceMap at all, so nothing stages a file for it and it is absent
from this list. Absence upstream and absence here mean the same thing.
"""

from dataclasses import dataclass
from pathlib import Path

PDB_SUFFIX = ".pdb"
ARTIFACT_SUFFIX = ".json"


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
