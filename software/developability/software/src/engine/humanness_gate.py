"""How human a sequence reads, and the sequence to read it on.

Wraps promb's OASis peptide-content measurement behind a narrow seam.
The rest of the package sees a 0-100 score, or `None` when a sequence
cannot be scored. It never sees the database, the tier the score is
measured at, or how loading it is cached.
"""

from functools import lru_cache

from engine import residue_store

MIN_WINDOW = 9
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYX")


@lru_cache(maxsize=1)
def _database():
    from promb import init_db

    return init_db("human-oas")


def identity(sequence: str | None) -> float | None:
    """Scores how human `sequence` reads: 0-100, rounded to two decimals.

    Returns `None` when the sequence cannot be scored — too short to yield one peptide window,
    or holding a character outside `AA_ALPHABET` once uppercased.

    An unscoreable sequence must discard its candidate. This function always returns `None` for
    that case, never raises."""
    if not isinstance(sequence, str) or len(sequence) < MIN_WINDOW:
        return None
    upper = sequence.upper()
    if not set(upper) <= AA_ALPHABET:
        return None
    try:
        fraction = _database().compute_peptide_content(upper)
    except Exception:
        return None
    return round(float(fraction) * 100.0, 2)


def chain_sequence(
    residues: list[residue_store.Residue],
    chain: str,
    substituted: list[residue_store.Residue],
) -> str:
    """One chain's in-scope residues, in offset order.

    Each residue is replaced by its entry in `substituted` when one shares its
    `(chain, offset)`; otherwise its own wild type stands.

    Reads the chain from `residue_store.in_scope_chains`, the one minter, so this measures
    the same sequence a scan would see without keeping a second derivation of it.

    Pass an empty `substituted` for a sequence with no edits applied."""
    replacement = {(r.chain, r.offset): r.wild_type for r in substituted}
    in_scope_chain = next(
        (c for c in residue_store.in_scope_chains(residues) if c.chain == chain), None
    )
    if in_scope_chain is None:
        return ""
    return "".join(
        replacement.get((r.chain, r.offset), r.wild_type) for r in in_scope_chain.residues
    )
