"""AntiFold tolerance read, the entrypoint that writes `tolerance.tsv`.

Calls AntiFold as a library, never its own CLI or subprocess:
`_load_IF1_local()` builds the architecture, `load_IF1_checkpoint` loads the
mounted weights, and `get_pdbs_logits(..., save_flag=False)` returns raw
logits per position, never a CSV on disk. Any exception from torch or from
AntiFold itself propagates out of this process uncaught — a per-antibody
backend fault becomes a failed exec, which the workflow, not this script,
turns into a `backend-failed` record for that one parent, without stopping
the run for the others.

The vendored `antifold` package under `vendor/AntiFold` shares this file's
own name, so the vendor directory is pushed to the front of `sys.path`
right before it is imported — the interpreter's own auto-prepended script
directory would otherwise shadow the package with this very file. That
import, and every other torch-dependent call, is deferred inside
`_run_model`, so loading this module for its pure functions never requires
the heavy backend to be installed.

The checkpoint always comes from the mounted `--weights` path, never a
download. The vendored package ships two functions that can reach the
network — `antiscripts.load_model()`'s `urllib.request.urlretrieve` call to
`opig.stats.ox.ac.uk`, and `esm.pretrained.load_hub_workaround`'s
`torch.hub.load_state_dict_from_url` — and a call-graph audit of the whole
vendored tree found neither one reachable from `_load_IF1_local()` +
`load_IF1_checkpoint()` + `get_pdbs_logits()`, the only three entry points
`_run_model` calls. `_block_network` is the belt-and-braces layer on top of
that audit: it makes any *future* vendor bump that reintroduces a reachable
download fail loudly during the model call, instead of silently fetching.
"""

import argparse
import math
import socket
import sys
from contextlib import contextmanager
from pathlib import Path

import residue_store
import tolerance_store

_VENDOR_DIR = str(Path(__file__).parent / "vendor" / "AntiFold")

AMINO_ACIDS = tolerance_store.AMINO_ACIDS


def _write_skip(path: str, reason: str) -> None:
    Path(path).write_text(reason)


def pick_chains(
    residues: list[residue_store.Residue],
) -> tuple[str, str | None, bool]:
    """The physical PDB chain letters carrying the `H` and `L` roles, and
    whether this antibody is a nanobody — decided here, from the parsed
    chain count, because one run may mix VHH and paired antibodies and
    Tengo has no residue index of its own to decide it from.

    Returns `(h_chain, l_chain, nanobody_mode)`; `l_chain` is `None` and
    `nanobody_mode` is `True` when no residue carries the `L` role."""
    h_chain: str | None = None
    l_chain: str | None = None
    for residue in residues:
        if residue.chain_role == "H" and h_chain is None:
            h_chain = residue.chain
        elif residue.chain_role == "L" and l_chain is None:
            l_chain = residue.chain
    if h_chain is None:
        raise ValueError(
            "no residue carries the H role — structure.py should have skipped this antibody"
        )
    return h_chain, l_chain, l_chain is None


def _log_softmax(values: list[float]) -> list[float]:
    """Plain-Python log-softmax, no torch tensor involved — the one
    conversion `save_flag=False`'s raw logits need before they are
    comparable to each other or to a threshold. Kept independent of the
    backend so this conversion, and everything built on it, stays
    unit-testable without torch installed."""
    m = max(values)
    shifted = [v - m for v in values]
    log_sum_exp = math.log(sum(math.exp(v) for v in shifted))
    return [v - log_sum_exp for v in shifted]


def build_tolerance_rows(
    residues: list[residue_store.Residue], logits_rows: list[dict]
) -> list[dict]:
    """Join AntiFold's own per-position frame onto this antibody's H/L
    residues by `(chain, posins)`. `logits_rows` is
    `[{"chain", "posins", "perplexity", "logits": {aa: float, ...}}, ...]`,
    already read out of AntiFold's own DataFrame.

    Raises, rather than trusting AntiFold's length-only assert, when a
    residue this antibody needs has no matching row: the recoverable
    `backend-failed` verdict belongs to the workflow driving this exec, not
    to a silently short join happening inside it."""
    by_key = {(row["chain"], row["posins"]): row for row in logits_rows}

    wanted_chains = {r.chain for r in residues if r.chain_role in ("H", "L")}
    missing = [
        (r.chain, r.imgt)
        for r in residues
        if r.chain in wanted_chains and (r.chain, r.imgt) not in by_key
    ]
    if missing:
        raise ValueError(
            f"AntiFold's output has no row for {len(missing)} residue(s), e.g. {missing[:3]}"
        )

    rows = []
    for row in logits_rows:
        log_probs = _log_softmax([row["logits"][aa] for aa in AMINO_ACIDS])
        out = {"chain": row["chain"], "posins": row["posins"], "perplexity": row["perplexity"]}
        out.update(dict(zip(AMINO_ACIDS, log_probs, strict=True)))
        rows.append(out)
    return rows


@contextmanager
def _block_network():
    """Raise instead of silently reaching the network, for the duration of
    the model call. `socket.socket` is the one chokepoint every stdlib and
    torch network path routes through — `urllib`, `torch.hub`, `requests`
    all end up constructing one — so patching it here catches any of them,
    known or not yet written, without having to name each caller."""

    def _blocked(*_args, **_kwargs):
        raise RuntimeError(
            "network access is disabled during AntiFold's model call; "
            "the checkpoint must come from the mounted --weights path, never a download"
        )

    original_socket = socket.socket
    socket.socket = _blocked
    try:
        yield
    finally:
        socket.socket = original_socket


def _run_model(
    pdb_path: str, weights_path: str, h_chain: str, l_chain: str | None, nanobody_mode: bool
) -> list[dict]:
    """The only function that touches torch or the vendored AntiFold
    package. Builds AntiFold's one-row `pdb, Hchain, Lchain` frame for this
    single antibody and reads its `df_logits` back into plain dicts, so
    every function above this one stays pandas- and torch-free."""
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)
    import antifold.antiscripts as antiscripts
    import antifold.esm.pretrained as pretrained
    import pandas as pd

    with _block_network():
        model, _ = pretrained._load_IF1_local()
        model = antiscripts.load_IF1_checkpoint(model, weights_path)
        model = model.eval()

        pdb = Path(pdb_path)
        row = {"pdb": pdb.stem, "Hchain": h_chain}
        if l_chain is not None:
            row["Lchain"] = l_chain
        pdbs_df = pd.DataFrame([row])

        [df_logits] = antiscripts.get_pdbs_logits(
            model, pdbs_df, str(pdb.parent), nanobody_mode=nanobody_mode, save_flag=False
        )

    return [
        {
            "chain": r["pdb_chain"],
            "posins": r["pdb_posins"],
            "perplexity": float(r["perplexity"]),
            "logits": {aa: float(r[aa]) for aa in AMINO_ACIDS},
        }
        for _, r in df_logits.iterrows()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read AntiFold's per-position tolerance for one antibody."
    )
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--residues", required=True, help="structure.py's output")
    parser.add_argument("--weights", required=True, help="mounted models/model.pt")
    parser.add_argument("--out-tolerance", required=True)
    parser.add_argument("--out-skip", required=True)
    args = parser.parse_args(argv)

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        raise SystemExit(f"--weights does not exist: {weights_path}")

    residues = residue_store.read_residues(args.residues)
    h_chain, l_chain, nanobody_mode = pick_chains(residues)

    logits_rows = _run_model(args.pdb, str(weights_path), h_chain, l_chain, nanobody_mode)
    rows = build_tolerance_rows(residues, logits_rows)

    tolerance_store.write_tolerance_tsv(args.out_tolerance, rows)
    _write_skip(args.out_skip, "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
