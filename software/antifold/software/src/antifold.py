"""AntiFold tolerance read, the entrypoint that writes one
`tolerance.tsv` per antibody.

Calls AntiFold as a library, never its own CLI or subprocess:
`_load_IF1_local()` builds the architecture, `load_IF1_checkpoint` loads the
mounted weights, and `get_pdbs_logits(..., save_flag=False)` returns raw
logits per position, never a CSV on disk.

The checkpoint is loaded **once per process**, before the batch loop —
that single load is the whole reason one exec covers the dataset rather
than one exec per antibody. A missing checkpoint therefore fails the run
outright, while a fault raised *by* one antibody's model call is caught
inside the loop and recorded as `backend-failed` against that antibody
alone. AntiFold swallows its own exceptions and exits 0, so the loop is
the only place such a fault can still be named after its cause.

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

import batch
import pdb_index
import residue_store
import sapiens_prior
import tolerance_store

_VENDOR_DIR = str(Path(__file__).parent / "vendor" / "AntiFold")

AMINO_ACIDS = tolerance_store.AMINO_ACIDS


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


def load_model(weights_path: str):
    """Build the architecture and load the mounted checkpoint — **once per
    process**, before the batch loop. The 540.6 MiB load is the whole reason
    one exec covers the dataset instead of one per antibody, so it must not
    be reachable from inside the loop."""
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)
    import antifold.antiscripts as antiscripts
    import antifold.esm.pretrained as pretrained

    with _block_network():
        model, _ = pretrained._load_IF1_local()
        model = antiscripts.load_IF1_checkpoint(model, weights_path)
        return model.eval()


def _pdbs_frame(pdb_stem: str, h_chain: str, l_chain: str | None):
    """AntiFold's one-row `pdb, Hchain, Lchain` frame for a single antibody.

    `Lchain` is always a column, `None` for a nanobody. AntiFold's own
    dataset loader checks column presence before it reads a single row
    value, so a dataframe missing the column fails for every antibody,
    nanobody_mode notwithstanding — the per-row NaN is what its own
    chain-count inference already handles correctly. Split out from
    `_run_model` so this shape stays checkable without torch installed."""
    import pandas as pd

    return pd.DataFrame([{"pdb": pdb_stem, "Hchain": h_chain, "Lchain": l_chain}])


def _run_model(
    model, pdb_path: str, h_chain: str, l_chain: str | None, nanobody_mode: bool
) -> list[dict]:
    """The only function that touches torch or the vendored AntiFold
    package. Builds AntiFold's one-row frame for this single antibody and
    reads its `df_logits` back into plain dicts, so every function above
    this one stays pandas- and torch-free.

    Takes an already-loaded `model` so the checkpoint is read once per
    process rather than once per antibody."""
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)
    import antifold.antiscripts as antiscripts

    with _block_network():
        pdb = Path(pdb_path)
        pdbs_df = _pdbs_frame(pdb.stem, h_chain, l_chain)

        # `custom_chain_mode` and `nanobody_mode` are two independent flags on
        # AntiFold's own call, but its H/L coordinate loader looks up BOTH
        # chain letters from the dataframe regardless of `nanobody_mode` —
        # nanobody_mode alone only changes postprocessing, never which chains
        # get loaded. A nanobody's `Lchain=None` then reaches biotite's chain
        # lookup and raises "Chain None not found in input file". Only
        # `custom_chain_mode=True` switches the loader to the chain-letter
        # list it actually has (built from the non-null columns), which is
        # what a nanobody needs. AntiFold's own CLI always sets the two
        # together for a nanobody, never one without the other.
        [df_logits] = antiscripts.get_pdbs_logits(
            model,
            pdbs_df,
            str(pdb.parent),
            nanobody_mode=nanobody_mode,
            custom_chain_mode=nanobody_mode,
            save_flag=False,
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


def process_one(
    model,
    pdb_path: str,
    residues_path: str,
    out_tolerance: str,
    sapiens_weights: str,
    out_prior: str,
) -> str:
    """Read one antibody's tolerance matrix, and beside it its human-repertoire
    prior. Nanobody mode is decided here, from this antibody's own parsed chain
    count — one batch may legally mix VHH and paired structures, so it can
    never be a run-level flag."""
    residues = residue_store.read_residues(residues_path)
    h_chain, l_chain, nanobody_mode = pick_chains(residues)

    logits_rows = _run_model(model, pdb_path, h_chain, l_chain, nanobody_mode)
    rows = build_tolerance_rows(residues, logits_rows)

    tolerance_store.write_tolerance_tsv(out_tolerance, rows)

    with _block_network():
        prior_rows = sapiens_prior.build_prior_rows(residues, sapiens_weights)
    Path(out_prior).write_text(sapiens_prior.render(prior_rows))

    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read AntiFold's per-position tolerance for every staged antibody."
    )
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--residues-dir", required=True, help="scan.py's --out-residues-dir")
    parser.add_argument(
        "--triaged-dir",
        required=True,
        help="scan.py's --out-triaged-dir, read as a gate only — never opened",
    )
    parser.add_argument("--pdb-index", required=True, help="the pdb_index")
    parser.add_argument("--weights", required=True, help="mounted models/model.pt")
    parser.add_argument(
        "--sapiens-weights",
        required=True,
        help="the mounted Sapiens asset root, holding vh/, vl/ and tokenizer/",
    )
    parser.add_argument("--out-tolerance-dir", required=True)
    parser.add_argument("--out-skip", required=True)
    args = parser.parse_args(argv)

    # Asserted and loaded before the loop, and deliberately outside the
    # per-antibody `try` below: a missing or unreadable checkpoint is a
    # broken run, and must never degrade into N `backend-failed` rows that
    # read as N bad antibodies.
    weights_path = Path(args.weights)
    if not weights_path.is_file():
        raise SystemExit(f"--weights does not exist: {weights_path}")

    pdb_dir = Path(args.pdb_dir)
    residues_dir = Path(args.residues_dir)
    triaged_dir = Path(args.triaged_dir)
    out_dir = Path(args.out_tolerance_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = pdb_index.read_index(args.pdb_index)
    # A missing triaged file is a gate, not an error: it means triage left this
    # antibody nothing actionable, or the index phase skipped it. Either way
    # exec 1 already named the reason and no variant can come from it, so the
    # model call is pure waste. Filtered here rather than in the loop so a
    # dataset where every antibody is gated out never pays the checkpoint load.
    runnable = [
        e
        for e in entries
        if (residues_dir / f"{e.stem}.json").is_file()
        and (triaged_dir / f"{e.stem}.json").is_file()
        and (pdb_dir / e.filename).is_file()
    ]
    # Nothing runnable means nothing to score, and loading the checkpoint
    # would buy a model no antibody uses.
    model = load_model(str(weights_path)) if runnable else None

    def one(entry: pdb_index.Entry) -> str:
        return process_one(
            model,
            str(pdb_dir / entry.filename),
            str(residues_dir / f"{entry.stem}.json"),
            str(out_dir / f"{entry.stem}.tsv"),
            args.sapiens_weights,
            str(out_dir / f"{entry.stem}{sapiens_prior.PRIOR_SUFFIX}"),
        )

    # `error_reason` is passed here and nowhere else: AntiFold swallows its
    # own exceptions and exits 0, so this loop is the only place a backend
    # fault can still be attributed to the antibody that caused it.
    return batch.run(runnable, one, args.out_skip, error_reason="backend-failed")


if __name__ == "__main__":
    sys.exit(main())
