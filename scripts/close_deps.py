#!/usr/bin/env python3
"""Satisfy the dependencies that ``pip install --no-deps`` left behind.

Why this is needed
------------------
SGLang must be installed with ``--no-deps`` (see filter_deps.py): its
dependency set cannot be resolved against the ROCm base image at all.
``compressed-tensors==0.18.0`` alone requires ``torch>=2.10.0``, and SGLang
itself pins ``torch==2.13.0``, so asking pip to resolve the tree ends in
ResolutionImpossible -- and the only way to make it resolvable is to let pip
install the torch it asks for, which is a CUDA build. ``--no-deps`` is
load-bearing, not stylistic.

The catch is that ``--no-deps`` makes the transitive closure somebody's
problem, and for a long time that somebody was a hand-written list in the
Dockerfile. It drifted, as hand-written lists do: the published image could
not run ``sglang serve`` at all, because starlette, traitlets, multidict,
attrs, yarl and pydantic-core were all missing. Adding names one at a time
only moves the failure -- repair the venv enough to import ``transformers``
and the next build dies on ``regex`` instead.

So compute the closure instead of guessing it: walk the installed
distributions' own metadata, find every requirement that is not satisfied,
install it, and repeat to a fixed point.

Requirements come in two flavours, and the difference is the whole point:

* **Genuinely unsatisfiable.** ``torch==2.13.0`` against a ROCm 2.6.0 base,
  and every CUDA package filter_deps.py dropped. These can never be
  satisfied, and "fixing" one is exactly how a CUDA torch replaces the ROCm
  one. Anything naming a package in filter_deps.EXCLUDE is reported and left
  alone, permanently.

* **Merely unsatisfied.** ``pydantic`` pins ``pydantic-core==2.46.4``; a
  --no-deps install grabs the newest instead, and pydantic then refuses to
  import at all. These are ordinary packages with ordinary versions, and the
  fix is to honour the specifier that was thrown away.

Because of the first flavour, a clean ``pip check`` is not the goal here and
never will be. The goal is that nothing *outside* the known-unsatisfiable set
is left unsatisfied.

Requirements are read from metadata rather than parsed out of ``pip check``
output because ``pip check`` reports a missing dependency by name only. That
is what produced the pydantic-core breakage in the first place: install by
bare name and you get whatever is newest.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import subprocess
import sys

from packaging.requirements import Requirement

from filter_deps import EXCLUDE, EXCLUDE_DIFFUSION, requirement_name


def unsatisfied(banned: set[str]):
    """Yield (requiring project, Requirement, installed version or None).

    Requirements gated behind an extra are skipped: nothing records which
    extras were asked for, and the ones that matter here (the diffusion
    extras) were installed explicitly by name, so their own requirements
    show up ungated on those distributions.
    """
    for dist in md.distributions():
        name = dist.metadata["Name"] or "?"
        for spec in dist.requires or []:
            req = Requirement(spec)
            if req.marker is not None:
                # extra-gated, or not applicable to this platform/python
                if "extra" in str(req.marker) or not req.marker.evaluate():
                    continue
            if requirement_name(req.name) in banned:
                yield name, req, "excluded"
                continue
            try:
                have = md.version(req.name)
            except md.PackageNotFoundError:
                yield name, req, None
            else:
                if req.specifier and not req.specifier.contains(
                        have, prereleases=True):
                    yield name, req, have


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-rounds", type=int, default=15,
                    help="Give up rather than loop forever if two packages "
                         "pin incompatible versions of a third")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what is unsatisfied without installing it")
    args = ap.parse_args(argv)

    banned = EXCLUDE | EXCLUDE_DIFFUSION
    added: list[str] = []
    excluded: set[str] = set()
    previous: set[str] | None = None

    for round_no in range(1, args.max_rounds + 1):
        missing: dict[str, str] = {}   # package -> requirement to install
        repinned: list[str] = []       # for the log

        for who, req, have in unsatisfied(banned):
            if have == "excluded":
                excluded.add(f"{req.name} (wanted by {who})")
                continue
            # Keep the specifier. Installing by bare name is what put an
            # incompatible pydantic-core in the venv.
            missing[req.name] = str(req)
            if have is not None:
                repinned.append(f"{req.name} {have} -> {req.specifier} ({who})")

        if not missing:
            print(f"dependency closure complete after {round_no - 1} round(s); "
                  f"{len(added)} package(s) installed or re-pinned")
            for line in sorted(excluded):
                print(f"  unsatisfiable by design, left alone: {line}")
            return 0

        specs = sorted(missing.values())
        if specs and set(specs) == previous:
            print(f"no progress: {', '.join(specs)}\n"
                  f"Two packages most likely pin incompatible versions of a "
                  f"third; this needs a human.", file=sys.stderr)
            return 1
        previous = set(specs)

        print(f"round {round_no}: {len(specs)} unsatisfied -> "
              f"{', '.join(specs)}", flush=True)
        for line in repinned:
            print(f"    re-pinning {line}")
        if args.dry_run:
            return 0

        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "-q", *specs])
        if proc.returncode != 0:
            print(f"pip install failed for: {', '.join(specs)}", file=sys.stderr)
            return proc.returncode
        added += specs

    print(f"closure did not converge in {args.max_rounds} rounds", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
