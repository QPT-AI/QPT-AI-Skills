#!/usr/bin/env python3
"""Report which installed module exports each Torx symbol, with its signature.

The Torx docs confirm `torx.psc` for gates, circuits and simulators, but do not
print the module path for the factor-graph core (Site, DFG, TiledFactor, ...),
draw_circuit, or AbstractSampler. Run this before importing those.

Usage:
    python scripts/locate_symbols.py Site DFG TiledFactor ChainFactor
    python scripts/locate_symbols.py --list            # every public name per module
    python scripts/locate_symbols.py --package other_pkg Name   # any package

For each name it prints the recommended import (the shallowest public module that
exports it), every other module that exports it, the defining module, and the call
signature when one is available. Unknown names get close-match suggestions.
"""

from __future__ import annotations

import argparse
import difflib
import importlib
import inspect
import pkgutil
import sys

DEFAULT_NAMES = [
    "AbstractFactor", "AbstractReferenceFactor", "AbstractHasLogProbability",
    "AbstractMatrixFactor", "Site", "DFG", "DFGInfo", "TiledFactor", "ChainFactor",
    "DeterministicFactor", "draw_circuit", "AbstractSampler", "JaxPRNGSampler",
    "AbstractContinuousGate", "AbstractAffineGaussianGate",
]


def iter_modules(package_name: str):
    """Yield (name, module) for the package and every importable submodule."""
    package = importlib.import_module(package_name)
    yield package_name, package
    if not hasattr(package, "__path__"):
        return
    for info in pkgutil.walk_packages(package.__path__, prefix=package_name + "."):
        try:
            yield info.name, importlib.import_module(info.name)
        except Exception as exc:  # optional deps, test modules, etc.
            print(f"# skipped {info.name}: {type(exc).__name__}: {exc}", file=sys.stderr)


def exported_names(module) -> set[str]:
    names = getattr(module, "__all__", None)
    if names is not None:
        return set(names)
    return {n for n in vars(module) if not n.startswith("_")}


def is_public_path(module_name: str) -> bool:
    return not any(part.startswith("_") for part in module_name.split("."))


def describe(obj) -> str:
    try:
        return f"{getattr(obj, '__name__', '?')}{inspect.signature(obj)}"
    except (TypeError, ValueError):
        return f"{getattr(obj, '__name__', repr(obj))} (no signature available)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="symbols to locate (default: common torx core names)")
    parser.add_argument("--package", default="torx", help="package to search (default: torx)")
    parser.add_argument("--list", action="store_true", help="list public names per module and exit")
    args = parser.parse_args()

    try:
        modules = list(iter_modules(args.package))
    except ImportError as exc:
        print(f"Cannot import {args.package!r}: {exc}. Is it installed in this environment?")
        return 2

    if args.list:
        for name, module in modules:
            exported = sorted(exported_names(module))
            if exported:
                print(f"{name}: {', '.join(exported)}")
        return 0

    wanted = args.names or DEFAULT_NAMES
    universe: set[str] = set()
    for _, module in modules:
        universe |= exported_names(module)

    missing = 0
    for symbol in wanted:
        hits = [(name, module) for name, module in modules if symbol in exported_names(module)]
        if not hits:
            missing += 1
            close = difflib.get_close_matches(symbol, sorted(universe), n=5)
            hint = f" Close matches: {', '.join(close)}." if close else ""
            print(f"{symbol}: NOT FOUND in {args.package}.{hint}\n")
            continue
        public = [h for h in hits if is_public_path(h[0])] or hits
        public.sort(key=lambda h: (h[0].count("."), h[0]))
        best_name, best_module = public[0]
        obj = getattr(best_module, symbol)
        print(f"from {best_name} import {symbol}")
        others = [name for name, _ in hits if name != best_name]
        if others:
            print(f"  also exported by: {', '.join(sorted(others))}")
        print(f"  defined in: {getattr(obj, '__module__', '?')}")
        print(f"  signature: {describe(obj)}\n")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
