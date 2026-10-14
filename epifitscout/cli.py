"""The ``epifitscout`` command.

A single entry point with one subcommand per task, so the common path is

    epifitscout check
    epifitscout build --structure complex.pdb --cdr H 93 102 --antigen-chain B ...

rather than hunting for a script in the source tree. Each subcommand delegates
to the module that implements it; the modules keep their own ``main`` so they
remain runnable directly with ``python -m``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

#: Subcommand name -> (summary, importable module path). Imported lazily: a
#: subcommand's dependencies should not be a cost for every invocation, and
#: ``--help`` must work even if an optional one is missing.
_COMMANDS: dict[str, tuple[str, str]] = {
    "check": ("Verify the setup: MASTER, the database, and the optional packer.",
              "epifitscout.tools.setup_check"),
    "build": ("Search, then write grafted antibody-antigen complexes.",
              "epifitscout.tools.build_complex"),
    "view": ("Export a PyMOL session showing where a fragment fits.",
             "epifitscout.tools.visualize_fit"),
}


def _usage() -> str:
    width = max(len(name) for name in _COMMANDS)
    lines = [
        "usage: epifitscout <command> [options]",
        "",
        "commands:",
        *(f"  {name:<{width}}  {summary}" for name, (summary, _) in _COMMANDS.items()),
        "",
        "Run 'epifitscout <command> --help' for a command's own options.",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    if args[0] in ("-V", "--version"):
        from epifitscout import __version__

        print(f"epifitscout {__version__}")
        return 0

    command, rest = args[0], args[1:]
    if command not in _COMMANDS:
        close = [n for n in _COMMANDS if n.startswith(command)]
        hint = f" Did you mean '{close[0]}'?" if len(close) == 1 else ""
        print(f"epifitscout: unknown command '{command}'.{hint}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    from importlib import import_module

    module = import_module(_COMMANDS[command][1])
    # Every tool's main takes an argv sequence, so dispatch is a plain call.
    entry: Callable[[list[str]], int] = module.main
    return entry(rest)


if __name__ == "__main__":
    sys.exit(main())
