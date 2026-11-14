"""
Draw one animated diagram per chunker from the chunkers' real output.

Usage (from the repo root, with the tokens extra installed):
    python tools/diagrams/make_chunk_diagrams.py            # write the SVGs
    python tools/diagrams/make_chunk_diagrams.py --check    # fail if any SVG is out of date
    python tools/diagrams/make_chunk_diagrams.py word page  # only some diagrams
"""

import sys

from render import chunk_segments, render
from specs import ROOT, SPECS

OUT = ROOT / "docs" / "assets" / "images" / "chunkers"


def check_spec(name, spec):
    """Make sure the chunks drawn are exactly what the chunker returns."""
    drawn = []
    for k in range(len(spec["groups"])):
        text = "".join(piece for piece, _ in chunk_segments(spec, k))
        drawn.append(text.replace("↵", "\n").replace("⇟", "\f"))

    if drawn != spec["chunks"]:
        raise SystemExit(f"{name}: drawn chunks differ from the chunker\n"
                         f"  drawn: {drawn}\n  real:  {spec['chunks']}")


def main(args):
    check = "--check" in args
    names = [arg for arg in args if not arg.startswith("--")] or list(SPECS)
    OUT.mkdir(parents=True, exist_ok=True)
    stale = []

    for name in names:
        spec = SPECS[name]()
        check_spec(name, spec)
        svg = render(spec)
        path = OUT / f"{name}-chunker.svg"

        if not check:
            path.write_text(svg, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)} ({len(spec['groups'])} chunks)")
        elif not path.exists() or path.read_text(encoding="utf-8") != svg:
            stale.append(name)

    if stale:
        raise SystemExit(f"out of date: {', '.join(stale)}. Run: python {sys.argv[0]}")
    if check:
        print(f"OK: {len(names)} diagram(s) match the chunkers")


if __name__ == "__main__":
    main(sys.argv[1:])
