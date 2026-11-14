"""Drawing helpers for the chunker diagrams, in the house style of docs/assets/images."""

import textwrap
from xml.sax.saxutils import escape

WIDTH = 960
LEFT, RIGHT = 40, 920
CHAR_W = 7.3   # width of one 12px monospace character (a little more than real, so text fits)
LINE_H = 16
CARD_CHARS = 112   # characters that fit on one line of a chunk card

STYLE = """
.bg { fill: #ffffff; stroke: #e2e8f0; }
.title { font: 700 22px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #0f172a; }
.sub { font: 400 13px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #475569; }
.h { font: 700 13px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #334155; }
.unit { fill: #eff6ff; stroke: #93c5fd; stroke-width: 1; }
.mono { font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; fill: #0f172a;
        white-space: pre; }
.pick { fill: none; stroke: #f97316; stroke-width: 2.5; }
.shared { fill: rgba(249, 115, 22, 0.28); stroke: #f97316; stroke-width: 2.5; }
.card { fill: #f8fafc; stroke: #cbd5e1; stroke-width: 1; }
.tag { font: 700 12px ui-monospace, SFMono-Regular, Menlo, monospace; fill: #64748b; }
.ov { fill: #ea580c; font-weight: 700; }
.foot { font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; fill: #0f172a; }
"""


def wrap(label, max_chars):
    """Cut a long label into lines of at most max_chars, preferring spaces."""
    if len(label) <= max_chars:
        return [label]
    return textwrap.wrap(label, max_chars, break_long_words=True, replace_whitespace=False,
                         drop_whitespace=False) or [label]


def flow(labels, top, max_chars=40, pad=8, gap=6):
    """
    Place labels left to right in rows, like words in a paragraph.

    Returns:
        (boxes, bottom): one (x, y, width, height, lines) per label, and the
        y just below the last row.
    """
    boxes, x, y, row_h = [], LEFT, top, 0

    for label in labels:
        lines = wrap(label, max_chars)
        w = max(len(line) for line in lines) * CHAR_W + 2 * pad
        h = len(lines) * LINE_H + 10
        if x + w > RIGHT and x > LEFT:
            x, y, row_h = LEFT, y + row_h + gap, 0
        boxes.append((x, y, w, h, lines))
        x, row_h = x + w + gap, max(row_h, h)

    return boxes, y + row_h


def unit_boxes(boxes, pad=8):
    """The light blue boxes and their text."""
    out = []
    for x, y, w, h, lines in boxes:
        out.append(f'<rect class="unit" x="{x:.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="5"/>')
        for i, line in enumerate(lines):
            out.append(f'<text class="mono" x="{x + pad:.1f}" y="{y + 18 + i * LINE_H}">'
                       f"{escape(line)}</text>")
    return "\n".join(out)


def outline(boxes, indexes, css_class):
    """Orange frames around the boxes at `indexes`."""
    return "\n".join(
        f'<rect class="{css_class}" x="{x - 2:.1f}" y="{y - 2}" width="{w + 4:.1f}" '
        f'height="{h + 4}" rx="6"/>'
        for i, (x, y, w, h, _) in enumerate(boxes) if i in indexes
    )


def card(top, number, segments):
    """
    One chunk in the list: "C0" and its text, with shared parts in orange.

    Args:
        segments: list of (text, shared) pieces of the chunk, in order.

    Returns:
        (svg, height)
    """
    lines, line = [], []
    for text, shared in segments:
        for char in text:
            line.append((char, shared))
            if len(line) == CARD_CHARS:
                lines.append(line)
                line = []
    lines.append(line)

    height = len(lines) * LINE_H + 14
    out = [f'<rect class="card" x="{LEFT}" y="{top}" width="{RIGHT - LEFT}" '
           f'height="{height}" rx="8"/>',
           f'<text class="tag" x="{LEFT + 10}" y="{top + 21}">C{number}</text>']
    for i, chars in enumerate(lines):
        out.append(f'<text class="mono" x="{LEFT + 44}" y="{top + 21 + i * LINE_H}">'
                   f"{tspans(chars)}</text>")
    return "\n".join(out), height


def tspans(chars):
    """Group (char, shared) pairs into plain text and orange <tspan>s."""
    out, run, run_shared = [], "", None
    for char, shared in chars + [(None, None)]:
        if shared != run_shared and run:
            out.append(f'<tspan class="ov">{escape(run)}</tspan>' if run_shared else escape(run))
            run = ""
        if char is not None:
            run, run_shared = run + char, shared
    return "".join(out)
