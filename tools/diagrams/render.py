"""Turn one diagram spec into an animated SVG: the chunker runs one chunk at a time."""

from xml.sax.saxutils import escape

from svg_parts import LEFT, STYLE, WIDTH, card, flow, outline, unit_boxes

STEP = 1.6    # seconds each chunk stays highlighted
PAUSE = 2.0   # seconds with every chunk shown, before the loop starts again


def keyframes(name, start, end, total, before, during, after):
    """CSS keyframes: `before` until start, `during` from start to end, then `after`."""
    a, b = 100 * start / total, 100 * end / total
    return (f"@keyframes {name} {{ 0%, {a:.2f}% {{ opacity: {before}; }} "
            f"{a + 0.5:.2f}%, {b - 0.5:.2f}% {{ opacity: {during}; }} "
            f"{b:.2f}%, 100% {{ opacity: {after}; }} }}")


def animation_css(count):
    """Highlight h<k> shows only during step k; card c<k> appears at step k and stays."""
    total = count * STEP + PAUSE
    rules = []
    for k in range(count):
        start, end = k * STEP, (k + 1) * STEP
        rules.append(keyframes(f"h{k}", start, end, total, 0, 1, 0))
        rules.append(keyframes(f"c{k}", start, total - 0.2, total, 0.15, 1, 0.15))
        rules.append(f".h{k} {{ opacity: 0; animation: h{k} {total:.1f}s infinite; }}")
        rules.append(f".c{k} {{ opacity: 0.15; animation: c{k} {total:.1f}s infinite; }}")
    hidden = ", ".join(f".h{k}" for k in range(count))
    shown = ", ".join(f".c{k}" for k in range(count))
    rules.append("@media (prefers-reduced-motion: reduce) { "
                 f"{hidden} {{ animation: none; opacity: 0; }} "
                 f"{shown} {{ animation: none; opacity: 1; }} }}")
    return "\n".join(rules)


def chunk_segments(spec, k):
    """The pieces of chunk k as (text, shared) pairs; shared = repeated from chunk k-1."""
    if "cards" in spec:   # chunkers that add text (a heading path): show their exact output
        return trim([(spec["cards"][k], False)], 0, False)
    group, previous = spec["groups"][k], set(spec["groups"][k - 1]) if k else set()
    parts = [(spec["units"][i], i in previous) for i in group]
    text = spec["joiner"].join(unit for unit, _ in parts)
    lead = len(text) - len(text.lstrip()) if spec["strip"] else 0
    segments = [((spec["joiner"] if index else "") + unit, shared)
                for index, (unit, shared) in enumerate(parts)]
    return trim(segments, lead, spec["strip"])


def trim(segments, lead, strip):
    """Drop `lead` leading characters (and trailing spaces when strip), keep the rest."""
    text = "".join(piece for piece, _ in segments)
    end = len(text.rstrip()) if strip else len(text)
    out, pos = [], 0
    for piece, shared in segments:
        keep = piece[max(0, lead - pos):max(0, end - pos)]
        if keep:
            out.append((keep.replace("\n", "↵").replace("\f", "⇟"), shared))
        pos += len(piece)
    return out


def render(spec):
    """Return the whole SVG for one spec."""
    labels = [spec["show"](unit) for unit in spec["units"]]
    boxes, bottom = flow(labels, 128, spec.get("max_chars", 40), spec.get("pad", 8))
    body = [f'<text class="h" x="{LEFT}" y="114">1. {escape(spec["units_label"])}</text>',
            unit_boxes(boxes, spec.get("pad", 8))]

    for k, group in enumerate(spec["groups"]):
        previous = set(spec["groups"][k - 1]) if k else set()
        body.append(f'<g class="h{k}">{outline(boxes, set(group) - previous, "pick")}'
                    f'{outline(boxes, set(group) & previous, "shared")}</g>')

    y = bottom + 34
    body.append(f'<text class="h" x="{LEFT}" y="{y - 10}">2. The chunks (real output)</text>')
    for k in range(len(spec["groups"])):
        svg, height = card(y, k, chunk_segments(spec, k))
        body.append(f'<g class="c{k}">{svg}</g>')
        y += height + 6

    height = y + 40
    body.append(f'<text class="foot" x="{LEFT}" y="{y + 20}">{escape(spec["footer"])}</text>')
    return page(spec, "\n".join(body), height)


def page(spec, body, height):
    """Wrap the drawing in the <svg> element with title, styles and background."""
    svg_open = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" '
                f'width="{WIDTH}" height="{height}" role="img" aria-labelledby="title desc">')
    note = "made by tools/diagrams/make_chunk_diagrams.py from real chunker output; do not edit"
    legend = ("Orange frame = the chunk being made. "
              "Filled orange = overlap, repeated from the chunk before.")
    return f"""{svg_open}
<title id="title">{escape(spec["title"])}</title>
<desc id="desc">{escape(spec["desc"])}</desc>
<!-- {note} -->
<defs><style>{STYLE}
{animation_css(len(spec["groups"]))}
</style></defs>
<rect class="bg" x="1" y="1" width="{WIDTH - 2}" height="{height - 2}" rx="14"/>
<text class="title" x="{LEFT}" y="46">{escape(spec["title"])}</text>
<text class="sub" x="{LEFT}" y="70">{escape(spec["subtitle"])}</text>
<text class="sub" x="{LEFT}" y="90">{legend}</text>
{body}
</svg>
"""
