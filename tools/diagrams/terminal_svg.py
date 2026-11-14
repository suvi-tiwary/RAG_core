"""Draw an animated terminal SVG in the style of docs/assets/images/cli.svg.

A demo is a list of (command, output_lines) pairs. Each command is "typed",
then its output lines fade in one by one; the whole terminal loops.
A command starting with "#" is a step title: typed in blue, never run.
"""

from xml.sax.saxutils import escape

WIDTH = 960
LEFT = 44
CHAR_W = 8.45      # width of one 14px monospace character
MAX_CHARS = 100    # characters that fit on one terminal line
LINE_H = 24
GROUP_GAP = 14     # extra space before every new command
TYPE_SECONDS = 1.6
LINE_SECONDS = 0.45
PAUSE_SECONDS = 0.9   # after typing, before the output appears
HOLD_SECONDS = 4.0    # how long the finished terminal stays on screen

STYLE = """
.t { font: 14px ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre; }
.bar { font: 600 12px system-ui, -apple-system, "Segoe UI", sans-serif;
       fill: #94a3b8; text-anchor: middle; }
@media (prefers-reduced-motion: reduce) {
  .t { animation: none !important; opacity: 1; }
  .cover { display: none; }
}
"""


def shorten(line):
    """Fit a long line on screen; a long list keeps its first items and counts the rest."""
    if len(line) <= MAX_CHARS:
        return line
    head, sep, rest = line.partition(": ")
    items = rest.split(", ")
    if not sep or len(items) < 3:
        return line[:MAX_CHARS - 1] + "…"
    kept = []
    for item in items:
        tail = f", … (+{len(items) - len(kept) - 1} more)"
        if len(head + sep + ", ".join(kept + [item]) + tail) > MAX_CHARS:
            break
        kept.append(item)
    return f"{head}{sep}{', '.join(kept)}, … (+{len(items) - len(kept)} more)"


def line_color(line):
    """Pick a colour so the important numbers stand out."""
    if line.startswith("Overall score:"):
        score = float(line.split(":")[1].split("/")[0])
        return "#86efac" if score >= 80 else "#fdba74"
    if line.startswith("Ready:"):
        return "#86efac" if line.endswith("True") else "#fca5a5"
    if line.startswith("First failing stage:"):
        return "#86efac" if line.endswith("none") else "#fdba74"
    if line.startswith("Regressions") or line.strip() == "1":
        return "#fca5a5"
    if "not measured" in line or "Provisional" in line:
        return "#fde68a"
    return "#94a3b8"


def split_command(command):
    """Cut a long command into shell lines ending in a backslash, as a person would type it."""
    pieces, current = [], ""
    for word in command.split(" "):
        if current and len(current) + len(word) + 4 > MAX_CHARS:
            pieces.append(current + " \\")
            current = "  " + word
        else:
            current = f"{current} {word}" if current else word
    return pieces + [current]


def timeline(demo):
    """Return the rows to draw: (kind, text, y, start_second), plus the loop length."""
    rows, y, now, after_note = [], 52, 0.4, False
    for command, output in demo:
        if command.startswith("#"):   # a step title, typed like a shell comment
            y += GROUP_GAP + LINE_H
            rows.append(("note", command, y, now))
            now += TYPE_SECONDS
            after_note = True
            continue
        y += 0 if after_note else GROUP_GAP
        after_note = False
        for index, piece in enumerate(split_command(command)):
            y += LINE_H
            rows.append(("cmd" if index == 0 else "more", piece, y, now))
            now += TYPE_SECONDS
        now += PAUSE_SECONDS
        for line in output:
            y += LINE_H
            rows.append(("out", shorten(line), y, now))
            now += LINE_SECONDS
        now += 0.6
    return rows, y + 34, now + HOLD_SECONDS


def keyframes(name, start, total):
    """CSS keyframes: hidden until start, shown until near the end, then fade out."""
    begin = 100 * start / total
    shown = 100 * (start + 0.2) / total
    return (f"@keyframes {name} {{ 0%, {begin:.2f}% {{ opacity: 0; }} "
            f"{shown:.2f}%, 94% {{ opacity: 1; }} 100% {{ opacity: 0; }} }}\n"
            f".{name} {{ animation: {name} {total:.1f}s infinite; }}\n")


def typing(name, start, total, width, chars):
    """CSS for a cover box that slides off a command, one character per step."""
    begin = 100 * start / total
    done = 100 * (start + TYPE_SECONDS) / total
    return (f"@keyframes {name} {{ 0%, {begin:.2f}% {{ transform: translateX(0); }} "
            f"{done:.2f}%, 100% {{ transform: translateX({width:.0f}px); }} }}\n"
            f".{name} {{ animation: {name} {total:.1f}s infinite steps({chars}, end); }}\n")


def draw_row(index, row, total):
    """Return (css, svg) for one terminal row."""
    kind, text, y, start = row
    name = f"l{index}"
    css = keyframes(name, start, total)
    if kind == "out":
        svg = (f'<text class="t {name}" x="{LEFT}" y="{y}" fill="{line_color(text)}">'
               f'{escape(text)}</text>')
        return css, svg
    width = len(text) * CHAR_W + 12
    prompt = {"cmd": '<tspan fill="#f472b6">$ </tspan>', "more": "  ", "note": ""}[kind]
    x = LEFT if kind == "note" else LEFT + 2 * CHAR_W
    color = "#7dd3fc" if kind == "note" else "#e2e8f0"
    css += typing(f"c{index}", start, total, width, len(text))
    svg = (f'<text class="t {name}" x="{LEFT}" y="{y}" fill="{color}">{prompt}'
           f'{escape(text)}</text>\n'
           f'<g clip-path="url(#screen)"><rect class="cover c{index}" x="{x:.1f}" y="{y - 17}" '
           f'width="{width:.0f}" height="{LINE_H}" fill="#0f172a"/></g>')
    return css, svg


def render(title, description, demo):
    """Return the full SVG text for a demo."""
    rows, height, total = timeline(demo)
    parts = [draw_row(i, row, total) for i, row in enumerate(rows)]
    css = STYLE + "".join(css for css, _ in parts)
    body = "\n".join(f"  {svg}" for _, svg in parts)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}"
     width="{WIDTH}" height="{height}" role="img" aria-labelledby="title desc">
  <title id="title">{escape(title)}</title>
  <desc id="desc">{escape(description)}</desc>
  <defs>
    <style>{css}    </style>
    <clipPath id="screen"><rect x="0" y="0" width="{WIDTH}" height="{height}"/></clipPath>
  </defs>
  <rect x="0.5" y="0.5" width="{WIDTH - 1}" height="{height - 1}" rx="14"
        fill="#0f172a" stroke="#1e293b"/>
  <rect x="0.5" y="0.5" width="{WIDTH - 1}" height="40" rx="14" fill="#1e293b"/>
  <rect x="0.5" y="26" width="{WIDTH - 1}" height="15" fill="#1e293b"/>
  <circle cx="26" cy="21" r="6" fill="#f87171"/>
  <circle cx="46" cy="21" r="6" fill="#fbbf24"/>
  <circle cx="66" cy="21" r="6" fill="#4ade80"/>
  <text class="bar" x="{WIDTH // 2}" y="25">rag-titan — terminal</text>
{body}
</svg>
"""
