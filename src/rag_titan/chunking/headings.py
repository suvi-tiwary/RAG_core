"""
Markdown heading rules, shared by HeadingChunker and the chunk locations.

The DOCX loader writes Word headings as Markdown too, so this works for
.md, .txt and .docx files.
"""

import re

# Markdown headings "# Title" to "###### Title"
HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
# ``` or ~~~ starts and ends a code block; "#" lines inside it are code, not headings
FENCE = re.compile(r"^\s*(```|~~~)")


def heading_marks(text):
    """
    Find every heading and the heading path it opens.

    Returns:
        List of (position, path): where the heading line starts in `text`, and
        the titles from the top level down to it, for example
        (120, ["Leave policy", "Sick leave"]).
    """
    marks, levels, in_code, position = [], [], False, 0

    for line in text.splitlines(keepends=True):
        if FENCE.match(line):
            in_code = not in_code
        match = None if in_code else HEADING.match(line.rstrip("\r\n"))

        if match:
            level = len(match.group(1))
            # a heading closes every open heading at its level or deeper
            levels = [(lvl, title) for lvl, title in levels if lvl < level]
            levels.append((level, match.group(2).strip()))
            marks.append((position, [title for _, title in levels]))
        position += len(line)

    return marks
