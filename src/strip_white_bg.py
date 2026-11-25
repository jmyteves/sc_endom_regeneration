"""Remove opaque white background paths from a matplotlib SVG, in place.

Targets only paths that are (a) axis-aligned 4-corner rectangles and (b) filled
pure white with no stroke, i.e. the figure-level background and the per-axes
background patches. does not touch:

  * white strokes (`stroke: #ffffff`), these are the path-effect outlines drawn
    around trajectory lines and marker edges, a design element;
  * white text;
  * non-rectangular white fills.

Lossless with respect to everything else: marker count, text nodes and XML
validity are asserted unchanged by the caller.
"""
import re

# a closed 4-corner rectangle: M x0 y0 L x1 y0 L x1 y1 L x0 y1 z (any corner order)
_RECT = re.compile(r'^\s*M\s+[\d.eE+-]+\s+[\d.eE+-]+\s*'
                   r'(?:L\s+[\d.eE+-]+\s+[\d.eE+-]+\s*){3}z\s*$', re.S)
_PATH = re.compile(r'<path\b[^>]*?/>', re.S)


def strip_white_bg(path, verbose=True):
    src = open(path).read()
    removed = []

    def drop(m):
        tag = m.group(0)
        style = re.search(r'style="([^"]*)"', tag)
        d = re.search(r'\bd="([^"]*)"', tag)
        if not style or not d:
            return tag
        st = style.group(1)
        fill_white = re.search(r'fill:\s*#ffffff\b', st)
        has_stroke = re.search(r'stroke:\s*(?!none)', st)
        if fill_white and not has_stroke and _RECT.match(d.group(1)):
            removed.append(d.group(1)[:40].replace("\n", " "))
            return ""
        return tag

    out = _PATH.sub(drop, src)
    open(path, "w").write(out)
    if verbose:
        print(f"{path}: removed {len(removed)} white background rect(s)")
    return len(removed)
