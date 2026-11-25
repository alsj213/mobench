"""Flame graph rendering from folded stacks (self-contained inline SVG).

Pure functions so they are unit-testable; the produced HTML is a single
standalone file (theme-aware, matching mobench chart style).
"""
from __future__ import annotations

import html
import hashlib

from collections import Counter

ROW_H = 16
MIN_W = 0.6           # skip rects narrower than this (px) to bound output size
MIN_FRACTION = 0.0005  # nodes below this share of samples are not drawn


class Node:
    __slots__ = ("name", "weight", "children")

    def __init__(self, name):
        self.name = name
        self.weight = 0        # leaf samples landing on this exact frame
        self.children = []

    def total(self) -> int:
        return self.weight + sum(c.total() for c in self.children)


def build_tree(folded: Counter) -> list:
    """Turn root-first folded paths into a forest; roots are the outermost frames."""
    forest = []

    def _find(node_list, name):
        for n in node_list:
            if n.name == name:
                return n
        return None

    for path, count in folded.items():
        frames = path.split(";")
        level = forest
        for i, name in enumerate(frames):
            node = _find(level, name)
            if node is None:
                node = Node(name)
                level.append(node)
            if i == len(frames) - 1:
                node.weight += count
            level = node.children
    return forest


def _palette(name: str) -> str:
    """Stable colour per function across light/dark (hue from name hash)."""
    h = int(hashlib.md5(name.encode()).hexdigest()[:6], 16)
    hue = h % 360
    return f"hsl({hue} 55% 45%)"


def render_flame_svg(forest: list, width: int = 1600, height=None,
                     max_depth: int = 90) -> str:
    """Render a folded forest as an SVG flame graph.

    Bottom row is the root frames (widest); each level stacks callers upward.
    Returns the inner SVG markup (rects + labels), not a full document.
    """
    total = sum(n.total() for n in forest) or 1
    depth_limit = min(max_depth, int(height / ROW_H) if height else max_depth)
    rects = []
    text = []

    def walk(nodes, depth, x0, x1):
        if depth >= depth_limit:
            return
        for node in nodes:
            wfrac = node.total() / total
            wpx = wfrac * width
            if wpx < MIN_W or wfrac < MIN_FRACTION:
                continue
            y = depth * ROW_H
            rects.append(
                f'<rect x="{x0:.2f}" y="{y}" width="{wpx:.2f}" '
                f'height="{ROW_H - 2}" rx="2" fill="{_palette(node.name)}" '
                f'opacity="0.9"><title>{html.escape(node.name)}'
                f' {wfrac * 100:.2f}%</title></rect>')
            label = node.name.split("(")[0]
            if wpx > 30 and label:
                text.append(
                    f'<text x="{x0 + 3:.2f}" y="{y + 12:.2f}" font-size="10" '
                    f'fill="#fff">{html.escape(label[:60])}</text>')
            walk(node.children, depth + 1, x0, x0 + wpx)
            x0 += wpx

    walk(forest, 0, 0.0, float(width))
    return (f'<svg viewBox="0 0 {width} {depth_limit * ROW_H + 4}" '
            f'xmlns="http://www.w3.org/2000/svg" style="max-width:100%">'
            + "".join(rects) + "".join(text) + "</svg>")


def flame_html(svg: str, title: str, notes: str = "") -> str:
    """Wrap flame SVG into a standalone themed HTML document."""
    return f"""<!doctype html><html><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
:root{{--bg:#eef1f6;--ink:#17202f;--mut:#5a6577}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0d1320;--ink:#e9eef7;--mut:#94a1b8}}}}
:root[data-theme="dark"]{{--bg:#0d1320;--ink:#e9eef7;--mut:#94a1b8}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC",sans-serif}}
.wrap{{max-width:1200px;margin:0 auto;padding:28px 20px 60px}}
h1{{font-size:22px;margin:0 0 4px}} p{{color:var(--mut);margin:2px 0 14px}}
</style>
<div class="wrap"><h1>{html.escape(title)}</h1><p>{notes}</p>{svg}</div>
</html>"""
