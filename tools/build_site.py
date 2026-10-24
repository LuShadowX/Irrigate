"""Build the GitHub Pages site in docs/ from the executed notebooks.

    python tools/build_site.py

docs/index.html      overview, results and the course map
docs/notebooks/*.html  every notebook with its outputs (nbconvert, lab template)
docs/figures/*.png     written by the notebooks themselves (plotting.save)
"""

import html
import re
import sys
from pathlib import Path

from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import site_content as content  # noqa: E402
import syllabus  # noqa: E402

DOCS = ROOT / "docs"
STYLE = """
:root { --ink: #1d2b36; --muted: #5b6b78; --line: #e3e8ec; --accent: %(accent)s; --soft: %(soft)s; --bg: #fbfcfc; }
* { box-sizing: border-box; }
body { margin: 0; color: var(--ink); background: var(--bg); font: 16px/1.6 Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }
a { color: var(--accent); }
header.top { border-bottom: 1px solid var(--line); background: #fff; }
.wrap { max-width: 1060px; margin: 0 auto; padding: 0 22px; }
nav.bar { display: flex; align-items: center; gap: 22px; height: 58px; font-size: 14.5px; }
nav.bar b { margin-right: auto; font-size: 17px; letter-spacing: -0.02em; }
nav.bar a { color: var(--ink); text-decoration: none; }
nav.bar a:hover { color: var(--accent); }
.hero { padding: 54px 0 30px; }
.hero .kicker { margin: 0; color: var(--accent); font: 600 12.5px ui-monospace, Menlo, monospace; letter-spacing: 0.12em; text-transform: uppercase; }
.hero h1 { margin: 10px 0 12px; font-size: 42px; line-height: 1.12; letter-spacing: -0.035em; }
.hero p { max-width: 760px; margin: 0 0 12px; color: var(--muted); font-size: 17.5px; }
.buttons { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
.btn { display: inline-block; padding: 9px 16px; color: #fff; background: var(--accent); border-radius: 8px; font-weight: 600; font-size: 14.5px; text-decoration: none; }
.btn.ghost { color: var(--ink); background: #fff; border: 1px solid var(--line); }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 26px 0 8px; }
.stat { padding: 16px; background: #fff; border: 1px solid var(--line); border-radius: 12px; }
.stat b { display: block; font-size: 26px; letter-spacing: -0.03em; }
.stat span { color: var(--muted); font-size: 13.5px; }
h2 { margin: 46px 0 14px; font-size: 26px; letter-spacing: -0.025em; }
.cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.card { display: block; padding: 16px 18px; color: inherit; text-decoration: none; background: #fff; border: 1px solid var(--line); border-radius: 12px; transition: border-color .15s, transform .15s; }
.card:hover { border-color: var(--accent); transform: translateY(-2px); }
.card small { color: var(--accent); font: 600 12px ui-monospace, Menlo, monospace; letter-spacing: 0.08em; }
.card h3 { margin: 6px 0 6px; font-size: 17px; }
.card p { margin: 0; color: var(--muted); font-size: 14px; }
.figs { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
figure { margin: 0; padding: 12px; background: #fff; border: 1px solid var(--line); border-radius: 12px; }
figure img { display: block; width: 100%%; border-radius: 6px; }
figcaption { margin-top: 8px; color: var(--muted); font-size: 14px; }
table { width: 100%%; border-collapse: collapse; font-size: 14.5px; background: #fff; }
th, td { padding: 8px 10px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--line); }
thead th { color: var(--muted); font: 600 12px ui-monospace, Menlo, monospace; letter-spacing: 0.06em; text-transform: uppercase; background: var(--soft); }
.unit { margin-top: 22px; }
.unit h3 { margin: 0 0 8px; font-size: 18px; }
.unit .tag { margin-left: 6px; padding: 2px 8px; color: var(--accent); background: var(--soft); border-radius: 99px; font: 600 12px ui-monospace, Menlo, monospace; }
pre.cmd { padding: 14px 16px; overflow-x: auto; color: #e6edf3; background: #1d2b36; border-radius: 10px; font: 13.5px/1.6 ui-monospace, Menlo, monospace; }
footer { margin: 60px 0 0; padding: 22px 0 40px; color: var(--muted); border-top: 1px solid var(--line); font-size: 14px; }
@media (max-width: 760px) { .stats, .cards, .figs { grid-template-columns: 1fr 1fr; } .hero h1 { font-size: 32px; } }
@media (max-width: 520px) { .stats, .cards, .figs { grid-template-columns: 1fr; } nav.bar a.hide-sm { display: none; } }
"""


def nav(prefix=""):
    links = "".join(f'<a class="hide-sm" href="{prefix}{href}">{label}</a>' for label, href in [("Notebooks", "index.html#notebooks"), ("Results", "index.html#results"), ("Course map", "index.html#map")])
    return (f'<header class="top"><div class="wrap"><nav class="bar"><b>{content.NAME}</b>{links}'
            f'<a href="{content.REPO}">GitHub</a></nav></div></header>')


def page(title, body, prefix=""):
    style = STYLE % {"accent": content.ACCENT, "soft": content.SOFT}
    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{html.escape(title)}</title><meta name="description" content="{html.escape(content.TAGLINE)}">'
            f'<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">'
            f'<style>{style}</style></head><body>{nav(prefix)}<main class="wrap">{body}</main>'
            f'<footer class="wrap">{content.FOOTER}</footer></body></html>')


def index():
    stats = "".join(f'<div class="stat"><b>{v}</b><span>{k}</span></div>' for v, k in content.STATS)
    cards = "".join(
        f'<a class="card" href="notebooks/{nb}.html"><small>{nb[:2]} · UNIT {unit}</small><h3>{html.escape(syllabus.NOTEBOOKS[nb])}</h3>'
        f'<p>{len(topics)} syllabus topics</p></a>'
        for unit, _, nb, topics in syllabus.UNITS)
    figs = "".join(f'<figure><img src="figures/{f}.png" alt="" loading="lazy"><figcaption>{c}</figcaption></figure>' for f, c in content.FIGURES)
    units = ""
    for unit, title, nb, topics in syllabus.UNITS:
        rows = "".join(f'<tr><td>{html.escape(t)}</td><td><a href="notebooks/{nb}.html">{nb[:2]}</a> § {html.escape(sec)}</td></tr>' for t, sec in topics)
        units += f'<div class="unit"><h3>Unit {unit}: {html.escape(title)}<span class="tag">{len(topics)} topics</span></h3><table><thead><tr><th>Syllabus topic</th><th>Where</th></tr></thead><tbody>{rows}</tbody></table></div>'
    outcomes = ""
    if syllabus.OUTCOMES:
        rows = "".join(f'<tr><td><b>{c}</b></td><td>{html.escape(t)}</td><td><a href="notebooks/{next(nb for nb in syllabus.NOTEBOOKS if nb.startswith(n))}.html">notebook {n}</a></td></tr>' for c, t, n in syllabus.OUTCOMES)
        outcomes = f'<h3 style="margin-top:18px">{content.OUTCOME_TITLE}</h3><table><thead><tr><th></th><th>Students learn to …</th><th>Where</th></tr></thead><tbody>{rows}</tbody></table>'
    total = sum(len(t) for *_, t in syllabus.UNITS)
    body = (f'<section class="hero"><p class="kicker">{syllabus.COURSE}</p><h1>{content.HEADLINE}</h1>'
            + "".join(f"<p>{p}</p>" for p in content.INTRO)
            + f'<div class="buttons"><a class="btn" href="notebooks/{next(iter(syllabus.NOTEBOOKS))}.html">Start with notebook 01</a>'
              f'<a class="btn ghost" href="{content.REPO}">Source on GitHub</a><a class="btn ghost" href="#map">Course map</a></div>'
              f'<div class="stats">{stats}</div></section>'
              f'<h2 id="notebooks">Notebooks</h2><div class="cards">{cards}</div>'
              f'<h2 id="results">Results</h2><div class="figs">{figs}</div>'
              f'<h2>Run it yourself</h2><pre class="cmd">{html.escape(content.RUN)}</pre>'
              f'<h2 id="map">Course map</h2><p>All {total} topics of the {syllabus.COURSE.split(" · ")[0]} syllabus, unit by unit, with the notebook section that covers each one.</p>'
              f'{outcomes}{units}')
    (DOCS / "index.html").write_text(page(f"{content.NAME} · {syllabus.COURSE}", body))


def notebooks():
    out = DOCS / "notebooks"
    out.mkdir(exist_ok=True)
    exporter = HTMLExporter(template_name="lab")
    for nb in syllabus.NOTEBOOKS:
        body, _ = exporter.from_filename(str(ROOT / "notebooks" / f"{nb}.ipynb"))
        bar = (f'<div style="position:sticky;top:0;z-index:10;display:flex;gap:18px;align-items:center;padding:10px 22px;background:#fff;'
               f'border-bottom:1px solid #e3e8ec;font:14px Inter,system-ui,sans-serif"><a href="../index.html" style="color:#1d2b36;font-weight:700;text-decoration:none">'
               f'{content.NAME}</a><span style="color:#5b6b78">{html.escape(syllabus.NOTEBOOKS[nb])}</span>'
               f'<a href="{content.REPO}/blob/main/notebooks/{nb}.ipynb" style="margin-left:auto;color:{content.ACCENT}">.ipynb on GitHub</a></div>')
        body = re.sub(r"<body([^>]*)>", lambda m: f"<body{m.group(1)}>{bar}", body, count=1)
        body = re.sub(r"<title>.*?</title>", f"<title>{html.escape(syllabus.NOTEBOOKS[nb])} · {content.NAME}</title>", body, count=1)
        (out / f"{nb}.html").write_text(body)
        print("wrote", out / f"{nb}.html")


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    notebooks()
    index()
    (DOCS / ".nojekyll").write_text("")
    print("wrote", DOCS / "index.html")
