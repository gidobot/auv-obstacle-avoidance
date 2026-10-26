"""Method figure: the commit-release manoeuvre.

Taken directly from the working doc (docs/traversing-the-cliff-manifold.html,
"Figure 1 - the manoeuvre") rather than redrawn, so the paper and the doc show
the same picture. The SVG is converted to PDF rather than screenshotted: the
output is vector, so it stays sharp at any print size and its text is real
text.

Three things have to be resolved on the way out, because the web original
leans on the page around it:

  * CSS custom properties (var(--ink) and friends) have no value standalone,
  * the class styles live in the page's stylesheet, not in the SVG,
  * IBM Plex is not installed here, so the stack falls back to DejaVu.

Type is scaled up and the viewBox cropped to the used area, because a figure
that was drawn for a browser at ~760 px is being set in a 3.45 in column.

    python3 make_manoeuvre_fig.py
"""
import re
import cairosvg

SRC = ("/home/gidobot/workspace/acfr/auv-obstacle-avoidance/docs/"
       "traversing-the-cliff-manifold.html")
OUT = ("/home/gidobot/workspace/acfr/auv-obstacle-avoidance/docs/paper/"
       "6aa364bd4d6d76d5dae0d68b/figures/manoeuvre")

VARS = {"--ink": "#10202B", "--accent": "#B1187A", "--muted": "#5A6B76",
        "--rule-firm": "#B8C7D0", "--d2": "#9EC3D4", "--d3": "#5793AE",
        "--d5": "#0E4257"}

# the page's own class rules, with the font stack repointed and sizes scaled
FONT_SCALE = 1.40
STYLE = """
  .svg-label-em {font-family:"DejaVu Sans Mono",monospace;font-size:%.1fpx;
                 fill:%s;font-weight:600;}
  .svg-label    {font-family:"DejaVu Sans Mono",monospace;font-size:%.1fpx;
                 fill:%s;}
  .svg-axis     {font-family:"DejaVu Sans Mono",monospace;font-size:%.1fpx;
                 fill:%s;}
""" % (11 * FONT_SCALE, VARS["--accent"],
       11 * FONT_SCALE, VARS["--muted"],
       10 * FONT_SCALE, VARS["--muted"])

CROP = (45, 18, 665, 275)          # x, y, w, h of the area actually drawn
WIDTH_PT = 248.0                   # one IEEE column

svg = re.findall(r"<svg.*?</svg>", open(SRC).read(), flags=re.S)[0]
for name, hexv in VARS.items():
    svg = svg.replace("var(%s)" % name, hexv)
assert "var(--" not in svg, "unresolved CSS custom property"

svg = svg.replace(
    'viewBox="0 0 760 320"',
    'xmlns="http://www.w3.org/2000/svg" width="%.2fpt" height="%.2fpt" '
    'viewBox="%d %d %d %d"'
    % (WIDTH_PT, WIDTH_PT * CROP[3] / CROP[2], *CROP))
svg = svg.replace("<defs>", "<style>%s</style>\n<defs>" % STYLE, 1)

# Three labels were positioned for 11 px type and collide once it is scaled
# for print. Re-fit them; everything else is the web original untouched.
REFIT = [
    # the depth hint hung off the left of the viewBox at the larger size
    ('<text class="svg-axis" x="52" y="160" text-anchor="end">depth \u2193</text>',
     '<text class="svg-axis" x="66" y="36">depth \u2193</text>'),
    # the altitude callout ran past the crest at the larger size
    ('<text class="svg-label" x="207" y="207">2 m imaging altitude</text>',
     '<text class="svg-label" x="207" y="207">2 m altitude</text>'),
    # magenta on the dark seabed gradient is unreadable in print
    ('<text class="svg-label-em" x="406" y="248" text-anchor="middle">'
     'standoff + vehicle length</text>',
     '<text class="svg-label-em" x="406" y="245" text-anchor="middle" '
     'style="fill:#EAF2F6">standoff + vehicle length</text>'),
]
for before, after in REFIT:
    assert before in svg, "re-fit anchor missing: " + before[:60]
    svg = svg.replace(before, after)

cairosvg.svg2pdf(bytestring=svg.encode(), write_to=OUT + ".pdf")
cairosvg.svg2png(bytestring=svg.encode(), write_to="/tmp/manoeuvre_preview.png",
                 scale=4.0)
print("wrote %s.pdf  (%.0f x %.0f pt, vector)"
      % (OUT, WIDTH_PT, WIDTH_PT * CROP[3] / CROP[2]))
