"""Method figure: the full commit-release-tail-clear sequence.

Drawn in the working doc's own idiom (docs/traversing-the-cliff-manifold.html,
"Figure 1 - the manoeuvre"): the same palette, type and marks, as SVG, then
converted to a vector PDF rather than screenshotted, so the text stays real
text and the figure stays sharp at any print size.

The doc's drawing stops at the crest. This one carries the profile through a
descending edge as well, so every mode the supervisor can command appears
once along a single transect:

    ALT_FOLLOW -> OBSTACLE_CLEAR -> OBSTACLE_HOLD -> ALT_FOLLOW
               -> TAIL_CLEAR -> ALT_CORRECTION -> ALT_FOLLOW

Two colours carry the two kinds of annotation: magenta for events the
controller reacts to, teal for the mode it is in between them.

Set across both columns. At one column the doc's 11 px type falls to about
4 pt, and scaling it up breaks a layout built for a browser; at
\\textwidth it lands near 6 pt with the original proportions intact.

    python3 make_manoeuvre_fig.py
"""
import cairosvg

OUT = ("/home/gidobot/workspace/acfr/auv-obstacle-avoidance/docs/paper/"
       "6aa364bd4d6d76d5dae0d68b/figures/manoeuvre")

INK, ACCENT, MUTED = "#10202B", "#B1187A", "#5A6B76"
RULE, D2, D3, D5 = "#B8C7D0", "#9EC3D4", "#5793AE", "#0E4257"
ON_SEABED = "#EAF2F6"                      # accent is unreadable on the fill

CROP = (45, 18, 920, 275)                  # the area actually drawn
WIDTH_PT = 516.0                           # \textwidth, two columns

# profile, in viewBox units: 19 px is the 2 m imaging altitude
X_COMMIT, X_CREST, X_RELEASE = 254, 300, 427
X_EDGE, X_CLEAR = 600, 727
Z_DEEP, Z_PLATEAU, Z_LOW = 212, 155, 225
Z_FLY_LOW, Z_FLY_HIGH, Z_FLY_END = 193, 136, 206


def ev(x, y, em, sub, anchor):
    """An event: what the controller reacted to, and where."""
    return ('<text class="ev" x="%d" y="%d" text-anchor="%s">%s</text>'
            '<text class="lb" x="%d" y="%d" text-anchor="%s">%s</text>'
            % (x, y, anchor, em, x, y + 14, anchor, sub))


def mode(x, y, name, anchor="middle"):
    return ('<text class="md" x="%d" y="%d" text-anchor="%s">%s</text>'
            % (x, y, anchor, name))


def vehicle(x, y, floor):
    """Hull plus the downward beam fan, which is what sees the floor."""
    return ('<rect x="%d" y="%d" width="30" height="11" rx="5" fill="%s"/>'
            '<path d="M %d,%d L %d,%d L %d,%d Z" fill="%s" opacity="0.4"/>'
            % (x, y, INK, x + 15, y + 11, x - 8, floor, x + 38, floor, D3))


def band(x0, x1):
    return ('<rect x="%d" y="26" width="%d" height="212" fill="%s" '
            'opacity="0.09"/>' % (x0, x1 - x0, ACCENT))


def rule_at(x):
    """Starts below the label rows: a 1.4 px dash through 11 px type bisects
    glyphs, and the labels sit directly above each line anyway."""
    return ('<line x1="%d" y1="110" x2="%d" y2="264" stroke="%s" '
            'stroke-width="1.4" stroke-dasharray="3 3"/>' % (x, x, ACCENT))


def dot(x, y):
    return '<circle cx="%d" cy="%d" r="4.5" fill="%s"/>' % (x, y, ACCENT)


def measure(x0, x1, y, label):
    return ('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" '
            'stroke-width="1.3" marker-end="url(#arw)"/>'
            '<text class="ev onsea" x="%d" y="%d" text-anchor="middle">%s'
            '</text>' % (x0, y, x1, y, ON_SEABED,
                         (x0 + x1) // 2, y - 7, label))


SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="%(w).2fpt"
  height="%(h).2fpt" viewBox="%(cx)d %(cy)d %(cw)d %(ch)d" role="img"
  aria-label="Depth profile of a transect crossing a rising crest and then a
  descending edge, annotated with the commit and release points, the interval
  over which the crest is unobservable, the interval over which the stern has
  not cleared the edge, and the control mode held along each segment.">
<style>
  .ev {font-family:"DejaVu Sans Mono",monospace;font-size:11px;fill:%(ac)s;
       font-weight:600;}
  .lb {font-family:"DejaVu Sans Mono",monospace;font-size:11px;fill:%(mu)s;}
  .md {font-family:"DejaVu Sans Mono",monospace;font-size:9.5px;fill:%(d3)s;
       font-weight:600;}
  .ax {font-family:"DejaVu Sans Mono",monospace;font-size:10px;fill:%(mu)s;}
  .onsea {fill:%(sea)s;}
</style>
<defs>
  <linearGradient id="seabed" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%%" stop-color="%(d2)s"/><stop offset="100%%" stop-color="%(d5)s"/>
  </linearGradient>
  <marker id="arw" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7"
          markerHeight="7" orient="auto">
    <path d="M0,0 L8,4 L0,8 z" fill="%(sea)s"/>
  </marker>
</defs>
%(bands)s
<path d="M 60,%(zd)d L %(xc)d,%(zd)d L %(xc)d,%(zp)d L %(xe)d,%(zp)d
         L %(xe)d,%(zl)d L 940,%(zl)d L 940,264 L 60,264 Z" fill="url(#seabed)"/>
<path d="M 60,%(zd)d L %(xc)d,%(zd)d L %(xc)d,%(zp)d L %(xe)d,%(zp)d
         L %(xe)d,%(zl)d L 940,%(zl)d" fill="none" stroke="%(d5)s"
      stroke-width="1.6"/>
<path d="M 60,%(fl)d L %(xcm)d,%(fl)d L %(xcm)d,%(fh)d L %(xcl)d,%(fh)d
         L %(xcl)d,%(fe)d L 940,%(fe)d" fill="none" stroke="%(ink)s"
      stroke-width="2" stroke-dasharray="6 4"/>
%(vehicles)s
%(rules)s
%(dots)s
%(captions)s
%(events)s
%(modes)s
%(measures)s
<line x1="160" y1="%(fl)d" x2="160" y2="%(zd)d" stroke="%(mu)s" stroke-width="1"/>
<text class="lb" x="167" y="207">2 m altitude</text>
<line x1="60" y1="264" x2="940" y2="264" stroke="%(ru)s" stroke-width="1"/>
<line x1="60" y1="26" x2="60" y2="264" stroke="%(ru)s" stroke-width="1"/>
<text class="ax" x="66" y="36">depth ↓</text>
<text class="ax" x="60" y="282">0</text>
<text class="ax" x="%(xc)d" y="282" text-anchor="middle">crest</text>
<text class="ax" x="%(xe)d" y="282" text-anchor="middle">edge</text>
<text class="ax" x="940" y="282" text-anchor="end">along-track →</text>
</svg>""" % dict(
    w=WIDTH_PT, h=WIDTH_PT * CROP[3] / CROP[2],
    cx=CROP[0], cy=CROP[1], cw=CROP[2], ch=CROP[3],
    ac=ACCENT, mu=MUTED, d2=D2, d3=D3, d5=D5, ink=INK, ru=RULE, sea=ON_SEABED,
    xc=X_CREST, xe=X_EDGE, xcm=X_COMMIT, xcl=X_CLEAR,
    zd=Z_DEEP, zp=Z_PLATEAU, zl=Z_LOW,
    fl=Z_FLY_LOW, fh=Z_FLY_HIGH, fe=Z_FLY_END,
    bands=band(X_CREST, X_RELEASE) + band(X_EDGE, X_CLEAR),
    captions="".join([
        ev(363, 44, "crest not observable",
           "no beam reaches above the vehicle", "middle"),
        ev(663, 44, "descent withheld",
           "stern not yet past the edge", "middle")]),
    vehicles="".join([
        vehicle(100, 187, Z_DEEP), vehicle(325, 130, Z_PLATEAU),
        vehicle(640, 130, Z_LOW), vehicle(835, 200, Z_LOW)]),
    rules="".join(rule_at(x) for x in (X_COMMIT, X_RELEASE, X_EDGE, X_CLEAR)),
    dots="".join([dot(X_COMMIT, Z_FLY_LOW), dot(X_RELEASE, Z_FLY_HIGH),
                  dot(X_EDGE, Z_FLY_HIGH), dot(X_CLEAR, Z_FLY_HIGH)]),
    events="".join([
        ev(248, 86, "commit", "crest enters window", "end"),
        ev(421, 86, "release", "stern past the crest", "end"),
        ev(594, 86, "edge", "descending edge reached", "end"),
        ev(737, 86, "descend", "stern past the edge", "start")]),
    modes="".join([
        mode(157, 186, "ALT_FOLLOW"), mode(248, 168, "OBSTACLE_CLEAR", "end"),
        mode(363, 124, "OBSTACLE_HOLD"), mode(513, 124, "ALT_FOLLOW"),
        mode(663, 124, "TAIL_CLEAR"),
        mode(737, 176, "ALT_CORRECTION", "start"),
        mode(850, 199, "ALT_FOLLOW")]),
    measures=(measure(X_CREST, X_RELEASE, 252, "standoff + vehicle length")
              + measure(X_EDGE, X_CLEAR, 252, "standoff + vehicle length")),
)

cairosvg.svg2pdf(bytestring=SVG.encode(), write_to=OUT + ".pdf")
cairosvg.svg2png(bytestring=SVG.encode(),
                 write_to="/tmp/manoeuvre_preview.png", scale=2.4)
print("wrote %s.pdf  (%.0f x %.0f pt, vector)"
      % (OUT, WIDTH_PT, WIDTH_PT * CROP[3] / CROP[2]))
