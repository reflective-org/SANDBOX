# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Fetch, process and embed the viz's static assets into plume_dynamics.html.

Asset groups, all embedded as data URIs OUTSIDE the bake markers (so re-running
bake_plume_dynamics.py never touches them):

  * DAY texture: NASA "Whole world - land and oceans" (public domain), full-resolution
    original from Wikimedia Commons (24000x12000, ~20 MB, cached); resized to 2048x1024,
    desaturated 10% / darkened 5%. -> MARBLE_URI in the JS.
  * NIGHT texture: NASA "The earth at night" city lights (public domain, 13500x6750);
    resized to 2048x1024. Blended in-shader on the night side. -> NIGHT_URI.
  * Source Serif 4 (OFL) Regular + Semibold, subset with pyftsubset (fonttools + brotli).
    -> the two @font-face blocks in the CSS.
  * Reflective logo, two recolors of the wordmark (yellow for dark, ink for light) at 56 px
    height. Source EPS is provided out-of-band (--logo-eps, needs ghostscript); without it
    the previously cached/embedded PNGs are reused. -> the two --logo CSS variables.

Only needed when changing an asset; the embedded copies are committed inside plume_dynamics.html.
Downloads cache in cache/ (gitignored).

Run (from anywhere): python coupled/viz/embed_assets.py
"""
import base64
import os
import re
import subprocess
import sys
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = os.path.join(_HERE, "cache")
_HTML = os.path.join(_HERE, "plume_dynamics.html")
_UA = "SANDBOX-viz-bake/1.0 (research visualization)"

_MARBLE_URL = ("https://upload.wikimedia.org/wikipedia/commons/8/8f/"
               "Whole_world_-_land_and_oceans_12000.jpg")
_NIGHT_URL = ("https://upload.wikimedia.org/wikipedia/commons/b/ba/The_earth_at_night.jpg")
_FONT_URL = ("https://raw.githubusercontent.com/adobe-fonts/source-serif/release/TTF/"
             "SourceSerif4-{weight}.ttf")
# basic latin + the page's extras: degree, micro, middot, multiply, right quote, ellipsis,
# subscript 2/3/4, en dash (labels may use it; em dash is banned from page text)
_UNICODES = "U+0020-007E,U+00B0,U+00B5,U+00B7,U+00D7,U+2013,U+2019,U+2026,U+2082-2084"


def _fetch(url, dest):
    if os.path.exists(dest):
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def build_textures():
    from PIL import Image, ImageEnhance
    Image.MAX_IMAGE_PIXELS = 400_000_000
    day_src = _fetch(_MARBLE_URL, os.path.join(_CACHE, "marble_full.jpg"))
    day = Image.open(day_src).convert("RGB").resize((2048, 1024), Image.LANCZOS)
    day = ImageEnhance.Color(day).enhance(0.90)
    day = ImageEnhance.Brightness(day).enhance(0.95)
    day_out = os.path.join(_CACHE, "marble_2048.jpg")
    day.save(day_out, quality=76, optimize=True)
    night_src = _fetch(_NIGHT_URL, os.path.join(_CACHE, "night_full.jpg"))
    night = Image.open(night_src).convert("RGB").resize((2048, 1024), Image.LANCZOS)
    night_out = os.path.join(_CACHE, "night_2048.jpg")
    night.save(night_out, quality=68, optimize=True)
    return day_out, night_out


def build_logos(eps_path):
    """Render the Reflective wordmark EPS to the two footer PNGs (56 px tall)."""
    raw = os.path.join(_CACHE, "logo_raw.png")
    subprocess.run(["gs", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-dEPSCrop",
                    "-sDEVICE=pngalpha", "-r300", f"-sOutputFile={raw}", eps_path], check=True)
    y = os.path.join(_CACHE, "logo_yellow.png")
    i = os.path.join(_CACHE, "logo_ink.png")
    subprocess.run(["magick", raw, "-trim", "+repage", "-resize", "x56", f"PNG32:{y}"], check=True)
    subprocess.run(["magick", y, "-channel", "RGB", "-fill", "#1c1d22", "-colorize", "100",
                    f"PNG32:{i}"], check=True)
    return y, i


def build_fonts():
    outs = []
    for weight in ("Regular", "Semibold"):
        ttf = _fetch(_FONT_URL.format(weight=weight),
                     os.path.join(_CACHE, f"SourceSerif4-{weight}.ttf"))
        out = os.path.join(_CACHE, f"SourceSerif4-{weight}.woff2")
        subprocess.run([sys.executable, "-m", "fontTools.subset", ttf,
                        f"--unicodes={_UNICODES}", "--flavor=woff2",
                        "--layout-features=kern,liga", f"--output-file={out}"], check=True)
        outs.append(out)
    return outs


def _patch(html, pattern, payloads, what):
    it = iter(payloads)
    out, n = re.subn(pattern, lambda m: m.group(1) + next(it), html, count=len(payloads))
    assert n == len(payloads), f"{what}: expected {len(payloads)} slots, patched {n}"
    return out


def _patch_optional(html, pattern, payloads, what):
    """Patch every slot the page has; pages without this asset group are skipped."""
    n = len(re.findall(pattern, html))
    if n == 0:
        print(f"  ({what}: no slots in this page, skipped)")
        return html
    assert n == len(payloads), f"{what}: page has {n} slots, expected {len(payloads)}"
    return _patch(html, pattern, payloads, what)


def inject(html_path, day, night, fonts, logos):
    b64 = lambda p: base64.b64encode(open(p, "rb").read()).decode()
    html = open(html_path).read()
    html = _patch_optional(html, r'(url\(data:font/woff2;base64,)[A-Za-z0-9+/=_]+',
                           [b64(f) for f in fonts], "@font-face")
    html = _patch_optional(html, r'(MARBLE_URI="data:image/jpeg;base64,)[A-Za-z0-9+/=_]+',
                           [b64(day)], "MARBLE_URI")
    html = _patch_optional(html, r'(NIGHT_URI="data:image/jpeg;base64,)[A-Za-z0-9+/=_]+',
                           [b64(night)], "NIGHT_URI")
    if logos:
        html = _patch_optional(html, r'(--logo:url\(data:image/png;base64,)[A-Za-z0-9+/=_]+',
                               [b64(logos[0]), b64(logos[1]), b64(logos[1])], "--logo")
    open(html_path, "w").write(html)
    print(f"embedded -> {html_path}   file = {os.path.getsize(html_path)/1024:.1f} KB")


if __name__ == "__main__":
    os.makedirs(_CACHE, exist_ok=True)
    eps = None
    if "--logo-eps" in sys.argv:
        eps = sys.argv[sys.argv.index("--logo-eps") + 1]
    html_path = _HTML
    if "--html" in sys.argv:
        html_path = sys.argv[sys.argv.index("--html") + 1]
    y, i = (build_logos(eps) if eps else
            (os.path.join(_CACHE, "logo_yellow.png"), os.path.join(_CACHE, "logo_ink.png")))
    logos = (y, i) if os.path.exists(y) and os.path.exists(i) else None
    day, night = build_textures()
    inject(html_path, day, night, build_fonts(), logos)
