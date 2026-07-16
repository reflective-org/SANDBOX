# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Fetch, process and embed the viz's static assets into d1_globe.html.

Two asset groups, both embedded as data URIs OUTSIDE the bake markers (so re-running
bake_d1_globe.py never touches them):

  * Blue Marble texture: NASA "Whole world - land and oceans" (public domain), via the
    Wikimedia Commons 1280px thumbnail; resized to 1024x512, desaturated 18% and darkened 8%
    so it reads as an illustration under the data. -> the MARBLE_URI constant in the JS.
  * Source Serif 4 (OFL) Regular + Semibold, subset to the page's character set with
    pyftsubset (fonttools + brotli). -> the two @font-face blocks in the CSS.

Only needed when changing the texture or the font subset; the embedded copies are committed
inside d1_globe.html. Downloads cache in cache/ (gitignored).

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
_HTML = os.path.join(_HERE, "d1_globe.html")
_UA = "SANDBOX-viz-bake/1.0 (research visualization)"

_MARBLE_URL = ("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/"
               "Whole_world_-_land_and_oceans_12000.jpg/"
               "1280px-Whole_world_-_land_and_oceans_12000.jpg")
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


def build_marble():
    from PIL import Image, ImageEnhance
    src = _fetch(_MARBLE_URL, os.path.join(_CACHE, "marble_1280.jpg"))
    img = Image.open(src).convert("RGB").resize((1024, 512), Image.LANCZOS)
    img = ImageEnhance.Color(img).enhance(0.82)
    img = ImageEnhance.Brightness(img).enhance(0.92)
    out = os.path.join(_CACHE, "marble_1024.jpg")
    img.save(out, quality=72, optimize=True)
    return out


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


def inject(marble, fonts):
    b64 = lambda p: base64.b64encode(open(p, "rb").read()).decode()
    html = open(_HTML).read()
    html, n = re.subn(r'(url\(data:font/woff2;base64,)[A-Za-z0-9+/=]+',
                      lambda m, it=iter(fonts): m.group(1) + b64(next(it)), html, count=2)
    assert n == 2, f"expected 2 @font-face data URIs, patched {n}"
    html, n = re.subn(r'(MARBLE_URI="data:image/jpeg;base64,)[A-Za-z0-9+/=]+',
                      lambda m: m.group(1) + b64(marble), html, count=1)
    assert n == 1, "MARBLE_URI constant not found"
    open(_HTML, "w").write(html)
    print(f"embedded -> {_HTML}   file = {os.path.getsize(_HTML)/1024:.1f} KB")


if __name__ == "__main__":
    os.makedirs(_CACHE, exist_ok=True)
    inject(build_marble(), build_fonts())
