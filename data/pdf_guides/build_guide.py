#!/usr/bin/env python3
"""Build the user guides (PDF) and SETUP.md.

    python3 data/pdf_guides/build_guide.py            # all guides + SETUP.md
    python3 data/pdf_guides/build_guide.py services   # just one guide

Each guide is <name>_src.html. Per guide we render once with a blank table of
contents, find which page each heading landed on (pdftotext), then render
again with the page numbers filled in (headless Chrome).

Shared text:
  <!--INCLUDE:waiver:s3-->   pulls <section id="s3"> out of the waiver guide
                             source (so set-up chapters can never drift apart)
  <!--AI_PROMPT-->           the AI-assistant set-up prompt (ai_setup_prompt.txt)
Screenshots are copied into img/ (chat shots cropped so the text is readable)
and any local file path found in them (via OCR) is blacked out.
"""
import html as _html
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
IMG = HERE / "img"
PROMPT = (HERE / "ai_setup_prompt.txt").read_text().rstrip("\n")

GUIDES = {
    "waiver": "waiver_adk_guide",
    "services": "services_adk_guide",
}

# Chat screenshots keep their full width (they have no side panel worth cropping)
KEEP_FULL = {"adk_00_welcome_before_selecting_app.jpg", "adk_01_app_picker.jpg",
             "adk_03_waiver_agent_empty_chat.jpg"}

# Text that marks a local path in a screenshot (OCR); the whole line is hidden
PATH_MARKERS = ("/users/", "/home/", "/private/")

# Per-section text patches when a section is included in another guide
INCLUDE_PATCHES = {
    ("services", "s2"): [
        ("A full extraction of the 125-page test waiver took about <b>6 minutes</b> and many model calls; a single custom value took about <b>20 to 30 seconds</b> and one call.",
         "Services extraction makes <b>one model call per service</b>, plus calls to find the service list, check the results and re-read anything flagged, so a waiver with many services makes many calls. In our test, extracting a single service from a waiver and running it through the rest of the pipeline took about <b>4 minutes</b>; one custom value took about <b>30 seconds</b> and one call."),
    ],
}


# ---------------------------------------------------------------- images

def _ocr_redact(im):
    """Black out every OCR line that looks like a local file path."""
    from PIL import ImageDraw
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "i.png"
        im.convert("RGB").save(p)
        tsv = subprocess.run(["tesseract", str(p), "-", "--psm", "6", "tsv"],
                             capture_output=True, text=True).stdout
    rows = []
    for ln in tsv.splitlines()[1:]:
        f = ln.split("\t")
        if len(f) < 12 or not f[11].strip():
            continue
        rows.append(dict(key=(f[2], f[3], f[4]), l=int(f[6]), t=int(f[7]), w=int(f[8]),
                         h=int(f[9]), text=f[11]))
    lines = {}
    for r in rows:
        lines.setdefault(r["key"], []).append(r)
    ordered = sorted(lines.items(), key=lambda kv: (min(x["t"] for x in kv[1]), min(x["l"] for x in kv[1])))
    d = ImageDraw.Draw(im)
    flagged_prev = 0
    for _, ws in ordered:
        txt = " ".join(w["text"] for w in ws)
        hit = any(m in txt.lower() for m in PATH_MARKERS)
        # a wrapped path continues on the next line(s): few words, contains / or long token
        cont = flagged_prev > 0 and len(ws) <= 3 and ("/" in txt or "-" in txt)
        if hit or cont:
            x0 = min(w["l"] for w in ws) - 4
            y0 = min(w["t"] for w in ws) - 3
            x1 = max(w["l"] + w["w"] for w in ws) + 4
            y1 = max(w["t"] + w["h"] for w in ws) + 3
            d.rectangle((x0, y0, x1, y1), fill=(28, 28, 28))
            flagged_prev = 3 if hit else flagged_prev - 1
        else:
            flagged_prev = max(0, flagged_prev - 1)
    return im


def prepare_images(html):
    from PIL import Image
    IMG.mkdir(exist_ok=True)

    def sub(m):
        name = m.group(1)
        im = Image.open(HERE.parent / "pdf_images" / name).convert("RGB")
        w, h = im.size
        if w >= 1500:
            im = _ocr_redact(im)
        if name not in KEEP_FULL and w >= 1500:
            im = im.crop((480, 96, w, h - 118 if h <= 720 else h))
        im.save(IMG / name, quality=92)
        return f'src="img/{name}"'
    return re.sub(r'src="\.\./pdf_images/([^"]+)"', sub, html)


# ---------------------------------------------------------------- build

def _strip(t):
    return re.sub(r"<[^>]+>", "", t).replace("&amp;", "&").strip()


def _include(src, guide):
    waiver = (HERE / "waiver_adk_guide_src.html").read_text()

    def sub(m):
        sid = m.group(1)
        if sid.startswith("app"):  # an appendix heading block, up to the end of its section
            sm = re.search(r'<h2 style="margin-top:1.4em" id="%s">.*?(?=</section>)' % re.escape(sid), waiver, re.S)
        else:
            sm = re.search(r'<section class="chapter" id="%s">.*?</section>' % re.escape(sid), waiver, re.S)
        if not sm:
            sys.exit(f"include: section {sid} not found in the waiver guide")
        text = sm.group(0)
        for old, new in INCLUDE_PATCHES.get((guide, sid), []):
            if old not in text:
                sys.exit(f"include patch not found in {sid}: {old[:50]}")
            text = text.replace(old, new)
        return text
    return re.sub(r"<!--INCLUDE:waiver:(\w+)-->", sub, src)


def build(guide):
    stem = GUIDES[guide]
    src = _include((HERE / f"{stem}_src.html").read_text(), guide)
    out_html, out_pdf = HERE / f"{stem}.html", HERE / f"{stem}.pdf"

    heads = []
    for m in re.finditer(r'<h([23])([^>]*)>(.*?)</h\1>', src, re.S):
        idm = re.search(r'id="([^"]+)"', m.group(2))
        if idm:
            heads.append((int(m.group(1)), idm.group(1), _strip(m.group(3))))
    for m in re.finditer(r'<section class="chapter" id="([^"]+)">\s*<h2>(.*?)</h2>', src, re.S):
        heads.append((2, m.group(1), _strip(m.group(2))))
    heads.sort(key=lambda h: src.find(f'id="{h[1]}"'))

    def toc(pg):
        out = ["<ul>"]
        for lvl, hid, text in heads:
            out.append(f'<li class="l{lvl}"><a href="#{hid}">{text}</a><span class="dots"></span><span>{pg.get(hid, "")}</span></li>')
        return "\n".join(out + ["</ul>"])

    def render(pg):
        page = src.replace("<!--TOC-->", toc(pg)).replace("<!--AI_PROMPT-->", _html.escape(PROMPT))
        out_html.write_text(prepare_images(page))
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={out_pdf}", f"file://{out_html}"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def pages():
        n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(out_pdf)], capture_output=True, text=True).stdout).group(1))
        return [subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), "-layout", str(out_pdf), "-"],
                               capture_output=True, text=True).stdout for i in range(1, n + 1)]

    render({})
    txt = pages()
    norm = lambda s: re.sub(r"\s+", " ", s)
    last = norm(heads[-1][2])
    first_body = next(i for i, t in enumerate(txt) if last in norm(t)) + 1  # TOC ends on the page naming the last heading
    pg = {}
    for lvl, hid, text in heads:
        for i in range(first_body, len(txt)):
            if norm(text) in norm(txt[i]):
                pg[hid] = i + 1
                break
    render(pg)
    missing = [h[2] for h in heads if h[1] not in pg]
    print(f"{guide}: {len(txt)} pages | {len(heads)} headings | not located: {missing}")


def build_setup():
    setup = (HERE / "SETUP_template.md").read_text().replace("{{AI_PROMPT}}", PROMPT)
    (HERE.parent.parent / "SETUP.md").write_text(setup)
    print("wrote SETUP.md")


if __name__ == "__main__":
    which = sys.argv[1:] or list(GUIDES)
    for g in which:
        build(g)
    if not sys.argv[1:]:
        build_setup()
