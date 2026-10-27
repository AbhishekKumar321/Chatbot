"""Convert PROJECT_DOCUMENTATION.md to a styled PDF."""

from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "PROJECT_DOCUMENTATION.md"
DST = ROOT / "PROJECT_DOCUMENTATION.pdf"

CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: Helvetica, sans-serif; font-size: 10.5pt; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 20pt; color: #0b3d91; border-bottom: 2px solid #0b3d91; padding-bottom: 6px; margin-top: 0; }
h2 { font-size: 15pt; color: #0b3d91; margin-top: 22px; border-bottom: 1px solid #ccc; padding-bottom: 3px; }
h3 { font-size: 12.5pt; color: #333; margin-top: 16px; }
code { background-color: #f2f2f2; padding: 1px 4px; font-family: Courier, monospace; font-size: 9.5pt; }
pre { background-color: #f2f2f2; padding: 8px; font-family: Courier, monospace; font-size: 9pt; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; }
th, td { border: 1px solid #999; padding: 5px 8px; font-size: 9.5pt; }
th { background-color: #0b3d91; color: white; }
a { color: #0b3d91; }
blockquote { border-left: 3px solid #0b3d91; margin-left: 0; padding-left: 10px; color: #444; }
"""


UNICODE_REPLACEMENTS = {
    "→": "->",
    "←": "<-",
    "✓": "[x]",
    "✔": "[x]",
    "•": "-",
}


def main() -> None:
    md_text = SRC.read_text(encoding="utf-8")
    for old, new in UNICODE_REPLACEMENTS.items():
        md_text = md_text.replace(old, new)
    body_html = markdown.markdown(md_text, extensions=["extra", "tables", "fenced_code", "toc"])
    html = f"<html><head><style>{CSS}</style></head><body>{body_html}</body></html>"

    with DST.open("wb") as f:
        result = pisa.CreatePDF(html, dest=f)

    if result.err:
        raise SystemExit(f"PDF generation failed with {result.err} error(s)")

    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
