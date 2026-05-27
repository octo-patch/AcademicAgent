import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pdf_parser import PDFParser
with PDFParser("data/papers/arxiv_2210_03629.pdf") as p:
    text = p.extract_full_text()

# write to a UTF-8 file (PowerShell cp936 console can't print)
out = PROJECT_ROOT / "data" / "_tmp_react_text.txt"
out.write_text(text, encoding="utf-8")
print(f"wrote {out}, len={len(text)}")
