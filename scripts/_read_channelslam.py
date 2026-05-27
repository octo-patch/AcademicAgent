import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pdf_parser import PDFParser
pdf = "EE_papers/[ChannelSLAM]Multipath_Assisted_Positioning_with_Simultaneous_Localization_and_Mapping.pdf"
with PDFParser(str(PROJECT_ROOT / pdf)) as p:
    text = p.extract_full_text()

out = PROJECT_ROOT / "data" / "_tmp_channelslam_text.txt"
out.write_text(text, encoding="utf-8")
print(f"wrote {out}, len={len(text)}")
