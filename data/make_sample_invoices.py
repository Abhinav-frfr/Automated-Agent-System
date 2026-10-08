"""Generate sample invoice PDFs for local Day 1 testing."""

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

OUT_DIR = Path(__file__).resolve().parent / "invoices"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    # Amazon
    ("amazon_july.pdf",     "Amazon Web Services", "AMZ-2026-0701", "July 1, 2026",      "$1,200.00", "August 1, 2026"),
    ("amazon_august.pdf",   "Amazon Web Services", "AMZ-2026-0801", "August 1, 2026",    "$1,350.00", "September 1, 2026"),
    ("amazon_september.pdf","Amazon Web Services", "AMZ-2026-0901", "September 1, 2026", "$1,480.00", "October 1, 2026"),
    # OpenAI
    ("openai_july.pdf",     "OpenAI",              "OAI-2026-0701", "July 1, 2026",      "$4,000.00", "August 1, 2026"),
    ("openai_august.pdf",   "OpenAI",              "OAI-2026-0801", "August 1, 2026",    "$4,120.00", "September 1, 2026"),
    ("openai_september.pdf","OpenAI",              "OAI-2026-0901", "September 1, 2026", "$4,250.00", "October 15, 2026"),
    # ABC Corp
    ("abccorp_july.pdf",    "ABC Corp",            "ABC-2026-0701", "July 1, 2026",      "$9,000.00", "August 1, 2026"),
    ("abccorp_august.pdf",  "ABC Corp",            "ABC-2026-0801", "August 1, 2026",    "$9,500.00", "September 1, 2026"),
    ("abccorp_september.pdf","ABC Corp",           "ABC-2026-0901", "September 1, 2026", "$9,800.00", "October 20, 2026"),
]


def make_pdf(filename: str, vendor: str, inv_no: str, inv_date: str, amount: str, due: str) -> None:
    path = OUT_DIR / filename
    c = canvas.Canvas(str(path), pagesize=LETTER)
    width, height = LETTER
    y = height - 100

    lines = [
        "INVOICE",
        "",
        f"Vendor:         {vendor}",
        f"Invoice Number: {inv_no}",
        f"Invoice Date:   {inv_date}",
        f"Amount Due:     {amount}",
        f"Due Date:       {due}",
        "",
        "Thank you for your business.",
    ]
    for line in lines:
        c.drawString(80, y, line)
        y -= 20

    c.showPage()
    c.save()


if __name__ == "__main__":
    for s in SAMPLES:
        make_pdf(*s)
    print(f"Created {len(SAMPLES)} sample invoices in {OUT_DIR}")