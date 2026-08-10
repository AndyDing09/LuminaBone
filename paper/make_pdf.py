"""Render the HTML paper to a downloadable PDF via headless Edge/Chrome.
Run: python make_pdf.py   ->  LuminaBone_paper.pdf
(The IEEE methods paper is LaTeX; compile LuminaBone_methods.tex on Overleaf.)
"""
import os, subprocess, glob

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, "LuminaBone_paper.html")
PDF = os.path.join(HERE, "LuminaBone_paper.pdf")

CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]
browser = next((p for p in CANDIDATES if os.path.exists(p)), None)
if not browser:
    hits = glob.glob(r"C:\Program Files*\**\msedge.exe", recursive=True)
    browser = hits[0] if hits else None
if not browser:
    raise SystemExit("No Edge/Chrome found; open the HTML and use Print -> Save as PDF.")

subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={PDF}", "file:///" + HTML.replace("\\", "/")],
               check=True)
print("wrote", PDF, os.path.getsize(PDF), "bytes")
