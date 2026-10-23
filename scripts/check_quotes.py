"""Checks that every passage we quote from the DLBAC paper occurs verbatim in its text. The paper is not distributed here: get arXiv:2203.15124v1 yourself and convert it with `pdftotext paper.pdf paper.txt` (default reading order, not -layout).
A quote is a "double-quoted" or ``LaTeX-quoted'' passage; "..." inside a quote splits it into pieces that are checked separately. Case, whitespace, ligatures, hyphenation at line ends, the space before 's and the space the PDF puts inside dataset names (u5k-r 5k) are ignored.
usage: python -I scripts/check_quotes.py paper.txt docs/claims_audit.md paper/main.tex ...   (prints MISSING lines, exit 1 if any)"""
import re, sys, unicodedata

def norm(s):
    s = unicodedata.normalize("NFKC", s).replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("−", "-")
    return re.sub(r"-r (\d)", r"-r\1", re.sub(r"\s+", " ", re.sub(r" ('s)", r"\1", re.sub(r"(\w)-\n(\w)", r"\1\2", s)))).lower()

paper = norm(open(sys.argv[1]).read()); bad = total = 0
for path in sys.argv[2:]:
    text = open(path).read()
    for q in re.findall(r'"([^"\n]{12,}?)"|``(.+?)\'\'', text, re.S):
        for piece in re.split(r"\.\.\.|\\ldots", "".join(q)):
            piece = norm(piece.replace("\\,", " ")).strip(" .,;")
            if len(piece) < 12: continue
            total += 1
            if piece not in paper: bad += 1; print(f"MISSING in {path}: {piece[:110]}")
print(f"{total - bad} of {total} quoted passages found in {sys.argv[1]}"); sys.exit(1 if bad else 0)
