# utils/file_handler.py
# Lightweight text-extraction helpers for hackathon MVP
# ----------------------------------------------------

import logging
from pathlib import Path

# Explicitly import errors for specific handling
import PyPDF2
from PyPDF2.errors import PdfReadError 
import docx

logging.basicConfig(level=logging.INFO)

# ---------- internal helpers -------------------------------------------------
def _extract_pdf(path: Path) -> str:
    """Return concatenated text from every page in a PDF."""
    text = ""
    try:
        with path.open("rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                page_txt = page.extract_text()
                # Use a cleaner way to concatenate non-empty text
                if page_txt and page_txt.strip():
                    text += page_txt.strip() + "\n"
    except PdfReadError:
        logging.error(f"PDF extraction failed - File is corrupted or encrypted: {path.name}")
    except Exception as err:
        logging.error(f"PDF extraction failed - {path.name}: {err}")
    return text


def _extract_docx(path: Path) -> str:
    """Return concatenated paragraph text from a DOCX."""
    text = ""
    try:
        doc = docx.Document(path)
        # Cleaner list comprehension for concatenation
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as err:
        logging.error(f"DOCX extraction failed – {path.name}: {err}")
    return text


# ---------- public API --------------------------------------------------------
def extract_text(file_path: str) -> str:
    """
    Extract raw text from a PDF or DOCX. 
    Unsupported types return an empty string.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the contract file.

    Returns
    -------
    str
        Extracted UTF-8 text (may be empty if parsing fails).
    """
    path = Path(file_path)
    
    # Improvement: Check if file exists first
    if not path.is_file():
        logging.error(f"File not found: {file_path}")
        return ""

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        return _extract_docx(path)

    logging.error(f"Unsupported file type: {path.name}")
    return ""


# ---------- quick CLI test ----------------------------------------------------
if __name__ == "__main__":
    # Note: These files must exist in the execution directory for the test to run properly
    for sample in ("sample_contract.pdf", "sample_contract.docx", "non_existent_file.txt"):
        txt = extract_text(sample)
        print(f"{sample}: {len(txt)} chars extracted")


