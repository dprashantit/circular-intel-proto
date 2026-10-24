"""PDF -> (metadata dict, full text). Confirmed against the sample NSE/BSE
PDFs: both are text-native, so PyMuPDF alone is sufficient — no OCR here.
Add an OCR fallback (pytesseract) only once a source with scanned PDFs shows
up (low chars-per-page is the signal to watch for).
"""
import re
import json
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import yaml

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
MONTHS_ABBR = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"


def load_sources_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_text(pdf_path: str) -> tuple[str, int]:
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages), len(pages)


def _parse_date(raw: str) -> str | None:
    raw = raw.strip()
    for fmt in ("%B %d, %Y", "%B %d %Y", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def detect_source(filename: str, sources_cfg: dict) -> str | None:
    for source, cfg in sources_cfg["sources"].items():
        if re.match(cfg["filename_regex"], filename):
            return source
    return None


def extract_cover_fields(text: str, source: str, sources_cfg: dict) -> dict:
    """Field patterns are matched against whitespace-normalized text — some
    sources (BSE notices) render as one PDF form field per line, so a label
    and its value can be separated by newlines rather than spaces."""
    cfg = sources_cfg["sources"].get(source, {})
    normalized = re.sub(r"\s+", " ", text)
    fields = {}
    for field, pattern in cfg.get("cover_fields", {}).items():
        m = re.search(pattern, normalized, re.IGNORECASE)
        if m:
            value = re.sub(r"\s+", " ", m.group(1)).strip()
            if field == "native_id":
                value = value.replace(" ", "")
            fields[field] = value
    if "issue_date" in fields:
        parsed = _parse_date(fields["issue_date"])
        if parsed:
            fields["issue_date"] = parsed
    if source == "NSE" and "subject" not in fields:
        # subject is the first non-blank line after "All Members,"
        m = re.search(r"All Members,\s*\n+\s*(.+)", text)
        if m:
            fields["title"] = m.group(1).strip()
    elif "subject" in fields:
        fields["title"] = fields.pop("subject")
    return fields


def parse_pdf(pdf_path: str, sources_cfg: dict) -> dict:
    filename = Path(pdf_path).name
    source = detect_source(filename, sources_cfg)
    text, n_pages = extract_text(pdf_path)
    fields = extract_cover_fields(text, source, sources_cfg) if source else {}

    chars_per_page = len(text) / max(n_pages, 1)
    quality = "ok" if chars_per_page > 50 else "low_text_possible_scan"

    return {
        "source": source,
        "filename": filename,
        "n_pages": n_pages,
        "text": text,
        "quality": quality,
        **fields,
    }


def parse_and_save(pdf_path: str, sources_cfg: dict, parsed_dir: str) -> dict:
    result = parse_pdf(pdf_path, sources_cfg)
    out_dir = Path(parsed_dir) / (result["source"] or "UNKNOWN")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(pdf_path).stem + ".json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    result["parsed_path"] = str(out_path)
    return result
