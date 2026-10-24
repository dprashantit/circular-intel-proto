"""Structure-aware chunking: split on blank lines / numbered paragraphs,
target ~400-700 chars with ~15% overlap. Each chunk's embedding text is
prefixed with a header line (source, id, date, title) so similarity search
doesn't need the chunk to repeat that context verbatim.
"""
import re
from dataclasses import dataclass

TARGET_CHARS = 550
OVERLAP_CHARS = 80
MIN_CHARS = 120


@dataclass
class Chunk:
    ordinal: int
    section_path: str
    text: str
    context_text: str


def _split_paragraphs(text: str) -> list[str]:
    # numbered paragraphs ("1.", "2.1", "a)") or blank-line breaks
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _pack(paragraphs: list[str]) -> list[str]:
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= TARGET_CHARS:
            current = f"{current}\n{para}".strip()
            continue
        if len(current) >= MIN_CHARS:
            chunks.append(current)
            current = (current[-OVERLAP_CHARS:] + "\n" + para).strip()
        else:
            current = f"{current}\n{para}".strip()
        while len(current) > TARGET_CHARS * 1.5:
            chunks.append(current[:TARGET_CHARS])
            current = current[TARGET_CHARS - OVERLAP_CHARS:]
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, header: str) -> list[Chunk]:
    paragraphs = _split_paragraphs(text)
    raw_chunks = _pack(paragraphs) or [text[:TARGET_CHARS]]
    return [
        Chunk(
            ordinal=i,
            section_path=f"para_{i}",
            text=c,
            context_text=f"{header}\n\n{c}",
        )
        for i, c in enumerate(raw_chunks)
    ]


def build_header(source: str, native_id: str, issue_date: str, title: str | None) -> str:
    title = title or ""
    return f"[{source} {native_id} dated {issue_date}] {title}".strip()
