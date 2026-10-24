"""Two-pass relation extraction:
1. regex finds candidate citations in a circular's text and resolves them
   against the register (same source, matching native_id).
2. for each resolved citation, ask the local LLM to label the relation
   type + confidence from the citing sentence's context.

Unresolved citations (reference to a circular not in the local corpus) are
still logged with to_circular=NULL — that's the review-queue case from the
DGX plan, not a bug to silently swallow.
"""
import re

from . import db, llm

RELATION_TYPES = [
    "amends", "supersedes", "rescinds", "partially_supersedes",
    "consolidated_into", "clarifies", "extends_timeline", "implements", "refers_to",
]

RELATION_SYSTEM_PROMPT = (
    "You classify how one regulatory circular refers to another, based only on the "
    "citing sentence given. Respond with compact JSON: "
    '{"relation": one of ' + str(RELATION_TYPES) + ', "confidence": 0.0-1.0}. '
    "If the sentence only points the reader elsewhere without changing or replacing "
    "anything, use refers_to."
)


def _context_window(text: str, match_start: int, match_end: int, window: int = 200) -> str:
    return text[max(0, match_start - window): match_end + window]


def find_candidate_citations(text: str, citation_regex: str) -> list[re.Match]:
    return list(re.finditer(citation_regex, text))


def resolve_citation(conn, source: str, raw_native_id: str, exclude_circular_id: str):
    rows = db.find_circular_by_native_id(conn, source, raw_native_id)
    rows = [r for r in rows if r["circular_id"] != exclude_circular_id]
    return rows[0] if rows else None


def classify_relation(citing_sentence: str) -> dict:
    result = llm.chat_json(
        RELATION_SYSTEM_PROMPT,
        f"Citing sentence: {citing_sentence}",
        max_tokens=100,
    )
    relation = result.get("relation")
    if relation not in RELATION_TYPES:
        relation = "refers_to"
    confidence = result.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    return {"relation": relation, "confidence": float(confidence)}


def extract_relations_for_circular(conn, circular_id: str, source: str, text: str, sources_cfg: dict) -> int:
    cfg = sources_cfg["sources"].get(source, {})
    citation_regex = cfg.get("citation_regex")
    if not citation_regex:
        return 0

    own_native_id_hits = {m.group(0) for m in find_candidate_citations(text, citation_regex)}
    n_found = 0
    for raw_ref in own_native_id_hits:
        match = re.search(re.escape(raw_ref), text)
        sentence = _context_window(text, match.start(), match.end())

        target = resolve_citation(conn, source, raw_ref, circular_id)
        to_circular = target["circular_id"] if target else None

        labeled = classify_relation(sentence)
        db.insert_relation(
            conn,
            from_circular=circular_id,
            to_circular=to_circular,
            raw_reference=raw_ref,
            relation=labeled["relation"],
            confidence=labeled["confidence"],
            method="regex+llm",
        )
        n_found += 1
    return n_found


def compute_status(conn, circular_id: str) -> None:
    """A circular is superseded/rescinded if a later edge says so, consolidated
    if folded into a master, else in_force. Trivial at prototype scale — no
    need for the DGX plan's confidence-threshold review queue yet, but the
    same rule applies."""
    incoming = db.relations_into(conn, circular_id)
    status, reason = "in_force", None
    for edge in incoming:
        if edge["relation"] in ("supersedes", "rescinds") and (edge["confidence"] or 0) >= 0.6:
            status = "superseded" if edge["relation"] == "supersedes" else "rescinded"
            reason = f"{edge['relation']} by {edge['from_circular']} (ref: {edge['raw_reference']})"
        elif edge["relation"] == "partially_supersedes" and status == "in_force":
            status = "partially_superseded"
            reason = f"partially superseded by {edge['from_circular']}"
        elif edge["relation"] == "consolidated_into" and status == "in_force":
            status = "consolidated"
            reason = f"consolidated into {edge['from_circular']}"
    conn.execute(
        "UPDATE circular SET status=?, status_reason=? WHERE circular_id=?",
        (status, reason, circular_id),
    )
