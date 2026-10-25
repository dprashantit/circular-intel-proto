"""Compose a cited answer from retrieved chunks, in the fixed shape from the
DGX plan: current position / how it evolved / open points. Then a cheap
citation check — does the claim's circular_id actually appear among what
was retrieved — flags answers that cite nothing real.
"""
import re

from . import db, llm

ANSWER_SYSTEM_PROMPT = """You are a regulatory circular assistant. Answer ONLY using the
numbered source passages given. Every factual claim must end with a citation
like [1], [2] matching the passage number. If the passages don't cover the
question, say so plainly instead of guessing.

Structure your answer as:
**Current position** — the rule as it stands today, with citations.
**How it evolved** — one or two lines on any amendment/supersession chain, if present.
**Open points** — conflicts between sources or anything not settled, if any.
"""


def format_sources(hits: list[dict], db_path: str) -> tuple[str, list[dict]]:
    with db.connect(db_path) as conn:
        formatted, refs = [], []
        for i, hit in enumerate(hits, start=1):
            circular_id = hit["metadata"].get("circular_id")
            row = conn.execute("SELECT * FROM circular WHERE circular_id=?", (circular_id,)).fetchone()
            label = f"{row['source']}/{row['native_id']} dated {row['issue_date']} (status: {row['status']})" if row else "unknown"
            text = hit.get("text") or ""
            formatted.append(f"[{i}] {label}\n{text}")
            refs.append({"index": i, "circular_id": circular_id, "label": label, "chunk_id": hit["chunk_id"]})
        return "\n\n".join(formatted), refs


def compose_answer(question: str, hits: list[dict], db_path: str) -> dict:
    sources_block, refs = format_sources(hits, db_path)
    if not refs:
        return {
            "answer": "No relevant circulars found for this question.",
            "citations": [],
            "unsupported": [],
            "ungrounded": False,
        }

    user_prompt = f"Question: {question}\n\nSource passages:\n{sources_block}"
    raw_answer = llm.chat(ANSWER_SYSTEM_PROMPT, user_prompt, max_tokens=800, temperature=0.1)

    cited_indices = {int(n) for n in re.findall(r"\[(\d+)\]", raw_answer)}
    unsupported = [i for i in cited_indices if i not in {r["index"] for r in refs}]
    # Sources were retrieved and the model didn't say "not found", but it cited
    # none of them — every claim in the answer is then unverifiable by
    # construction, not just individually wrong. A naive check that only looks
    # for bad citation marks misses this case entirely (an empty cited_indices
    # set makes `unsupported` empty too), so it needs its own flag.
    ungrounded = not cited_indices

    return {
        "answer": raw_answer,
        "citations": [r for r in refs if r["index"] in cited_indices],
        "unsupported": unsupported,  # citation marks that don't map to a real source — should stay empty
        "ungrounded": ungrounded,  # answer had sources available but cited none of them — should be False
    }
