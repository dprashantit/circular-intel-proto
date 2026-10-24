"""Dense (Chroma) + keyword (BM25) retrieval, merged by reciprocal-rank
fusion, then expanded along circular_relation to pull in successors and
drop superseded/rescinded hits — the "latest only" step.
"""
from rank_bm25 import BM25Okapi

from . import db, embed


def build_bm25_index(chroma_dir: str):
    collection = embed.get_collection(chroma_dir)
    all_docs = collection.get(include=["documents", "metadatas"])
    ids = all_docs["ids"]
    texts = all_docs["documents"]
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, ids, texts, all_docs["metadatas"]


def bm25_search(bm25, ids, texts, metadatas, query: str, top_k: int = 20) -> list[dict]:
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {"chunk_id": ids[i], "text": texts[i], "metadata": metadatas[i], "score": float(scores[i])}
        for i in ranked
    ]


def reciprocal_rank_fusion(*ranked_lists: list[dict], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
            items[cid] = item
    merged = sorted(items.values(), key=lambda it: scores[it["chunk_id"]], reverse=True)
    return merged


def expand_to_latest(conn, hits: list[dict], include_history: bool = False) -> list[dict]:
    """For each hit, drop it if its circular is superseded/rescinded (unless
    history was asked for) and pull in the successor's chunks instead."""
    expanded, seen_circulars = [], set()
    for hit in hits:
        circular_id = hit["metadata"].get("circular_id")
        row = conn.execute(
            "SELECT * FROM circular WHERE circular_id=?", (circular_id,)
        ).fetchone()
        if row is None:
            expanded.append(hit)
            continue

        if row["status"] in ("superseded", "rescinded") and not include_history:
            successors = conn.execute(
                "SELECT * FROM circular_relation WHERE to_circular=? AND relation IN "
                "('supersedes','rescinds','consolidated_into')",
                (circular_id,),
            ).fetchall()
            for succ in successors:
                if succ["from_circular"] not in seen_circulars:
                    seen_circulars.add(succ["from_circular"])
                    succ_chunks = conn.execute(
                        "SELECT chunk_id FROM chunk_meta WHERE circular_id=? ORDER BY ordinal LIMIT 3",
                        (succ["from_circular"],),
                    ).fetchall()
                    for c in succ_chunks:
                        expanded.append({
                            "chunk_id": c["chunk_id"],
                            "text": None,  # filled in by caller from Chroma if needed
                            "metadata": {"circular_id": succ["from_circular"]},
                            "score": hit["score"],
                            "via_expansion_from": circular_id,
                        })
            continue  # drop the superseded hit itself

        expanded.append(hit)
        seen_circulars.add(circular_id)
    return expanded


def retrieve(db_path: str, chroma_dir: str, query: str, top_k: int = 15, include_history: bool = False):
    dense_hits = embed.dense_search(chroma_dir, query, top_k=top_k * 2)
    bm25, ids, texts, metadatas = build_bm25_index(chroma_dir)
    keyword_hits = bm25_search(bm25, ids, texts, metadatas, query, top_k=top_k * 2)
    merged = reciprocal_rank_fusion(dense_hits, keyword_hits)

    with db.connect(db_path) as conn:
        expanded = expand_to_latest(conn, merged, include_history=include_history)

    return expanded[:top_k]
