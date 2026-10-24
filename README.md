# circular-intel-proto

Colab prototype of the Regulatory Circular Intelligence Assistant — the
free-tier, local-everything precursor to the DGX Spark production build.
See `colab_prototype_plan.md` in the parent repo for the full design
rationale and the mapping from each prototype choice to its DGX equivalent.

## Quickstart (Google Colab)

1. Open `notebooks/01_prototype.ipynb` in Colab.
2. `Runtime → Change runtime type → T4 GPU`.
3. Run the setup cell — it mounts Drive, clones this repo (if not already
   present), installs `requirements.txt`, and downloads the Qwen2.5-7B-Instruct
   GGUF (Q4_K_M, ~4.5GB) into Drive on first run only.
4. Upload your sample circular PDFs to `MyDrive/circular-intel-proto/data/raw/<SOURCE>/`
   (or run the provided cell to unzip `circulars_filtered.zip` there).
5. Run the pipeline cells in order: register → parse → chunk+embed → relations → status.
6. Run the last cell to launch the Gradio chat UI.

## Project layout

```
circular-intel-proto/
├── config/sources.yaml   per-source filename/ID/citation regex patterns
├── src/
│   ├── db.py             SQLite register + relation graph
│   ├── parse.py          PDF -> text + cover-page metadata (PyMuPDF)
│   ├── chunk.py           structure-aware chunker
│   ├── embed.py           bge-small-en-v1.5 embeddings -> Chroma
│   ├── relations.py       regex citation extraction + LLM relation typing + status
│   ├── llm.py             local GGUF model wrapper (llama-cpp-python)
│   ├── retrieve.py        dense + BM25 retrieval, RRF merge, "latest only" graph expansion
│   └── answer.py          cited answer composition + unsupported-citation check
├── notebooks/01_prototype.ipynb   orchestrates everything end-to-end + Gradio UI
└── eval/golden_qa.jsonl   hand-written Q&A for sanity-checking retrieval/answers
```

All data (`data/`, `db/*.sqlite`, `chroma_db/`, model weights) lives on
Google Drive, not in this repo — see `.gitignore`. Re-running the notebook
on a fresh Colab session just needs Drive remounted; nothing is reprocessed
unless you delete the Drive folders.

## Status

Prototype — validated against 19 sample NSE/BSE circulars. Known gap: the
sample set is mostly repetitive "mock trading" notices with no real
supersession chains, so the status-computation logic (`relations.compute_status`)
hasn't been tested against a real amendment chain yet. Add a few SEBI master
circulars before trusting that part.
