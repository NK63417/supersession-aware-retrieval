#!/usr/bin/env python3
"""
Baseline comparison. Answers "why not just use a reranker / a date filter?" with
numbers instead of an argument. Produces the README's cross-approach table.

Runs the same labelled cases through four ranking configurations:

  (a) control        cosine similarity only — no reranking at all
  (b) temporal       cosine + exponential recency + authority  (the system under test)
  (c) llm-rerank     cosine shortlist, then an LLM scores query-document relevance
                     and re-orders by that score — the strongest available
                     *relevance* ranker, standing in for a cross-encoder
  (d) naive-recency  cosine + LINEAR min-max recency boost — the naive "isn't this
                     just a date filter?" approach

THE HYPOTHESIS TESTED — AND REJECTED: that (c) scores WORSE than (a) on current-wins,
since a superseding document spends its words negating another document and so reads
as less relevant to a plain how-to query. On the SKB, (c) instead raised current-wins
from 0% to 67%: the model recognises explicit supersession language. It failed only
where the correction lived in a different document type (a schema note) from the one
the query targets. The script still prints the verdict, so the claim stays testable.

Note on (c): a conventional cross-encoder (e.g. ms-marco MiniLM) needs torch, which
had no wheels for the Python this was built on. An LLM scoring (query, document) pairs
jointly is the same shape of computation — joint encoding rather than independent
embeddings — and a stronger relevance ranker, so it tests the hypothesis harder.
The README's results used the "fast" tier in rag.MODELS (claude-haiku-4-5), temp 0.
Config (c) needs ANTHROPIC_API_KEY; the other three run offline.

Usage:
    python baseline.py                 # all four configs on the SKB
    python baseline.py --skip-llm      # offline configs only (a, b, d)
"""
import os, json, argparse
import numpy as np
import rag
import eval as ev

SHORTLIST = 10          # candidates handed to the LLM reranker
LLM_TIER = "fast"


def _components(query):
    """cosine, recency-decay, linear-recency and authority for every chunk."""
    vecs, meta = rag.load_index()
    q = rag.embed([query])[0]
    vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    cos = vn @ (q / (np.linalg.norm(q) + 1e-9))

    ords = np.array([rag._date_ord(m.get("updated", "2026-01-01")) for m in meta], dtype=float)
    age = ords.max() - ords
    decay = 0.5 ** (age / rag.CORPUS.halflife)                     # exponential
    span = ords.max() - ords.min()
    linear = (ords - ords.min()) / span if span else np.zeros_like(ords)   # naive
    auth = np.array([1.0 if m.get("authority", 1) >= 2 else 0.0 for m in meta])
    return cos, decay, linear, auth, meta


def _llm_scores(query, cands):
    """Joint query-document relevance scores from an LLM, 0-100, one call."""
    rag.load_env()
    import anthropic
    listing = "\n\n".join(
        f"[{i+1}] ({c['source']} :: {c['heading']})\n{c['text'][:700]}"
        for i, c in enumerate(cands))
    system = ("You are a relevance ranker. Score how well each document answers the "
              "user's question, 0-100. Judge RELEVANCE ONLY — do not consider dates, "
              "recency, or which document supersedes another. Reply with JSON only: "
              '{"scores":[{"id":1,"score":87}, ...]} — one entry per document.')
    client = anthropic.Anthropic()
    resp = rag._call_with_retry(
        client, model=rag.MODELS[LLM_TIER], max_tokens=1000, temperature=0,
        system=system,
        messages=[{"role": "user", "content": f"Question: {query}\n\nDocuments:\n{listing}"}])
    txt = "".join(b.text for b in resp.content if b.type == "text")
    try:
        start, end = txt.index("{"), txt.rindex("}") + 1
        data = json.loads(txt[start:end])
        out = np.zeros(len(cands))
        for row in data.get("scores", []):
            i = int(row["id"]) - 1
            if 0 <= i < len(cands):
                out[i] = float(row["score"])
        return out
    except Exception:
        return np.full(len(cands), np.nan)      # signal failure, don't fake a ranking


def rank(query, k, mode):
    cos, decay, linear, auth, meta = _components(query)

    if mode == "control":
        scores = cos
    elif mode == "temporal":
        scores = cos + rag.CORPUS.w_rec * decay + rag.CORPUS.w_auth * auth
    elif mode == "naive-recency":
        scores = cos + rag.CORPUS.w_rec * linear
    elif mode == "llm-rerank":
        top = np.argsort(-cos)[:SHORTLIST]
        cands = [meta[i] for i in top]
        s = _llm_scores(query, cands)
        if np.isnan(s).all():
            raise SystemExit("LLM reranker returned no usable scores")
        order = top[np.argsort(-s)]
        return [meta[i]["source"] for i in order[:k]]
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    return [meta[i]["source"] for i in np.argsort(-scores)[:k]]


def score_config(cases, k, mode):
    hits, rr, sup_t, sup_w = 0, 0.0, 0, 0
    for c in cases:
        ranked = rank(c["query"], k, mode)
        r = ev.rank_of(ranked, c["expect"])
        hits += r is not None
        rr += (1.0 / r) if r else 0.0
        if c.get("stale"):
            sup_t += 1
            # `stale` may name several superseded docs — normalise, as eval.py does,
            # so both harnesses score current-wins identically.
            stale = c["stale"] if isinstance(c["stale"], list) else [c["stale"]]
            sr = ev.rank_of(ranked, stale)
            sup_w += bool(r and (sr is None or r < sr))
    n = len(cases)
    return {"recall": hits / n, "mrr": rr / n,
            "cw": (sup_w / sup_t) if sup_t else None, "cw_n": sup_t, "cw_w": sup_w}


def main():
    ap = argparse.ArgumentParser(description="Baseline comparison across ranking configurations")
    ap.add_argument("--corpus", default="support")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--skip-llm", action="store_true")
    args = ap.parse_args()

    rag.use_corpus(args.corpus)
    cases = ev.load_cases(ev.default_cases_path(args.corpus))

    modes = ["control", "temporal", "naive-recency"] + ([] if args.skip_llm else ["llm-rerank"])
    labels = {"control": "(a) cosine only — no rerank",
              "temporal": "(b) cosine + temporal rerank",
              "llm-rerank": "(c) LLM relevance rerank",
              "naive-recency": "(d) cosine + linear recency"}

    print(f"\n  BASELINE — corpus '{args.corpus}', {len(cases)} cases @ k={args.k}")
    print("  " + "─" * 68)
    print(f"  {'configuration':<32}{'recall@5':>10}{'MRR':>8}{'current-wins':>18}")
    print("  " + "─" * 68)
    results = {}
    for m in modes:
        r = score_config(cases, args.k, m)
        results[m] = r
        cw = "—" if r["cw"] is None else f"{r['cw']:.0%}  ({r['cw_w']}/{r['cw_n']})"
        print(f"  {labels[m]:<32}{r['recall']:>9.0%}{r['mrr']:>8.3f}{cw:>18}")
    print("  " + "─" * 68)

    if "llm-rerank" in results:
        a, c = results["control"]["cw"], results["llm-rerank"]["cw"]
        if a is not None and c is not None:
            verdict = ("CONFIRMED — a better relevance ranker is WORSE at supersession"
                       if c < a else
                       "NOT confirmed — the LLM reranker matched or beat the control")
            print(f"\n  PREDICTION [(c) worse than (a) on current-wins]: {verdict}")
            print(f"    control {a:.0%}  vs  LLM rerank {c:.0%}")
    print()


if __name__ == "__main__":
    main()
