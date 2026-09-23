#!/usr/bin/env python3
"""
Retrieval eval harness — turn "is retrieval better?" from a vibe into a number.

Runs a set of labelled questions through `retrieve()` and scores how well the
*correct* source page is surfaced. Two things are measured, because this RAG has
two distinct jobs:

  1. FIND IT       — is a correct page in the top-k at all?   (recall@k, MRR)
  2. PREFER TRUTH  — when the corpus contains a stale answer AND a current one,
                     does the CURRENT page outrank the stale one?  (current-wins)

(2) is the whole point of the temporal-authority direction. A RAG that finds the
right page but ranks a superseded page above it will still hand the model a
confidently-citable wrong answer — the "a citation is not a fact" failure.

Cases live in a JSONL file per corpus. The published SKB cases are eval.support.jsonl.
For a private corpus, name the file eval.<corpus>.local.jsonl — the .gitignore keeps
those out of version control.

Case schema (one JSON object per line; lines starting with # are comments):
    {
      "id":     "refund-window",
      "query":  "how long is the return window",
      "expect": ["docs/policy-current.md", "CURRENT.md"],   # any of these = a hit
      "stale":  ["docs/policy-2023.md"],   # OPTIONAL: pages that must NOT outrank expect
      "note":   "supersession case: an old policy page still in the corpus"   # OPTIONAL
    }

Usage:
    python eval.py                     # SKB cases, no-rerank vs rerank, k=5
    python eval.py -k 3                # tighter window (exposes ranking fragility)
    python eval.py --corpus mine       # your own corpus profile
    python eval.py --cases path.jsonl  # explicit case file
"""
import os, sys, json, argparse
import rag   # reuse the exact retrieve() under test


def default_cases_path(corpus):
    """Published cases first, then a private local file, then the legacy name."""
    for name in (f"eval.{corpus}.jsonl", f"eval.{corpus}.local.jsonl", "eval.local.jsonl"):
        path = os.path.join(rag.HERE, name)
        if os.path.exists(path):
            return path
    return os.path.join(rag.HERE, f"eval.{corpus}.jsonl")   # so the error names the expected file


def load_cases(path):
    if not os.path.exists(path):
        raise SystemExit(f"No eval cases at {path} — see the schema in this file's docstring.")
    cases = []
    for i, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{i}: bad JSON — {e}")
    return cases


def rank_of(sources, targets):
    """1-based rank of the first source in `targets`, or None if absent."""
    tset = set(targets)
    for i, s in enumerate(sources, 1):
        if s in tset:
            return i
    return None


def run(cases, k, weighted, quiet=False):
    rows, hits, rr_sum = [], 0, 0.0
    sup_total, sup_win = 0, 0

    for c in cases:
        ranked = [h["source"] for _, h in rag.retrieve(c["query"], k, weighted=weighted)]
        r = rank_of(ranked, c["expect"])
        hit = r is not None
        hits += hit
        rr_sum += (1.0 / r) if r else 0.0

        # supersession check: does a correct page outrank the stale page?
        verdict = "✅" if hit else "❌ MISS"
        if c.get("stale"):
            sup_total += 1
            # `stale` may name several superseded docs — a real supersession case
            # often has more than one (an old runbook AND the ticket that applied it).
            # The current doc must beat the BEST-ranked of them, not just one.
            stale = c["stale"] if isinstance(c["stale"], list) else [c["stale"]]
            stale_rank = rank_of(ranked, stale)
            # current-wins iff a correct page is present AND ranks above the stale one
            # (stale absent from top-k counts as a win — the trap never fires)
            win = hit and (stale_rank is None or r < stale_rank)
            sup_win += win
            verdict = "✅ current-wins" if win else "⚠️ STALE-WINS"
        rows.append((c["id"], "sup" if c.get("stale") else "fact", r, verdict))

    n = len(cases)
    if not quiet:
        label = "weighted (recency+authority)" if weighted else "semantic-only"
        print(f"\n  RETRIEVAL EVAL — {n} cases @ k={k} — {label}\n  " + "─" * 46)
        print(f"  {'id':<22}{'type':<6}{'rank':<6}verdict")
        for cid, typ, r, v in rows:
            print(f"  {cid:<22}{typ:<6}{str(r or '—'):<6}{v}")
        print("  " + "─" * 46)
        print(f"  recall@{k}     {hits}/{n}   ({hits/n:.0%})")
        print(f"  MRR          {rr_sum/n:.3f}")
        if sup_total:
            print(f"  current-wins {sup_win}/{sup_total}   ({sup_win/sup_total:.0%})")
        print()
    return {"recall": hits / n, "mrr": rr_sum / n,
            "current_wins": (sup_win / sup_total) if sup_total else None}


def compare(cases, k):
    before = run(cases, k, weighted=False)
    after = run(cases, k, weighted=True)
    def pct(x): return "—" if x is None else f"{x:.0%}"
    print(f"  BEFORE → AFTER  @ k={k}   (semantic-only → +recency/authority)\n  " + "─" * 46)
    print(f"  recall@{k}     {pct(before['recall'])}  →  {pct(after['recall'])}")
    print(f"  MRR          {before['mrr']:.3f}  →  {after['mrr']:.3f}")
    print(f"  current-wins {pct(before['current_wins'])}  →  {pct(after['current_wins'])}   ← the temporal-authority metric")
    print()


def main():
    ap = argparse.ArgumentParser(description="Retrieval eval — recall@k, MRR, and current-wins")
    ap.add_argument("--corpus", default=rag.DEFAULT_CORPUS,
                    help="corpus profile to evaluate (default: %(default)s)")
    ap.add_argument("--cases", default=None,
                    help="labelled cases (default: eval.<corpus>.jsonl, "
                         "then eval.<corpus>.local.jsonl)")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--mode", choices=["compare", "semantic", "weighted"], default="compare")
    args = ap.parse_args()

    rag.use_corpus(args.corpus)
    cases = load_cases(args.cases or default_cases_path(args.corpus))
    if args.mode == "compare":
        compare(cases, args.k)
    else:
        run(cases, args.k, weighted=(args.mode == "weighted"))


if __name__ == "__main__":
    main()
