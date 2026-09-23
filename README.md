# Supersession-Aware Retrieval

**Second-Level Inference: Why Retrieval Returns the Superseded Fix** — code, data and
evaluation for the paper. 📄 [Paper (PDF)](paper.pdf)
<!-- Demo link goes here once deployed. -->

Retrieval systems are measured on whether they find a *relevant* document, not on whether
that document is *still correct*. In an operational knowledge base those diverge. A
temporary fix recorded in a resolved incident stays retrievable long after a permanent fix
replaces it — nobody reopens a closed ticket to say its workaround is now wrong. And the
superseded record is usually the *better* match to the question an engineer asks.

A stale date is an inconvenience; a stale procedure is a production incident.

## The result

Four ranking approaches, the same labelled questions, `k = 5`. **Current-wins** asks: when
both a superseded document and its replacement are retrieved, does the current one rank higher?

| Ranking | SKB current-wins | Notes corpus current-wins | Model call per query |
|---|---|---|---|
| Similarity only | 0% | 0% | no |
| LLM relevance rerank | 67% | — | **yes** |
| Linear date boost | 100% | **0%** | no |
| **Similarity + temporal-authority rerank** | **100%** | **100%** | no |

Three findings:

- **Relevance finds a correction only when the correction looks like an answer.** A correcting
  document works through negation and reference to another document — *second-level
  inference*. On the synthetic knowledge base it trails the document it supersedes by **0.081**
  and **0.156** cosine. An LLM reranker recovers the case where the fix is a revised runbook,
  and misses the one where the fix is a schema note.
- **A perfect score on one corpus means nothing.** The linear date boost scores a perfect
  1.000 MRR on the small corpus and 0% current-wins on the larger one.
- **Tuned weights do not transfer.** Tuned on one corpus, they moved the other's top score by 0.001.

This is an experiment, not a new architecture. See the paper for method, related work and
limitations.

## Quickstart

Tested on Python 3.14. No GPU and no torch — embeddings run locally.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python rag.py index        # embeds the SKB (downloads a ~30 MB model on first run)
python eval.py             # reproduces the SKB rows: no-rerank vs temporal rerank
python baseline.py --skip-llm   # three of the four approaches, offline
```

Expected from `eval.py`:

```
  recall@5     88%  →  100%
  MRR          0.667  →  0.938
  current-wins 0%  →  100%
```

The LLM reranker row and grounded answers need an Anthropic API key:

```bash
cp .env.example .env       # then add your key
python baseline.py         # all four approaches
python rag.py ask "how do I fix an invoice stuck in PENDING?"
```

## How it works

Documents are chunked by heading and embedded locally
([`potion-base-8M`](https://huggingface.co/minishlab/potion-base-8M), 256-d). Each chunk
inherits its document's last-updated date and an authority flag. Ranking is

```
score = cosine(q, d) + w_rec · 2^(−age(d) / h) + w_auth · [d is an authority doc]
```

where `age` is measured in days from the newest document, `h` is the half-life, and authority
docs (e.g. `CURRENT.md`) override ordinary pages. Answers are cite-or-refuse: every claim cites
a retrieved chunk, or the system returns the corpus's refusal string.

Exponential decay matters. Linear normalisation is anchored to the *oldest* document, so a long
tail of old material squeezes every recent date together — which is exactly how the linear
boost scores 0% on the larger corpus.

## Using your own corpus

Copy `corpora.example.json` to `corpora.json` and describe your documents. Then write
labelled questions to `eval.<name>.local.jsonl` (the schema is in `eval.py`) and tune:

```bash
python rag.py --corpus mine index
python eval.py --corpus mine
```

Two rules of thumb from the paper, from two corpora:

- **Half-life tracks how often the corpus is revised.** Weekly notes: 21 days. Procedures
  that turn over across months: 120 days.
- **The recency weight must exceed the similarity penalty on correcting documents.** The SKB
  needs `w_rec = 0.40`; the personal-notes corpus needs 0.10.

Do not reuse the SKB's weights. Measure your own.

## What's here

| | |
|---|---|
| `rag.py` | engine: indexing, temporal-authority retrieval, cite-or-refuse answering |
| `eval.py` | recall@k, MRR and current-wins, with and without reranking |
| `baseline.py` | the four-approach comparison |
| `corpora/support/` | the synthetic knowledge base: 4 runbooks, 3 resolved tickets, 2 schema references, 1 current-state doc |
| `eval.support.jsonl` | 8 labelled questions, 3 of them supersession cases |

**On the data.** The SKB describes a fictional billing product. It was generated with Claude
from a description of the author's L2 support workflow, and its supersession cases were planted
deliberately — a controlled corpus to study the method. The second corpus — "the vault" in the
paper — is the author's personal notes, an organic corpus where nobody planted anything.
It is not published, so its results are reported but cannot be reproduced from this repository.

## Limitations

In short: recency comes from a last-updated field rather than change tracking, so a copy-edit
can make a stale document look fresh; two corpora is a small basis for the tuning rules; a
correction without a date is invisible; and weights need labelled questions per corpus. The
full list is in the paper, §8. Treat the output as decision support — check it before acting on it.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
