# Supersession-Aware Retrieval

**Why retrieval returns the superseded fix — and the cheapest signal that stops it.**

Retrieval systems are measured on whether they find a *relevant* document, not on whether
that document is *still correct*. In an operational knowledge base those diverge. A
temporary fix recorded in a resolved incident stays retrievable long after a permanent fix
replaces it — nobody reopens a closed ticket to say its workaround is now wrong. And the
superseded record is usually the *better* match to the question an engineer asks.

A stale date is an inconvenience; a stale procedure is a production incident.

This is an experiment, not a new architecture: failure scenarios, studied with a metric
chosen to expose them, to find where existing ranking methods break and what fixes it.

## See it

One question, two rankings, from `python rag.py compare "…"`:

```
  how do I fix an invoice stuck in PENDING?

     similarity only                            similarity + temporal-authority
  1. inc-3312.md (2026-03-04)                   rb-001-stuck-invoice-v2.md (2026-07-06)
  2. rb-001-stuck-invoice.md (2026-02-18)       inc-3312.md (2026-03-04)
  3. lookup-queries.md (2026-01-22)             inc-4890.md (2026-06-19)
  4. inc-4890.md (2026-06-19)                   CURRENT.md (2026-08-14)
  5. rb-001-stuck-invoice-v2.md (2026-07-06)    rb-001-stuck-invoice.md (2026-02-18)
```

On the left, the top two results are a closed ticket that applied the old data fix and the
original runbook that prescribes it. Both are superseded: on a patched system that fix posts
the invoice twice and corrupts the ledger. The current runbook comes **last**. On the right it comes first. Nothing
about the documents changed, only the ranking.

## The result

Four ranking approaches, the same labelled questions, the same indexes, `k = 5`. Only the
ranking function changes.

| Ranking | SKB recall@5 | SKB MRR | SKB current-wins | Notes recall@5 | Notes MRR | Notes current-wins | Model call per query |
|---|---|---|---|---|---|---|---|
| Similarity only | 88% | 0.667 | 0% | 86% | 0.720 | 0% | no |
| LLM relevance rerank | 100% | 0.938 | 67% | — | — | — | **yes** |
| Linear date boost | 100% | **1.000** | 100% | 93% | 0.792 | **0%** | no |
| **Similarity + temporal-authority rerank** | 100% | 0.938 | **100%** | 93% | 0.798 | **100%** | no |

**Current-wins** is the metric that matters here. When both a superseded document and its
replacement are retrieved, does the current one rank higher? Recall and MRR are both
satisfied by retrieving the superseded document; current-wins is not. Similarity-only
retrieval scores 88% recall on the SKB while failing every supersession case.

What the experiment found:

- **Relevance finds a correction only when the correction looks like an answer.** A correcting
  document works through negation ("do not restart the service") and reference to another
  document ("replaces the v1 procedure"). It doesn't answer the question; it invalidates the
  document that does. I call this *second-level inference*. It costs the correction a
  measurable similarity penalty: on the SKB the current document trails the one it supersedes
  by **0.081** and **0.156** cosine.
- **I predicted an LLM reranker would make this worse. It didn't.** Current-wins went from
  0% to 67%, stable across three runs. The model handles corrections that say plainly what
  they replace. It misses the one where the fix lives in a different document type: a schema
  note recording that a field is deprecated loses to the lookup procedure that filters on that
  field, because the procedure *is* the better answer to the question as asked. Relevance
  ranking can't tell those two cases apart from inside the relevance objective.
- **A perfect score on one corpus means nothing.** The linear date boost scores a perfect
  1.000 MRR on the small corpus and 0% current-wins on the larger one.
- **Tuned weights do not transfer.** Tuned on one corpus, they moved the other's top score by 0.001.
- **The weights held as the corpus changed.** Tuned on the notes corpus on 16 July and never
  retuned, current-wins stayed at 100% through 18 September, by which point 28 of 82
  documents had been edited and the corpus had grown from 575 to 674 chunks.
- **Evaluation labels decay too.** The one recall drop in September was the retriever being
  right: a question's best answer had moved to a page created after the label was written.
  The answer key had been superseded. Both the original and corrected results are reported
  below.

## Quickstart

Tested on Python 3.14. No GPU and no torch — embeddings run locally.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python rag.py index        # embeds the SKB (downloads a ~30 MB model on first run)
python eval.py             # reproduces the SKB rows: no-rerank vs temporal rerank
python baseline.py --skip-llm   # three of the four approaches, offline
python rag.py compare "how do I look up pending invoices for an account?"
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
([`potion-base-8M`](https://huggingface.co/minishlab/potion-base-8M), 256-d), so no corpus
content leaves the machine. Each chunk inherits its document's last-updated date and an
authority flag. Retrieval is exhaustive cosine similarity, then

```
score = cosine(q, d) + w_rec · 2^(−age(d) / h) + w_auth · [d is an authority doc]
```

where `age` is measured in days from the newest document in the corpus, `h` is the half-life,
and authority docs (e.g. `CURRENT.md`) are current-state documents that override ordinary
pages. The authority term doesn't depend on dates, so it settles ties where a current-state
document and an ordinary page carry similar timestamps. Weights come from a grid search
against labelled questions: highest recall, subject to current-wins at 100%.

Answers are cite-or-refuse: every claim cites a retrieved chunk, or the system returns the
corpus's refusal string.

**Why exponential decay and not a date sort.** Linear normalisation is anchored to the
*oldest* document, so a long tail of old material squeezes every recent date together. A page
superseded in June and the ruling that replaced it in July end up with nearly the same
recency value, which is exactly how the linear boost scores 0% on the larger corpus.
Exponential decay is anchored to the newest document and doesn't compress.

A date *filter* fails worse. It throws away old documents that are still correct (most of a
stable corpus is old and correct), and it can't order what survives, which hands the problem
straight back to the relevance ranking that caused it. And when documents get edited for
grammar or formatting, modification date stops meaning supersession at all.

## Using your own corpus

Copy `corpora.example.json` to `corpora.json` and describe your documents. Then write
labelled questions to `eval.<name>.local.jsonl` (the schema is in `eval.py`) and tune:

```bash
python rag.py --corpus mine index
python eval.py --corpus mine
```

Two rules of thumb, from two corpora — treat them as starting points, not laws:

- **Half-life tracks how often the corpus is revised.** A document should read as half as
  authoritative at roughly the interval its corpus publishes a newer version. Weekly notes:
  21 days. Procedures that turn over across months: 120 days.
- **The recency weight must exceed the similarity penalty on correcting documents.** A weight
  below that penalty can't promote the correction at any half-life. The SKB needs
  `w_rec = 0.40` against penalties up to 0.156; the notes corpus needs 0.10.

Do not reuse the SKB's weights. Measure your own. The stronger the recency weight your corpus
needs, the more it depends on its dates being accurate.

## Related work

Retrieval-augmented generation grounds a model in retrieved passages
([Lewis et al., 2020](https://arxiv.org/abs/2005.11401)). Grounding constrains the model to its
context; it doesn't establish that the context is current. A system can produce a fully cited
answer where every citation is real and every citation is superseded.

Lexical and dense retrievers ([BM25](https://doi.org/10.1561/1500000019),
[Sentence-BERT](https://arxiv.org/abs/1908.10084)) and cross-encoder rerankers
([Nogueira & Cho, 2019](https://arxiv.org/abs/1901.04085)) all optimise relevance, which is the
assumption tested here. Vector databases expose timestamp filters and recency boosts, with
little guidance on what to set them to.

[GraphRAG](https://arxiv.org/abs/2404.16130) builds an entity graph that could represent a
`supersedes` edge, at the cost of an LLM extraction pass repeated as the corpus changes.
[Zep's Graphiti](https://arxiv.org/abs/2501.13956) is the closest prior work: it keeps validity
intervals on extracted facts and invalidates a fact when a newer one contradicts it.

Those systems handle supersession once a contradiction has been detected. This experiment sits
upstream of that: given an unmodified corpus, when does relevance ranking fail to retrieve the
correction in the first place, and what is the cheapest signal that fixes it?

## Limitations

1. **Recency comes from a last-updated field, not change tracking.** A copy-edit to a
   superseded document makes it look fresh, and the stronger the recency weight, the bigger
   the error. Deriving recency from version history is the main piece of future work.
2. **Two corpora, one of them synthetic.** The SKB was generated with an LLM and its
   supersession cases were planted. The organic corpus offsets that partly, but two is a
   small basis for the tuning rules.
3. **The stability result covers two months,** not a year.
4. **A correction with no date, or dated earlier than what it corrects, is invisible** to this
   method. It is only as good as the metadata.
5. **One labelled question is unretrieved in every configuration:** a vaguely worded query
   that shares little vocabulary with its answer. Hybrid lexical + dense retrieval is the
   indicated fix; it isn't implemented.
6. **Tuning needs labelled questions per corpus,** which many teams don't maintain. That is
   the real adoption cost.
7. **It's a heuristic, not comprehension.** It approximates "newer means more authoritative",
   which is false for corpora that mix living procedures with timeless reference material.
   Global recency will over-promote a recently edited but tangential document. Scoping the
   recency term to a similarity band is the indicated refinement; it isn't implemented.
8. **Decision support, not a decision-maker.** In real use I run the same checks I expect the
   method to run — whether it got the overall context right, and where it went off if it didn't.

## What's here

| | |
|---|---|
| `rag.py` | engine: indexing, temporal-authority retrieval, side-by-side `compare`, cite-or-refuse answering |
| `eval.py` | recall@k, MRR and current-wins, with and without reranking |
| `baseline.py` | the four-approach comparison |
| `corpora/support/` | the synthetic knowledge base: 4 runbooks, 3 resolved tickets, 2 schema references, 1 current-state doc |
| `eval.support.jsonl` | 8 labelled questions, 3 of them supersession cases |

**On the data.** Two corpora, on purpose: controlled to study the method, organic to test it
in a real environment.

- **The SKB (controlled)** describes a fictional billing product. Real incident data carries
  company liability, so I had Claude generate this one from a description of my L2 support
  workflow — runbooks, resolved tickets and schema references, the three things I actually
  reason from — and its supersession cases were planted deliberately.
- **The notes corpus (organic)** is my personal notes, revised weekly over nine months: 76
  documents, 575 chunks, 14 labelled questions, one supersession case nobody planted (a
  scheduling decision was revised and an older page kept the superseded date). It is
  personal, so it isn't published. Its numbers are reported here but can't be reproduced
  from this repository.

**The label correction, both ways (notes corpus, 18 September):**

| | recall@5 | MRR | current-wins |
|---|---|---|---|
| original label | 86% | 0.821 | 100% |
| corrected label | 93% | 0.893 | 100% |

Changing an answer key after seeing results deserves suspicion, so both are here. All 14
labels were audited; that was the only change.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
